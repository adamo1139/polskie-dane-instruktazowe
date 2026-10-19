"""Top-k logprobów nauczyciela (HF transformers) dla każdego tokenu cache SFT poziomka-sft-v2.

Do off-policy KD: nauczyciel czyta gotowe rekordy cache (teacher forcing, te same ID tokenów co
w treningu ucznia, bez ponownej tokenizacji) i dla każdej pozycji zapisuje swoje k najbardziej
prawdopodobnych następnych tokenów. Jeden forward na paczkę rekordów: rekordy są sklejone w jedną
sekwencję, a position_ids zaczynają się od 0 w każdym rekordzie, więc flash-attention (varlen)
nie miesza rozmów. Na starcie każdy proces sprawdza to, porównując paczkę z osobnymi forwardami.
lm_head + log_softmax + topk liczone na GPU w kawałkach (pełne logity paczki nie mieszczą się).

Układ wyniku — pliki obok siebie, wiersz j odpowiada tokenowi j w {prefix}.tokens.bin shardu
cache (te same offsety):
  {prefix}.topk_ids.bin    <u2 [tokens, k]  ID k najlepszych tokenów, malejąco
  {prefix}.topk_logp.bin   <f2 [tokens, k]  ich log p nauczyciela (T=1, pełny słownik)
  {prefix}.target_logp.bin <f2 [tokens]     log p nauczyciela dla tokenu, który tam faktycznie stoi
Wiersz j to rozkład p(token j | tokeny rekordu przed j). Wiersz BOS (pierwszy token rekordu)
nie ma predykcji i zostaje zerami — maska cache i tak nie ma tam lossu. Masa ogona to
1 - sum(exp(topk_logp)). W treningu: tokens = ids[s:e-1], labels = ids[s+1:e], więc labelom
odpowiadają wiersze s+1..e-1.

Praca dzielona między procesy (jeden na GPU): każdy shard cache jest cięty na --world kawałków
o podobnej liczbie tokenów, proces --rank liczy swój kawałek każdego shardu i pisze w swój zakres
wierszy wspólnych plików. Zakończony kawałek dostaje plik .done (statystyki), więc po przerwaniu
wystarczy puścić to samo jeszcze raz. --finalize sprawdza komplet i pisze manifest.json.

  # benchmark: 1 GPU, bez zapisu
  CUDA_VISIBLE_DEVICES=0 python teacher_topk.py --cache C --model M --out O --benchmark 2000
  # pełny przebieg: run_teacher_topk.sh
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

FORMAT = "teacher-topk-v1"
CACHE_FORMAT = "poziomka-sft-v2"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cache(cache):
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["format"] != CACHE_FORMAT:
        raise SystemExit(f"{cache}: format {manifest['format']}, oczekiwano {CACHE_FORMAT}")
    return manifest, [s for s in manifest["shards"] if s["records"]]


def chunk_bounds(offsets, world):
    """Granice kawałków (w rekordach) o podobnej liczbie tokenów, po jednym na rank."""
    targets = np.linspace(0, int(offsets[-1]), world + 1)[1:-1]
    cuts = np.searchsorted(offsets, targets).tolist()
    return [0] + cuts + [len(offsets) - 1]


def out_paths(out, prefix):
    base = out / prefix
    return {name: Path(f"{base}.{name}.bin") for name in ("topk_ids", "topk_logp", "target_logp")}


def open_outputs(out, shard, k):
    """Wspólne pliki shardu, otwierane przez wszystkie ranki; każdy pisze tylko swoje wiersze."""
    n = shard["tokens"]
    spec = {"topk_ids": ("<u2", (n, k)), "topk_logp": ("<f2", (n, k)), "target_logp": ("<f2", (n,))}
    maps = {}
    for name, path in out_paths(out, shard["prefix"]).items():
        dtype, shape = spec[name]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            # ftruncate do tej samej długości jest idempotentny, więc kolejność ranków nie ma znaczenia.
            if os.fstat(fd).st_size not in (0, size):
                raise SystemExit(f"{path}: zły rozmiar {os.fstat(fd).st_size}, oczekiwano {size}")
            os.ftruncate(fd, size)
        finally:
            os.close(fd)
        maps[name] = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    return maps


def batches(offsets, first, last, tokens_per_batch):
    """Kolejne rekordy [first, last) w paczkach o <= tokens_per_batch pozycjach wejścia."""
    batch, used = [], 0
    for i in range(first, last):
        n = int(offsets[i + 1] - offsets[i]) - 1  # wejście bez ostatniego tokenu
        if batch and used + n > tokens_per_batch:
            yield batch
            batch, used = [], 0
        batch.append(i)
        used += n
    if batch:
        yield batch


def check_model_vocab(model, cache):
    """Nauczyciel musi mieć słownik APT4 identyczny z cache (oba backendy; ID tokenów są nasze)."""
    from transformers import AutoConfig, PreTrainedTokenizerFast
    config = AutoConfig.from_pretrained(model)
    if config.vocab_size != 32000:
        raise SystemExit(f"vocab_size {config.vocab_size} != 32000: to nie jest model z APT4")
    # Ten sam słownik co cache (raz już pobrał się Minitron z innym tokenizerem).
    if not (Path(model) / "tokenizer.json").exists():
        raise SystemExit(f"Brak tokenizer.json w {model} — nie da się sprawdzić słownika")
    a = PreTrainedTokenizerFast.from_pretrained(model).get_vocab()
    b = PreTrainedTokenizerFast.from_pretrained(str(cache / "tokenizer")).get_vocab()
    if a != b:
        raise SystemExit("Tokenizer modelu różni się od tokenizera cache")
    return config


class Teacher:
    def __init__(self, args, cache):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        config = check_model_vocab(args.model, cache)
        extra = {}
        quant = getattr(config, "quantization_config", None) or {}
        if (quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)) \
                == "compressed-tensors":
            # W8A8 (llm-compressor): skompresowane wagi int8 nie przechodzą przez zwykły forward
            # (F.linear bf16 × int8), więc rozpakowujemy je raz do bf16 przy ładowaniu. Wartości wag
            # zostają skwantyzowane; kwantyzacji aktywacji (jak w SGLang/vLLM) tu nie ma.
            from transformers import CompressedTensorsConfig
            extra["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn, **extra,
        ).to(args.device).eval()
        self.k, self.chunk, self.device = args.k, args.logit_chunk, args.device

    def __call__(self, inputs, targets, position_ids):
        """inputs/targets/position_ids: tensory [T] na GPU. Zwraca (ids [T,k], logp [T,k], target [T])."""
        torch = self.torch
        with torch.inference_mode():
            h = self.model.model(input_ids=inputs[None], position_ids=position_ids[None]).last_hidden_state[0]
            ids, logp, target = [], [], []
            for s in range(0, len(h), self.chunk):
                lp = torch.log_softmax(self.model.lm_head(h[s:s + self.chunk]).float(), dim=-1)
                v, i = lp.topk(self.k, dim=-1)
                ids.append(i.to(torch.int32))
                logp.append(v)
                target.append(lp.gather(1, targets[s:s + self.chunk, None].long())[:, 0])
            return torch.cat(ids), torch.cat(logp), torch.cat(target)


def gather_batch(tokens, offsets, batch):
    """Wejście, cele, position_ids i wiersze wyniku dla paczki rekordów."""
    inputs, targets, positions, rows = [], [], [], []
    for i in batch:
        s, e = int(offsets[i]), int(offsets[i + 1])
        inputs.append(tokens[s:e - 1])
        targets.append(tokens[s + 1:e])
        positions.append(np.arange(e - s - 1))
        rows.append(np.arange(s + 1, e))
    return (np.concatenate(inputs).astype(np.int64), np.concatenate(targets).astype(np.int64),
            np.concatenate(positions), np.concatenate(rows))


def check_packing(teacher, tokens, offsets, batch):
    """Paczka musi dać to samo co rekordy osobno; inaczej attention przecieka między rozmowami."""
    torch = teacher.torch
    batch = batch[:8]
    if len(batch) < 2:
        return False
    x, y, p, _ = gather_batch(tokens, offsets, batch)
    dev = lambda a: torch.from_numpy(a).to(teacher.device)
    packed = teacher(dev(x), dev(y), dev(p))
    # Wzorzec przecieku: ta sama paczka z ciągłymi position_ids, czyli zwykła causal attention
    # przez wszystkie rekordy.
    leaked = teacher(dev(x), dev(y), dev(np.arange(len(x))))
    single = [teacher(*(dev(a) for a in gather_batch(tokens, offsets, [i])[:3])) for i in batch]
    single = [torch.cat(parts) for parts in zip(*single)]
    # Pierwszy rekord paczki jest zawsze identyczny; przeciek widać dopiero w kolejnych.
    rest = slice(int(offsets[batch[0] + 1] - offsets[batch[0]]) - 1, None)
    top1 = (packed[0][rest, 0] == single[0][rest, 0]).float().mean().item()
    diff = (packed[2][rest] - single[2][rest]).abs().mean().item()
    leak = (leaked[2][rest] - single[2][rest]).abs().mean().item()
    print(f"check pakowania: {len(batch)} rekordów, {len(x)} pozycji, zgodność top-1 {top1:.4f}, "
          f"śr. |Δ target logp| {diff:.4f} (przy przecieku {leak:.4f})", flush=True)
    # Szum bf16 (inne kształty, inna kolejność redukcji) zależy od rekordów, więc nie stały próg,
    # tylko porównanie ze wzorcem przecieku.
    if diff > 0.25 * leak:
        raise SystemExit("Paczka różni się od osobnych forwardów — packing nie działa (attn_implementation?)")
    return True


class SGLangTeacher:
    """Ten sam wynik co Teacher, ale przez silnik SGLang offline (prawdziwe kernele W8A8 INT8).

    Każdy rekord idzie w całości jako input_ids; SGLang zwraca dla każdej pozycji promptu top-k
    i logprob tokenu, który tam stoi (pozycja 0 nie ma predykcji). Paczkowanie robi sam silnik.
    Radix cache wyłączony: rekordy się nie powtarzają, a dla pozycji wziętych z cache prefiksu
    SGLang nie liczyłby logprobów od nowa.
    """

    def __init__(self, args, cache):
        import sglang as sgl
        self.k = args.k
        self.vocab_size = check_model_vocab(args.model, cache).vocab_size
        self.engine = sgl.Engine(
            model_path=args.model, skip_tokenizer_init=True, disable_radix_cache=True,
            context_length=args.context_length, mem_fraction_static=args.mem_fraction,
            chunked_prefill_size=args.chunked_prefill,
            # Generujemy 1 token: grafy CUDA dla dekodowania nic nie dają, a zjadają pamięć,
            # której logity wszystkich pozycji promptu (chunk × słownik × fp32) potrzebują.
            disable_cuda_graph=True, log_level="error")

    def score(self, tokens, offsets, batch):
        """→ (rows, topk_ids <u2 [n,k], topk_logp f32 [n,k], target_logp f32 [n], targets int64 [n])."""
        prompts = [tokens[int(offsets[i]):int(offsets[i + 1])].astype(np.int64).tolist() for i in batch]
        outs = self.engine.generate(
            input_ids=prompts, return_logprob=True, logprob_start_len=0, top_logprobs_num=self.k,
            sampling_params=[{"max_new_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1}] * len(prompts))
        rows, ids, logp, target, labels = [], [], [], [], []
        for i, ids_in, o in zip(batch, prompts, outs):
            meta = o["meta_info"]
            tops, toks = meta["input_top_logprobs"], meta["input_token_logprobs"]
            if len(toks) != len(ids_in) or len(tops) != len(ids_in):
                raise SystemExit(f"SGLang zwrócił {len(toks)}/{len(tops)} pozycji dla rekordu o {len(ids_in)} "
                                 "tokenach (logprob_start_len=0 powinien dać wszystkie)")
            # Pozycja 0 (BOS) nie ma predykcji; pozycja j to rozkład dla tokenu j.
            # SGLang (logprob_result_processor.py) raportuje ID >= vocab_size - 1 jako 0 — błąd o jeden
            # w czyszczeniu hashy obrazów; dotyczy tylko raportu (wejście modelu i logprob liczone są
            # z prawdziwym ID), więc ostatni token słownika (APT4: 31999 = '°') porównujemy jako 0.
            last_id = self.vocab_size - 1
            got = [t[1] for t in toks[1:]]
            expected = [0 if t == last_id else t for t in ids_in[1:]]
            if got != expected:
                diff = next(j for j, (a, b) in enumerate(zip(got, expected)) if a != b) + 1
                raise SystemExit(
                    "Tokeny w logprobach SGLang nie zgadzają się z rekordem cache.\n"
                    f"  rekord {i}: {len(ids_in)} tokenów, pierwsza różnica na pozycji {diff}\n"
                    f"  ids rekordu[:8]:          {ids_in[:8]}\n"
                    f"  input_token_logprobs[:8]: {toks[:8]}\n"
                    f"  input_top_logprobs[:2]:   {tops[:2]}\n"
                    f"  wokół różnicy: rekord {ids_in[max(diff - 3, 0):diff + 3]}, "
                    f"SGLang {[t[1] for t in toks[max(diff - 3, 0):diff + 3]]}\n"
                    f"  inne klucze meta_info: {sorted(meta)}")
            s, e = int(offsets[i]), int(offsets[i + 1])
            rows.append(np.arange(s + 1, e))
            labels.append(np.asarray(ids_in[1:], dtype=np.int64))
            target.append(np.fromiter((t[0] for t in toks[1:]), np.float32, e - s - 1))
            flat = [x for pos in tops[1:] for x in pos[:self.k]]
            if len(flat) != (e - s - 1) * self.k:
                raise SystemExit(f"SGLang zwrócił mniej niż top-{self.k} na pozycję (--max-logprobs?)")
            logp.append(np.fromiter((x[0] for x in flat), np.float32, len(flat)).reshape(-1, self.k))
            ids.append(np.fromiter((x[1] for x in flat), np.uint16, len(flat)).reshape(-1, self.k))
        return (np.concatenate(rows), np.concatenate(ids), np.concatenate(logp),
                np.concatenate(target), np.concatenate(labels))


def hf_score(teacher, tokens, offsets, batch, device):
    import torch
    x, y, p, rows = gather_batch(tokens, offsets, batch)
    ids, logp, target = teacher(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                                torch.from_numpy(p).to(device))
    return (rows, ids.to(torch.int16).cpu().numpy().view("<u2"), logp.float().cpu().numpy(),
            target.float().cpu().numpy(), y)


def run(args):
    cache, out = Path(args.cache).resolve(), Path(args.out).resolve()
    manifest, shards = load_cache(cache)
    cache_sha = sha256(cache / "manifest.json")
    if args.tokens_per_batch is None:
        # HF: jeden forward na paczkę; SGLang: jedno generate() na paczkę, silnik dzieli ją sam.
        args.tokens_per_batch = 262144 if args.backend == "sglang" else 16384
    if not args.context_length:
        args.context_length = manifest["seq_length"] + 1  # rekord ma do seq_length + 1 tokenów
    if args.tokens_per_batch < manifest["seq_length"]:
        raise SystemExit(f"--tokens-per-batch musi być >= seq_length ({manifest['seq_length']})")
    shards = [s for s in shards if s["split"] in args.splits]
    # Postęp tego ranku dla launchera (run_teacher_topk.sh pokazuje sumę w jednej linii).
    total = finished = 0
    for shard in shards:
        offsets = np.memmap(cache / f"{shard['prefix']}.offsets.bin", dtype="<u8", mode="r").astype(np.int64)
        bounds = chunk_bounds(offsets, args.world)
        n = int(offsets[bounds[args.rank + 1]] - offsets[bounds[args.rank]]) \
            - (bounds[args.rank + 1] - bounds[args.rank])
        total += n
        if (out / f"{shard['prefix']}.rank{args.rank}of{args.world}.done").exists():
            finished += n
    progress = out / "logs" / f"rank{args.rank}.progress"
    started, new_positions = time.time(), 0

    def report():
        if args.benchmark:
            return
        progress.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(done=finished + new_positions, total=total, new=new_positions,
                                       elapsed=time.time() - started)))
        tmp.replace(progress)

    report()
    sglang = args.backend == "sglang"
    teacher = SGLangTeacher(args, cache) if sglang else Teacher(args, cache)
    started = time.time()
    checked = sglang  # test pakowania dotyczy tylko HF (SGLang paczkuje sam)
    for shard in shards:
        prefix = shard["prefix"]
        tokens = np.memmap(cache / f"{prefix}.tokens.bin", dtype="<u2", mode="r")
        offsets = np.memmap(cache / f"{prefix}.offsets.bin", dtype="<u8", mode="r").astype(np.int64)
        bounds = chunk_bounds(offsets, args.world)
        first, last = bounds[args.rank], bounds[args.rank + 1]
        if args.benchmark:
            last = min(last, first + args.benchmark)
        done = out / f"{prefix}.rank{args.rank}of{args.world}.done"
        if not args.benchmark and done.exists():
            print(f"{prefix}: kawałek {args.rank}/{args.world} już gotowy", flush=True)
            continue
        maps = None if args.benchmark else open_outputs(out, shard, args.k)
        stats = dict(records=last - first, positions=0, target_logp_sum=0.0, mass_sum=0.0,
                     top1_hits=0, mass_min=1.0)
        start = time.time()
        for b, batch in enumerate(batches(offsets, first, last, args.tokens_per_batch)):
            if not checked:
                checked = check_packing(teacher, tokens, offsets, batch)
            rows, ids, logp, target, y = (teacher.score(tokens, offsets, batch) if sglang
                                          else hf_score(teacher, tokens, offsets, batch, args.device))
            mass = np.exp(logp.astype(np.float64)).sum(-1)
            stats["positions"] += len(rows)
            new_positions += len(rows)
            stats["target_logp_sum"] += float(target.astype(np.float64).sum())
            stats["mass_sum"] += float(mass.sum())
            stats["mass_min"] = min(stats["mass_min"], float(mass.min()))
            stats["top1_hits"] += int((ids[:, 0].astype(np.int64) == y).sum())
            if maps is not None:
                maps["topk_ids"][rows] = ids
                maps["topk_logp"][rows] = logp.astype(np.float16)
                maps["target_logp"][rows] = target.astype(np.float16)
            report()
            if b % args.log_every == 0:
                rate = stats["positions"] / (time.time() - start)
                print(f"{prefix} r{args.rank}: {batch[-1] + 1 - first}/{last - first} rekordów, "
                      f"{rate:,.0f} tok/s", flush=True)
        seconds = time.time() - start
        n = max(stats["positions"], 1)
        summary = (f"{prefix} r{args.rank}: {stats['positions']:,} pozycji w {seconds:.0f} s "
                   f"({stats['positions'] / seconds:,.0f} tok/s); NLL nauczyciela "
                   f"{-stats['target_logp_sum'] / n:.4f}, top-1 {stats['top1_hits'] / n:.4f}, "
                   f"masa top-{args.k} śr. {stats['mass_sum'] / n:.4f} min {stats['mass_min']:.4f}")
        print(summary, flush=True)
        if args.benchmark:
            return
        for m in maps.values():
            m.flush()
        stats.update(seconds=seconds, k=args.k, model=str(Path(args.model).resolve()), backend=args.backend,
                     cache_manifest_sha256=cache_sha, records_range=[first, last])
        done.write_text(json.dumps(stats, indent=1))
        report()


def finalize(args):
    cache, out = Path(args.cache).resolve(), Path(args.out).resolve()
    manifest, shards = load_cache(cache)
    cache_sha = sha256(cache / "manifest.json")
    shards = [s for s in shards if s["split"] in args.splits]
    result, totals = [], {}
    for shard in shards:
        parts = []
        for r in range(args.world):
            done = out / f"{shard['prefix']}.rank{r}of{args.world}.done"
            if not done.exists():
                raise SystemExit(f"Brak {done}")
            parts.append(json.loads(done.read_text()))
        if {p["cache_manifest_sha256"] for p in parts} != {cache_sha}:
            raise SystemExit(f"{shard['prefix']}: kawałki liczone dla innego cache")
        if {p["k"] for p in parts} != {args.k} or len({p["model"] for p in parts}) != 1:
            raise SystemExit(f"{shard['prefix']}: kawałki z różnym k albo modelem")
        if sum(p["records"] for p in parts) != shard["records"] or \
                sum(p["positions"] for p in parts) != shard["tokens"] - shard["records"]:
            raise SystemExit(f"{shard['prefix']}: liczba rekordów/pozycji nie zgadza się z cache")
        files = {name: path.relative_to(out).as_posix() for name, path in out_paths(out, shard["prefix"]).items()}
        result.append(dict(prefix=shard["prefix"], split=shard["split"], records=shard["records"],
                           tokens=shard["tokens"], files=files))
        t = totals.setdefault(shard["split"], dict(positions=0, target_logp_sum=0.0, mass_sum=0.0, top1_hits=0))
        for key in t:
            t[key] += sum(p[key] for p in parts)
    for split, t in totals.items():
        n = t["positions"]
        t.update(teacher_nll=-t["target_logp_sum"] / n, top1=t["top1_hits"] / n, mass_mean=t["mass_sum"] / n)
        print(f"{split}: {n:,} pozycji, NLL {t['teacher_nll']:.4f}, top-1 {t['top1']:.4f}, "
              f"masa top-{args.k} {t['mass_mean']:.4f}")
    (out / "manifest.json").write_text(json.dumps(dict(
        format=FORMAT, k=args.k, model=parts[0]["model"], temperature=1.0,
        cache=str(cache), cache_manifest_sha256=cache_sha, seq_length=manifest["seq_length"],
        dtypes=dict(topk_ids="<u2", topk_logp="<f2", target_logp="<f2"),
        totals=totals, shards=result), indent=1))
    print(f"Gotowe: {out / 'manifest.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="katalog cache z manifest.json (poziomka-sft-v2)")
    p.add_argument("--model", help="katalog HF nauczyciela (bf16)")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--tokens-per-batch", type=int, default=None,
                   help="pozycji wejścia na paczkę (domyślnie hf 16384, sglang 262144)")
    p.add_argument("--logit-chunk", type=int, default=4096, help="pozycji na raz przez lm_head + log_softmax")
    p.add_argument("--attn", default="flash_attention_2", help="musi obsługiwać packing przez position_ids")
    p.add_argument("--device", default="cuda", help="cpu tylko do testów")
    p.add_argument("--backend", choices=["hf", "sglang"], default="hf",
                   help="sglang: silnik SGLang offline (np. W8A8 INT8 z kernelami int8); hf: transformers")
    p.add_argument("--context-length", type=int, default=0, help="sglang: kontekst silnika (0 = seq_length+1 cache)")
    p.add_argument("--mem-fraction", type=float, default=0.5,
                   help="sglang: mem_fraction_static (reszta karty na logity promptu)")
    p.add_argument("--chunked-prefill", type=int, default=4096,
                   help="sglang: chunked_prefill_size; logity jednego kawałka = chunk × 32000 × 4 B")
    p.add_argument("--benchmark", type=int, default=0, help="N rekordów z pierwszego shardu, bez zapisu")
    p.add_argument("--log-every", type=int, default=50, help="co ile paczek wypisać postęp")
    p.add_argument("--finalize", action="store_true", help="sprawdź komplet .done i zapisz manifest.json")
    args = p.parse_args()
    if not 0 <= args.rank < args.world:
        raise SystemExit("--rank musi być w [0, world)")
    if args.finalize:
        finalize(args)
    else:
        if not args.model:
            raise SystemExit("--model jest wymagany")
        run(args)


if __name__ == "__main__":
    main()
