"""Generacja pytań użytkownika metodą Magpie (Muse Glimmer, wariant glimmer8) — asyncio, wznawialna.

Concurrency: stała pula --concurrency workerów ciągnących zadania z kolejki — zakończone zapytanie
natychmiast zwalnia slot i worker bierze następne. AsyncLimiter ogranicza tempo startów
(--rate zapytań/s), żeby nie zalać serwera naraz na starcie.

Timeouty (streaming, więc da się odróżnić „wolno generuje” od „zawisło”):
  --ttft-timeout   czekanie na pierwszy fragment (kolejka + prefill)
  --chunk-timeout  maks. przerwa między kolejnymi fragmentami strumienia
  --total-timeout  twardy limit całego zapytania
Zerwanie połączenia przerywa też zapytanie po stronie vLLM. Każdy wiersz ma ttft_s/latency_s,
postęp pokazuje ich p50/p95 i liczbę timeoutów.

Zadania (id, system, seed, persona, podtematy) wynikają deterministycznie z --master-seed i
topics.py. Nowy run (nowy --out) losuje seed i zapisuje go w <out>/meta.json — każdy run ma inne
zadania. Wznowienie = ten sam wiersz poleceń: seed jest czytany z meta.json, gotowe id są pomijane.

Wyjście (w --out):
  raw.jsonl       — wszystkie odpowiedzi (także odrzucone), dopisywane co --flush-every
  accepted.jsonl  — po zakończeniu: OK + deduplikacja po znormalizowanej treści
"""
import argparse
import asyncio
import json
import os
import re
import secrets
import statistics
import time
from collections import Counter

import aiohttp
from aiolimiter import AsyncLimiter

from .filters import SELF_ANSWER, classify, clean
from .prompt import STOP_STRINGS, STOP_TOKEN_IDS, build_jobs, build_prompt


class StreamTimeout(Exception):
    pass


async def generate(session: aiohttp.ClientSession, args: argparse.Namespace, job: dict) -> dict:
    payload = {
        "model": args.model,
        "prompt": build_prompt(job["system"]),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": job["seed"],
        "stop": STOP_STRINGS,
        "stop_token_ids": STOP_TOKEN_IDS,
        "stream": True,
    }
    t0 = time.monotonic()
    ttft = None
    parts: list[str] = []
    finish = stop_reason = None
    async with asyncio.timeout(args.total_timeout):
        async with session.post(f"{args.url}/completions", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            buf = b""
            while True:
                limit = args.ttft_timeout if ttft is None else args.chunk_timeout
                try:
                    chunk = await asyncio.wait_for(resp.content.readany(), timeout=limit)
                except TimeoutError:
                    phase = "ttft" if ttft is None else "chunk"
                    raise StreamTimeout(f"{phase} > {limit}s po {time.monotonic() - t0:.0f}s")
                if not chunk:
                    break
                if ttft is None:
                    ttft = time.monotonic() - t0
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    ch = json.loads(data)["choices"][0]
                    parts.append(ch.get("text") or "")
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                        stop_reason = ch.get("stop_reason")
    raw = "".join(parts).strip()
    return {"raw": raw, "first": raw.split("\n")[0].strip() if raw else "",
            "finish": finish, "matched_stop": stop_reason,
            "ttft_s": round(ttft or 0.0, 2), "latency_s": round(time.monotonic() - t0, 2)}


async def work(session: aiohttp.ClientSession, limiter: AsyncLimiter, args: argparse.Namespace,
               job: dict, stats: Counter) -> dict:
    last_err = None
    for attempt in range(args.retries + 1):
        async with limiter:
            pass  # limiter ogranicza tylko tempo startów; slot trzyma worker, nie limiter
        try:
            out = await generate(session, args, job)
            break
        except (StreamTimeout, TimeoutError) as e:
            stats["timeout"] += 1
            last_err = f"timeout: {e}" if str(e) else "timeout: total"
        except (aiohttp.ClientError, RuntimeError, json.JSONDecodeError) as e:
            stats["http_err"] += 1
            last_err = repr(e)[:200]
        if attempt < args.retries:
            await asyncio.sleep(5 * (attempt + 1))
    else:
        return {**job, "finish": "error", "raw": "", "first": "", "ok": False,
                "reason": f"błąd: {last_err}", "attempts": args.retries + 1}
    out = clean(out)
    reason = classify(out, job["system"])
    return {**job, **out, "reason": reason, "ok": reason is None, "attempts": attempt + 1,
            "self_answer_suspect": bool(SELF_ANSWER.search(out["raw"]))}


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100)[int(q) - 1] if len(values) > 1 else values[0]


async def run(args: argparse.Namespace, jobs: list[dict], raw_path: str) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    limiter = AsyncLimiter(args.rate, 1)
    stats: Counter = Counter()
    buffer: list[dict] = []
    ttfts: list[float] = []
    lats: list[float] = []
    t0 = time.monotonic()
    total = len(jobs)
    n_workers = min(args.concurrency, total)
    idle = [0]

    fout = open(raw_path, "a", encoding="utf-8")

    def flush() -> None:
        if buffer:
            fout.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buffer))
            fout.flush()
            os.fsync(fout.fileno())
            buffer.clear()

    def progress() -> None:
        n = stats["done"]
        el = time.monotonic() - t0
        rate = n / el if el else 0
        print(f"{n}/{total}  OK {stats['ok']} ({100 * stats['ok'] / max(n, 1):.0f}%)  "
              f"błędy {stats['error']}  timeouty {stats['timeout']}  http {stats['http_err']}  "
              f"w locie {n_workers - idle[0]}  {rate:.1f} zap/s  "
              f"ttft p50/p95 {pct(ttfts, 50):.0f}/{pct(ttfts, 95):.0f}s  "
              f"lat p50/p95 {pct(lats, 50):.0f}/{pct(lats, 95):.0f}s  "
              f"ETA {(total - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    async def worker(session: aiohttp.ClientSession) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                idle[0] += 1
                return
            row = await work(session, limiter, args, job, stats)
            stats["done"] += 1
            stats["ok" if row["ok"] else ("error" if row["finish"] == "error" else "rej")] += 1
            if row.get("ttft_s"):
                ttfts.append(row["ttft_s"])
                lats.append(row["latency_s"])
            buffer.append(row)
            if len(buffer) >= args.flush_every:
                flush()
            if stats["done"] % args.progress_every == 0:
                progress()

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    # Brak globalnego timeoutu aiohttp — pilnują go własne ttft/chunk/total powyżej.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            await asyncio.gather(*(worker(session) for _ in range(n_workers)))
    finally:
        flush()
        fout.close()
        progress()


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def finalize(out_dir: str) -> None:
    rows = {}
    with open(os.path.join(out_dir, "raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and r.get("finish") == "error":
                continue  # nie nadpisuj udanej odpowiedzi późniejszym błędem
            rows[r["id"]] = r
    seen: set[str] = set()
    accepted = []
    dups = 0
    for r in sorted(rows.values(), key=lambda r: r["id"]):
        if not r["ok"]:
            continue
        key = normalize(r["raw"])
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        accepted.append(r)
    with open(os.path.join(out_dir, "accepted.jsonl"), "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    reasons = Counter(r.get("reason") for r in rows.values() if not r["ok"])
    print(f"\nwierszy: {len(rows)}  OK: {sum(r['ok'] for r in rows.values())}  duplikaty: {dups}  "
          f"zaakceptowane (accepted.jsonl): {len(accepted)}  "
          f"w tym podejrzane samoodpowiedzi: {sum(r.get('self_answer_suspect', False) for r in accepted)}")
    print("odrzuty:", dict(reasons.most_common(12)))
    print("per temat:", dict(Counter(r["topic"] for r in accepted)))
    print("persona:", dict(Counter(r["persona"] for r in accepted)))


def resolve_master_seed(args: argparse.Namespace, raw_path: str) -> int:
    """Seed runu: z <out>/meta.json przy wznowieniu, losowy (albo podany) dla nowego runu.

    Stały domyślny seed sprawiał, że każdy nowy run (nowy --out) dostawał te same zadania.
    """
    meta_path = os.path.join(args.out, "meta.json")
    params = {"subs_min": args.subs_min, "subs_max": args.subs_max}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if args.master_seed is not None and args.master_seed != meta["master_seed"]:
            raise SystemExit(f"--master-seed {args.master_seed} ≠ {meta['master_seed']} z {meta_path} — "
                             "wznowienie z innym seedem dałoby inne zadania pod tymi samymi id.")
        if {k: meta.get(k) for k in params} != params:
            raise SystemExit(f"parametry {params} ≠ zapisane w {meta_path}: "
                             f"{ {k: meta.get(k) for k in params} } — zadania byłyby inne.")
        return meta["master_seed"]
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0 and args.master_seed is None:
        raise SystemExit(f"{raw_path} istnieje, ale brak {meta_path} (run sprzed zapisu seeda). Podaj "
                         "--master-seed jawnie — stare runy używały 20261004.")
    seed = args.master_seed if args.master_seed is not None else secrets.randbits(31)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"master_seed": seed, **params}, f, indent=2)
    return seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://192.168.1.26:2300/v1")
    p.add_argument("--model", default="/media/nvme_2tb/maked/models/glimmer")
    p.add_argument("--total", type=int, default=50000)
    p.add_argument("--master-seed", type=int, default=None,
                   help="domyślnie losowany dla nowego runu i zapisywany w <out>/meta.json")
    p.add_argument("--concurrency", type=int, default=1500)
    p.add_argument("--rate", type=float, default=200, help="maks. startów zapytań na sekundę")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--subs-min", type=int, default=15)
    p.add_argument("--subs-max", type=int, default=15)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--ttft-timeout", type=float, default=300)
    p.add_argument("--chunk-timeout", type=float, default=120)
    p.add_argument("--total-timeout", type=float, default=600)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--out", default="data/magpie_glimmer8")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.jsonl")
    if args.finalize_only:
        finalize(args.out)
        return
    args.master_seed = resolve_master_seed(args, raw_path)

    done: set[int] = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("finish") != "error":
                    done.add(r["id"])
    jobs = [j for j in build_jobs(args.total, args.master_seed, args.subs_min, args.subs_max)
            if j["id"] not in done]
    print(f"zadań: {args.total}, gotowych: {len(done)}, do zrobienia: {len(jobs)}  master-seed {args.master_seed}  "
          f"(concurrency {args.concurrency}, rate {args.rate}/s, timeouty ttft/chunk/total "
          f"{args.ttft_timeout}/{args.chunk_timeout}/{args.total_timeout}s)", flush=True)
    if jobs:
        asyncio.run(run(args, jobs, raw_path))
    finalize(args.out)


if __name__ == "__main__":
    main()
