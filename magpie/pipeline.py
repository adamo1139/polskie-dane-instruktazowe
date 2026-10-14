"""Pytanie + odpowiedź jednym modelem w jednym przebiegu (Bielik po SFT na nauka-lotnictwa, szablon v13).

Każde zadanie to dwa zapytania /v1/completions wysłane jedno po drugim:
  1. prompt glimmer8 + otwarta tura user  → pytanie (stop <|im_end|>), filtry jak w magpie.generate;
  2. TEN SAM tekst + pytanie + <|im_end|>\\n<|im_start|>assistant\\n<think>\\n (albo <think></think>\\n)
     → odpowiedź. Prefiks drugiego zapytania to dokładnie pierwsze zapytanie z wygenerowanym pytaniem,
     więc SGLang (RadixAttention) / vLLM (prefix caching) nie liczy go od nowa.
Rozumowanie wydzielamy sami po </think> (completions nie przechodzi przez parser reasoningu).

Odrzucane odpowiedzi: ucięte/puste, przy rozumowaniu brak </think> albo puste rozumowanie, tagi think
w odpowiedzi, wywołanie narzędzia (<function=, <tool_call>…). Pytania odrzucone przez filtry nie dostają
odpowiedzi.

Tryb myślenia: --thinking off|on|mix (mix: --think-share zadań z rozumowaniem, losowane z seeda zadania).
Zadania, seed i wznawianie jak w magpie.generate (master-seed w <out>/meta.json, gotowe id pomijane).

Wyjście (w --out): raw.jsonl (wszystko) i po zakończeniu sft.jsonl — rozmowy user, assistant
(z --system-in-sft: system, user, assistant), id = SHA256 pytania, deduplikacja po pytaniu.

  python -m magpie.pipeline --model <ścieżka modelu na serwerze> --total 100000 --thinking off \\
      --out data/pipeline_bielik7_nothink
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
from collections import Counter

import aiohttp
from aiolimiter import AsyncLimiter

from .filters import classify, clean
from .generate import StreamTimeout, generate, resolve_master_seed, run
from .prompt import build_jobs, build_prompt

ASSISTANT_OPEN = {"on": "<|im_end|>\n<|im_start|>assistant\n<think>\n",
                  "off": "<|im_end|>\n<|im_start|>assistant\n<think></think>\n"}
THINK_END = re.compile(r"\s*</think>\s*")
LEFTOVER_THINK = re.compile(r"</?think>")
TOOL_CALL = re.compile(r"<function=|<parameter=|</?tool_call>")


def answer_args(args: argparse.Namespace) -> argparse.Namespace:
    """Kopia args z samplowaniem i limitami odpowiedzi (generate() czyta je z args)."""
    a = copy.copy(args)
    a.temperature, a.top_p, a.max_tokens = args.answer_temperature, args.answer_top_p, args.answer_max_tokens
    a.top_k, a.min_p = args.answer_top_k, args.answer_min_p
    a.presence_penalty, a.frequency_penalty = args.answer_presence_penalty, args.answer_frequency_penalty
    a.repetition_penalty = args.answer_repetition_penalty
    a.ttft_timeout, a.chunk_timeout, a.total_timeout = (args.answer_ttft_timeout, args.answer_chunk_timeout,
                                                        args.answer_total_timeout)
    return a


def mode_for(job: dict, args: argparse.Namespace) -> str:
    if args.thinking != "mix":
        return args.thinking
    return "on" if random.Random(job["seed"] ^ 0x5EED).random() < args.think_share else "off"


async def call(session, limiter, args, job, stats) -> dict:
    """generate() z ponowieniami; zwraca wynik albo {'finish': 'error', 'error': …}."""
    last = None
    for attempt in range(args.retries + 1):
        async with limiter:
            pass
        try:
            return await generate(session, args, job)
        except (StreamTimeout, TimeoutError) as e:
            stats["timeout"] += 1
            last = f"timeout: {e}" if str(e) else "timeout: total"
        except (aiohttp.ClientError, RuntimeError, json.JSONDecodeError) as e:
            stats["http_err"] += 1
            last = repr(e)[:200]
        if attempt < args.retries:
            await asyncio.sleep(5 * (attempt + 1))
    return {"finish": "error", "raw": "", "error": last}


def split_answer(text: str, mode: str) -> tuple[str, str, str | None]:
    """→ (rozumowanie, odpowiedź, powód odrzucenia albo None)."""
    if mode == "on":
        parts = THINK_END.split(text, maxsplit=1)
        if len(parts) < 2:
            return text.strip(), "", "niezamknięte rozumowanie"
        reasoning, answer = parts[0].strip(), parts[1].strip()
        if not reasoning:
            return "", answer, "brak rozumowania"
    else:
        reasoning, answer = "", text.strip()
    if not answer:
        return reasoning, answer, "pusta odpowiedź"
    if LEFTOVER_THINK.search(answer):
        return reasoning, answer, "tag think w odpowiedzi"
    if TOOL_CALL.search(answer):
        return reasoning, answer, "wywołanie narzędzia"
    return reasoning, answer, None


async def work(session, limiter, args, job, stats) -> dict:
    meta = {k: v for k, v in job.items() if k != "prompt"}
    prompt = build_prompt(job["system"], "v13")
    q = await call(session, limiter, args, {**job, "prompt": prompt}, stats)
    if q["finish"] == "error":
        return {**meta, "stage": "pytanie", "finish": "error", "ok": False, "reason": f"błąd: {q.get('error')}"}
    q = clean(q)
    reason = classify(q, job["system"])
    row = {**meta, "question": q["raw"], "question_finish": q["finish"]}
    if reason:
        stats["odrzucone pytanie"] += 1
        return {**row, "stage": "pytanie", "finish": q["finish"], "ok": False, "reason": reason}

    mode = mode_for(job, args)
    a_job = {**job, "prompt": prompt + q["raw"] + ASSISTANT_OPEN[mode]}
    # Limit odpowiedzi: kontekst serwera minus ostrożny szacunek promptu (≥ 1 token na 2 znaki).
    a_args = copy.copy(args.answer)
    budget = args.context - len(a_job["prompt"]) // 2 - 32
    a_args.max_tokens = min(args.answer_max_tokens, budget) if args.answer_max_tokens else budget
    a = await call(session, limiter, a_args, a_job, stats)
    if a["finish"] == "error":
        return {**row, "thinking": mode, "stage": "odpowiedź", "finish": "error", "ok": False,
                "reason": f"błąd: {a.get('error')}"}
    reasoning, answer, why = (split_answer(a["raw"], mode) if a["finish"] == "stop"
                              else ("", "", "ucięte"))
    if why:
        stats[why] += 1
    return {**row, "thinking": mode, "stage": "odpowiedź", "finish": a["finish"], "reasoning": reasoning,
            "answer": answer, "ok": why is None, "reason": why,
            "ttft_s": q.get("ttft_s"), "latency_s": round((q.get("latency_s") or 0) + (a.get("latency_s") or 0), 2),
            "answer_ttft_s": a.get("ttft_s")}


def finalize(out_dir: str, model: str, system_in_sft: bool) -> None:
    rows = {}
    with open(os.path.join(out_dir, "raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and r.get("finish") == "error":
                continue
            rows[r["id"]] = r
    seen, good = set(), []
    for r in sorted(rows.values(), key=lambda r: r["id"]):
        if not r["ok"]:
            continue
        h = hashlib.sha256(r["question"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        assistant = {"role": "assistant", "content": r["answer"]}
        if r["reasoning"]:
            assistant["reasoning_content"] = r["reasoning"]
        messages = ([{"role": "system", "content": r["system"]}] if system_in_sft else []) + \
                   [{"role": "user", "content": r["question"]}, assistant]
        good.append({"id": h, "messages": messages,
                     "meta": {"topic": r["topic"], "persona": r["persona"], "model": os.path.basename(model),
                              "thinking": r["thinking"], "pipeline": "magpie.pipeline"}})
    with open(os.path.join(out_dir, "sft.jsonl"), "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    reasons = Counter(f"{r.get('stage')}: {r.get('reason')}" for r in rows.values() if not r["ok"])
    print(f"\nzadań: {len(rows)}  do sft.jsonl: {len(good)}  "
          f"z rozumowaniem: {sum('reasoning_content' in r['messages'][-1] for r in good)}  "
          f"duplikaty pytań: {sum(r['ok'] for r in rows.values()) - len(good)}")
    print("odrzuty:", dict(reasons.most_common(12)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://192.168.1.26:2300/v1")
    p.add_argument("--model", required=True, help="nazwa modelu na serwerze (Bielik po SFT, szablon v13)")
    p.add_argument("--total", type=int, default=100000)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--subs-min", type=int, default=15)
    p.add_argument("--subs-max", type=int, default=15)
    p.add_argument("--thinking", choices=["off", "on", "mix"], default="off")
    p.add_argument("--think-share", type=float, default=0.2, help="przy --thinking mix: udział z rozumowaniem")
    p.add_argument("--system-in-sft", action="store_true", help="zapisz prompt glimmer8 w sft.jsonl")
    # pytanie
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--ttft-timeout", type=float, default=600)
    p.add_argument("--chunk-timeout", type=float, default=300)
    p.add_argument("--total-timeout", type=float, default=900)
    # odpowiedź
    p.add_argument("--answer-temperature", type=float, default=0.7)
    p.add_argument("--answer-top-p", type=float, default=0.95)
    p.add_argument("--answer-top-k", type=int, default=None)
    p.add_argument("--answer-min-p", type=float, default=None)
    p.add_argument("--answer-presence-penalty", type=float, default=None)
    p.add_argument("--answer-frequency-penalty", type=float, default=None)
    p.add_argument("--answer-repetition-penalty", type=float, default=None)
    p.add_argument("--answer-max-tokens", type=int, default=0,
                   help="limit odpowiedzi; 0 = reszta kontekstu serwera (max_model_len z /v1/models) po prompcie")
    p.add_argument("--context", type=int, default=0,
                   help="kontekst serwera w tokenach (0 = max_model_len z /v1/models; router SGLang go nie podaje)")
    p.add_argument("--answer-ttft-timeout", type=float, default=3600)
    p.add_argument("--answer-chunk-timeout", type=float, default=600)
    p.add_argument("--answer-total-timeout", type=float, default=7200)
    # wspólne
    p.add_argument("--no-stream", action="store_true",
                   help="zapytania bez streamingu (mniej pracy dla klienta i routera; brak pomiaru ttft)")
    p.add_argument("--concurrency", type=int, default=1500)
    p.add_argument("--rate", type=float, default=200)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    args.url = args.url.rstrip("/")
    args.template = "v13"
    args.answer = answer_args(args)
    if not args.context:
        # Pojedynczy SGLang/vLLM podaje max_model_len w /v1/models; router SGLang (DP) — nie.
        import urllib.request
        with urllib.request.urlopen(f"{args.url}/models", timeout=30) as resp:
            info = next((m for m in json.load(resp)["data"] if m["id"] == args.model), {})
        args.context = info.get("max_model_len")
        if not args.context:
            raise SystemExit("Serwer nie podaje max_model_len (np. router SGLang) — podaj --context, "
                             "np. --context 8192 (= --context-length serwera)")

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.jsonl")
    if args.finalize_only:
        finalize(args.out, args.model, args.system_in_sft)
        return
    args.master_seed = resolve_master_seed(args, raw_path)
    done: set[int] = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, f) if r.get("finish") != "error"}
    jobs = [j for j in build_jobs(args.total, args.master_seed, args.subs_min, args.subs_max) if j["id"] not in done]
    print(f"zadań: {args.total}, gotowych: {len(done)}, do zrobienia: {len(jobs)}  master-seed {args.master_seed}  "
          f"(model {args.model}, thinking {args.thinking}"
          f"{f' {args.think_share:.0%}' if args.thinking == 'mix' else ''}, concurrency {args.concurrency}, "
          f"rate {args.rate}/s; odpowiedź: temperature {args.answer_temperature}, top_p {args.answer_top_p}, "
          f"frequency_penalty {args.answer_frequency_penalty}, repetition_penalty {args.answer_repetition_penalty}, "
          f"max_tokens {args.answer_max_tokens or 'reszta kontekstu'}, kontekst serwera {args.context})", flush=True)
    if jobs:
        asyncio.run(run(args, jobs, raw_path, work_fn=work))
    finalize(args.out, args.model, args.system_in_sft)


if __name__ == "__main__":
    main()
