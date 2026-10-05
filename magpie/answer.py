"""Generacja odpowiedzi asystenta (np. Bielik 11B v3.0) na pytania z Magpie — asyncio, wznawialna.

Wejście: verified.jsonl z magpie.verify (bierzemy verdict == "ok") albo accepted.jsonl z
magpie.generate (ok == True). Zapytania idą do /v1/chat/completions (vLLM i SGLang) ze streamingiem.

Rozumowanie: --thinking off|on → chat_template_kwargs.enable_thinking. Rozumowanie zapisujemy
osobno: z pola reasoning_content (gdy serwer ma parser reasoningu) albo z <think>…</think> w treści.

Timeouty: --ttft-timeout (pierwszy fragment), --chunk-timeout (przerwa w strumieniu),
--total-timeout (całe zapytanie). Wznowienie = ten sam wiersz poleceń; gotowe id są pomijane.

Wyjście (w --out):
  answers.jsonl  — wszystkie odpowiedzi (także ucięte i błędy), dopisywane co --flush-every
  sft.jsonl      — po zakończeniu: rozmowy {messages: [user, assistant]} z finish_reason == stop
"""
import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter

import aiohttp
from aiolimiter import AsyncLimiter

THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


class StreamTimeout(Exception):
    pass


def split_reasoning(content: str, reasoning: str) -> tuple[str, str]:
    # Bez parsera reasoningu po stronie serwera rozumowanie przychodzi w treści jako <think>…</think>.
    m = THINK.match(content)
    if m:
        reasoning = reasoning or m[1].strip()
        content = content[m.end():]
    return content.strip(), reasoning.strip()


async def chat(session: aiohttp.ClientSession, args: argparse.Namespace, row: dict) -> dict:
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": row["raw"]}]
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    t0 = time.monotonic()
    ttft = None
    content, reasoning = [], []
    finish = None
    async with asyncio.timeout(args.total_timeout):
        async with session.post(f"{args.url}/chat/completions", json=payload) as resp:
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
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    choices = json.loads(line[5:])["choices"]
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    finish = choices[0].get("finish_reason") or finish
    answer, thought = split_reasoning("".join(content), "".join(reasoning))
    return {"answer": answer, "reasoning": thought, "finish": finish,
            "ttft_s": round(ttft or 0.0, 2), "latency_s": round(time.monotonic() - t0, 2)}


async def work(session: aiohttp.ClientSession, args: argparse.Namespace, row: dict, stats: Counter) -> dict:
    base = {k: row[k] for k in ("id", "raw", "topic", "persona", "subtopics") if k in row}
    base.update(model=args.model, thinking=args.thinking)
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            out = await chat(session, args, row)
            ok = out["finish"] == "stop" and bool(out["answer"])
            stats["ok" if ok else "ucięte/puste"] += 1
            return {**base, **out, "ok": ok, "attempts": attempt + 1}
        except (StreamTimeout, TimeoutError) as e:
            stats["timeout"] += 1
            last_err = f"timeout: {e}" if str(e) else "timeout: total"
        except (aiohttp.ClientError, RuntimeError, json.JSONDecodeError, KeyError) as e:
            stats["http_err"] += 1
            last_err = repr(e)[:200]
        if attempt < args.retries:
            await asyncio.sleep(5 * (attempt + 1))
    stats["error"] += 1
    return {**base, "finish": "error", "answer": "", "reasoning": "", "ok": False, "error": last_err}


async def run(args: argparse.Namespace, rows: list[dict], out_path: str) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)
    limiter = AsyncLimiter(args.rate, 1)
    stats: Counter = Counter()
    buffer: list[dict] = []
    t0 = time.monotonic()
    fout = open(out_path, "a", encoding="utf-8")

    def flush() -> None:
        if buffer:
            fout.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buffer))
            fout.flush()
            os.fsync(fout.fileno())
            buffer.clear()

    def progress() -> None:
        n = stats["done"]
        speed = n / max(time.monotonic() - t0, 1e-9)
        print(f"{n}/{len(rows)}  ok {stats['ok']}  ucięte/puste {stats['ucięte/puste']}  "
              f"błędy {stats['error']}  timeouty {stats['timeout']}  http {stats['http_err']}  "
              f"{speed:.2f} odp/s  ETA {(len(rows) - n) / max(speed, 1e-9) / 60:.0f} min", flush=True)

    async def worker(session: aiohttp.ClientSession) -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            async with limiter:
                pass  # limiter ogranicza tylko tempo startów; slot trzyma worker
            buffer.append(await work(session, args, row, stats))
            stats["done"] += 1
            if len(buffer) >= args.flush_every:
                flush()
            if stats["done"] % args.progress_every == 0:
                progress()

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    try:
        async with aiohttp.ClientSession(connector=connector,
                                         timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as session:
            await asyncio.gather(*(worker(session) for _ in range(min(args.concurrency, len(rows)))))
    finally:
        flush()
        fout.close()
        progress()


def finalize(out_dir: str, system: str | None) -> None:
    rows = {}
    with open(os.path.join(out_dir, "answers.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and not r["ok"]:
                continue  # nie nadpisuj udanej odpowiedzi późniejszym błędem
            rows[r["id"]] = r
    good = [r for r in sorted(rows.values(), key=lambda r: r["id"]) if r["ok"]]
    with open(os.path.join(out_dir, "sft.jsonl"), "w", encoding="utf-8") as f:
        for r in good:
            assistant = {"role": "assistant", "content": r["answer"]}
            if r["reasoning"]:
                assistant["reasoning_content"] = r["reasoning"]
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": r["raw"]}, assistant]
            f.write(json.dumps({"id": r["id"], "messages": messages,
                                "meta": {k: r.get(k) for k in ("topic", "persona", "model", "thinking")}},
                               ensure_ascii=False) + "\n")
    finish = Counter(r["finish"] for r in rows.values())
    print(f"\nodpowiedzi: {len(rows)}  do sft.jsonl: {len(good)}  finish_reason: {dict(finish)}  "
          f"z rozumowaniem: {sum(bool(r['reasoning']) for r in good)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://192.168.1.26:2300/v1", help="adres API z /v1")
    p.add_argument("--model", default=None, help="nazwa modelu (domyślnie pierwszy z /v1/models)")
    p.add_argument("--input", required=True, help="verified.jsonl z magpie.verify albo accepted.jsonl")
    p.add_argument("--out", required=True, help="katalog wynikowy")
    p.add_argument("--thinking", choices=["off", "on"], default="off")
    p.add_argument("--system", default=None, help="opcjonalny prompt systemowy")
    p.add_argument("--limit", type=int, default=0, help="tylko pierwsze N pytań (0 = wszystkie)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=None, help="domyślnie 5000 (off) / 16384 (on)")
    p.add_argument("--concurrency", type=int, default=500)
    p.add_argument("--rate", type=float, default=100, help="maks. startów zapytań na sekundę")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--ttft-timeout", type=float, default=300)
    p.add_argument("--chunk-timeout", type=float, default=120)
    p.add_argument("--total-timeout", type=float, default=1800)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    args.url = args.url.rstrip("/")
    if args.max_tokens is None:
        args.max_tokens = 16384 if args.thinking == "on" else 5000

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "answers.jsonl")
    if args.finalize_only:
        finalize(args.out, args.system)
        return

    with open(args.input, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r.get("verdict", "ok" if r.get("ok") else "") == "ok"]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, f) if r["finish"] != "error"}
    todo = [r for r in rows if r["id"] not in done]

    if args.model is None and todo:
        async def first_model() -> str:
            async with aiohttp.ClientSession() as s, s.get(f"{args.url}/models") as resp:
                return (await resp.json())["data"][0]["id"]
        args.model = asyncio.run(first_model())
    print(f"pytań: {len(rows)}, gotowych: {len(done)}, do zrobienia: {len(todo)}  (model {args.model}, "
          f"thinking {args.thinking}, max_tokens {args.max_tokens}, concurrency {args.concurrency})", flush=True)
    if todo:
        asyncio.run(run(args, todo, out_path))
    finalize(args.out, args.system)


if __name__ == "__main__":
    main()
