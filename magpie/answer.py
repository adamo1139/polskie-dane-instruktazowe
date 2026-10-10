"""Generacja odpowiedzi asystenta (np. Bielik 11B v3.0) na pytania z Magpie — asyncio, wznawialna.

Wejście: verified.jsonl z magpie.verify (bierzemy verdict == "ok") albo accepted.jsonl z
magpie.generate (ok == True). Zapytania idą do /v1/chat/completions (vLLM i SGLang) ze streamingiem.

Rozumowanie: --thinking off|on → zmienna szablonu enable_thinking (albo dowolne --template-kwargs,
np. {"reasoning_effort": "high"} dla Mistrala Small 4). --backend vllm wysyła je w chat_template_kwargs,
--backend tabby (TabbyAPI/exl3) w template_vars. Rozumowanie zapisujemy osobno: z pola
reasoning_content (gdy serwer ma parser reasoningu) albo z <think>…</think> / [THINK]…[/THINK] w treści.
Prompt systemowy (--system) służy do generowania i domyślnie nie trafia do sft.jsonl (--system-in-sft).

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
from collections import Counter, deque

import aiohttp
from aiolimiter import AsyncLimiter

# Domyślny prompt systemowy przy --thinking on: bez niego modele zwykle myślą po angielsku.
SYSTEM_PL_REASONING = ("Jesteś pomocnym asystentem. Zanim odpowiesz, przemyśl problem krok po kroku. "
                       "Całe rozumowanie prowadź wyłącznie po polsku, a odpowiedź również napisz po polsku.")
SPEED_WINDOW = 300  # s — prędkość i ETA z ostatnich 5 minut, nie średnia od startu
LEFTOVER_THINK = re.compile(r"\[/?THINK\]|</?think>")
THINK = re.compile(r"^\s*(?:<think>(.*?)</think>|\[THINK\](.*?)\[/THINK\])\s*", re.DOTALL)


class StreamTimeout(Exception):
    pass


def split_reasoning(content: str, reasoning: str) -> tuple[str, str]:
    # Bez parsera reasoningu po stronie serwera rozumowanie przychodzi w treści jako <think>…</think>.
    m = THINK.match(content)
    if m:
        reasoning = reasoning or (m[1] if m[1] is not None else m[2]).strip()
        content = content[m.end():]
    return content.strip(), reasoning.strip()


async def chat(session: aiohttp.ClientSession, args: argparse.Namespace, row: dict) -> dict:
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": row["raw"]}]
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": not args.no_stream,
    }
    if args.max_tokens:
        payload["max_tokens"] = args.max_tokens
    for k in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if getattr(args, k) is not None:
            payload[k] = getattr(args, k)
    if args.template_kwargs:
        payload["template_vars" if args.backend == "tabby" else "chat_template_kwargs"] = args.template_kwargs
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    t0 = time.monotonic()
    ttft = None
    content, reasoning = [], []
    finish = None
    async with asyncio.timeout(args.total_timeout):
        async with session.post(f"{args.url}/chat/completions", json=payload, headers=args.headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
            if args.no_stream:
                body = await resp.json()
                choice = body["choices"][0]
                msg = choice.get("message") or {}
                content.append(msg.get("content") or "")
                reasoning.append(msg.get("reasoning_content") or msg.get("reasoning") or "")
                finish = choice.get("finish_reason")
                answer, thought = split_reasoning("".join(content), "".join(reasoning))
                return {"answer": answer, "reasoning": thought, "finish": finish,
                        "ttft_s": None, "latency_s": round(time.monotonic() - t0, 2)}
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
    base.update(model=args.model, thinking=args.thinking, template_kwargs=args.template_kwargs,
                reasoning_effort=args.reasoning_effort)
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            out = await chat(session, args, row)
            # Niezamknięty blok rozumowania (np. [THINK] bez [/THINK]) — nie da się oddzielić myślenia od odpowiedzi.
            unclosed = bool(LEFTOVER_THINK.search(out["answer"]))
            ok = out["finish"] == "stop" and bool(out["answer"]) and not unclosed
            out["unclosed_reasoning"] = unclosed
            stats["ok" if ok else ("niezamknięte rozumowanie" if unclosed else "ucięte/puste")] += 1
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
    recent: deque = deque()  # czasy ukończenia odpowiedzi z ostatnich SPEED_WINDOW sekund
    fout = open(out_path, "a", encoding="utf-8")

    def flush() -> None:
        if buffer:
            fout.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in buffer))
            fout.flush()
            os.fsync(fout.fileno())
            buffer.clear()

    def progress() -> None:
        n = stats["done"]
        now = time.monotonic()
        while recent and recent[0] < now - SPEED_WINDOW:
            recent.popleft()
        speed = len(recent) / max(min(SPEED_WINDOW, now - t0), 1e-9)
        print(f"{n}/{len(rows)}  ok {stats['ok']}  ucięte/puste {stats['ucięte/puste']}  "
              f"niezamknięte rozum. {stats['niezamknięte rozumowanie']}  "
              f"błędy {stats['error']}  timeouty {stats['timeout']}  http {stats['http_err']}  "
              f"{speed:.2f} odp/s (ost. 5 min)  ETA {(len(rows) - n) / max(speed, 1e-9) / 60:.0f} min", flush=True)

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
            recent.append(time.monotonic())
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


def finalize(out_dir: str, system: str | None = None) -> None:
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
                                "meta": {k: r.get(k) for k in ("topic", "persona", "model", "thinking",
                                                               "template_kwargs", "reasoning_effort")
                                         if r.get(k) is not None}},
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
    p.add_argument("--backend", choices=["vllm", "tabby"], default="vllm",
                   help="vllm: chat_template_kwargs; tabby (TabbyAPI/exl3): template_vars")
    p.add_argument("--template-kwargs", default=None,
                   help='zmienne szablonu jako JSON, np. \'{"reasoning_effort": "high"}\' '
                        '(domyślnie {"enable_thinking": <thinking>})')
    p.add_argument("--reasoning-effort", default=None,
                   help="pole reasoning_effort w zapytaniu (np. high) — dla vLLM z tokenizerem Mistrala, który "
                        "odrzuca chat_template_kwargs; wtedy zmienne szablonu wysyłane tylko z --template-kwargs")
    p.add_argument("--no-stream", action="store_true",
                   help="zapytania bez streamingu — vLLM z tokenizerem Mistrala psuje w strumieniu tokeny "
                        "[THINK]/[/THINK] (np. „fony”, „</analysis>”); bez streamingu przychodzą poprawnie")
    p.add_argument("--api-key", default=None, help="klucz API (nagłówek Authorization: Bearer)")
    p.add_argument("--system", default=None,
                   help="prompt systemowy do generowania (domyślnie przy --thinking on: SYSTEM_PL_REASONING, "
                        "„none” = bez promptu)")
    p.add_argument("--system-in-sft", action="store_true", help="zapisz prompt systemowy w sft.jsonl")
    p.add_argument("--limit", type=int, default=0, help="tylko pierwsze N pytań (0 = wszystkie)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    # Niepodane = nie wysyłane; vLLM bierze wtedy wartości z generation_config.json modelu.
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="domyślnie 5000 (off) / 6000 (on) — prompt + odpowiedź muszą zmieścić się w kontekście serwera; "
                        "0 = nie wysyłaj max_tokens (serwer generuje do końca kontekstu)")
    p.add_argument("--concurrency", type=int, default=500)
    p.add_argument("--rate", type=float, default=100, help="maks. startów zapytań na sekundę")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--ttft-timeout", type=float, default=7200, help="czekanie na pierwszy token — obejmuje kolejkę serwera")
    p.add_argument("--chunk-timeout", type=float, default=3600, help="maks. przerwa w strumieniu — obejmuje wywłaszczenie z KV cache")
    p.add_argument("--total-timeout", type=float, default=14400)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    args.url = args.url.rstrip("/")
    if args.max_tokens is None:
        args.max_tokens = 6000 if args.thinking == "on" else 5000
    if args.system is None and args.thinking == "on":
        args.system = SYSTEM_PL_REASONING
    elif args.system == "none":
        args.system = None
    args.template_kwargs = (json.loads(args.template_kwargs) if args.template_kwargs
                            else None if args.reasoning_effort else {"enable_thinking": args.thinking == "on"})
    args.headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    sft_system = args.system if args.system_in_sft else None

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "answers.jsonl")
    if args.finalize_only:
        finalize(args.out, sft_system)
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
            async with aiohttp.ClientSession() as s, s.get(f"{args.url}/models", headers=args.headers) as resp:
                return (await resp.json())["data"][0]["id"]
        args.model = asyncio.run(first_model())
    print(f"pytań: {len(rows)}, gotowych: {len(done)}, do zrobienia: {len(todo)}  (model {args.model}, "
          f"thinking {args.thinking}, {args.backend}: szablon {args.template_kwargs}, reasoning_effort "
          f"{args.reasoning_effort}, max_tokens {args.max_tokens}, temperature {args.temperature}, top_p {args.top_p}, "
          f"top_k {args.top_k}, min_p {args.min_p}, presence_penalty {args.presence_penalty}, "
          f"repetition_penalty {args.repetition_penalty}, "
          f"concurrency {args.concurrency})", flush=True)
    if todo:
        asyncio.run(run(args, todo, out_path))
    finalize(args.out, sft_system)


if __name__ == "__main__":
    main()
