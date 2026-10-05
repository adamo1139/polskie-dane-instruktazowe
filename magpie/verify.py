"""Weryfikacja wygenerowanych pytań modelem basal (basal-serve, endpoint /v1/systemone).

Każda wiadomość = jeden stan (state) z kilkoma pytaniami tak/nie (noul) w jednym zapytaniu —
basal liczy je na wspólnym stanie. Dla każdego pytania zapisujemy P(tak) i to, czy wynik jest
zgodny z oczekiwanym (CHECKS[...]["want"]). Wiadomość przechodzi, gdy dla każdego pytania
P(oczekiwanej odpowiedzi) >= --threshold (domyślnie 0.7 — celowo agresywnie: lepiej odrzucić
za dużo niż przepuścić śmieci).

Wejście: accepted.jsonl z magpie.generate (pola id, raw, topic).
Wyjście (w --out):
  verified.jsonl  — wiersz wejściowy + basal: {pytanie: P(tak)}, failed: [pytania], verdict
Wznowienie = ten sam wiersz poleceń; gotowe id są pomijane.
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

# Opis zadania i definicje dobrej/złej wiadomości idą do stanu (wspólny dla wszystkich pytań
# w zapytaniu — zalecenie basala: fakty i reguły w stanie). Pytania: jeden warunek na pytanie;
# want = odpowiedź oznaczająca dobrą wiadomość.
CHECKS = {
    "prosba_do_czatbota": {
        "want": True,
        "instructions": "Czy wiadomość zawiera pytanie, prośbę, polecenie lub zadanie dla czatbota?",
    },
    "wpisalby_czlowiek": {
        "want": True,
        "instructions": "Czy to jest tekst, który człowiek mógłby sam wpisać do czatbota?",
    },
    "instrukcja_pisania": {
        "want": False,
        "instructions": "Czy wiadomość opisuje, jak ma wyglądać pytanie lub wiadomość, którą ktoś ma napisać?",
    },
    "glos_czatbota": {
        "want": False,
        "instructions": "Czy wiadomość jest napisana głosem czatbota lub systemu, a nie użytkownika?",
    },
    "po_polsku": {
        "want": True,
        "instructions": "Czy wiadomość jest zrozumiałym tekstem po polsku?",
    },
    "zawiera_odpowiedz": {
        "want": False,
        "instructions": "Czy po pytaniu lub prośbie wiadomość od razu zawiera odpowiedź na nie?",
    },
}

STATE_INTRO = (
    "Budujemy zbiór danych typu instruct do SFT, na którym będzie trenowany polski asystent AI "
    "(czatbot). Każdy przykład zaczyna się od pierwszej wiadomości użytkownika w nowej rozmowie, "
    "a model uczy się na nią odpowiadać. Oceniamy, czy poniższa wiadomość nadaje się na taką "
    "pierwszą wiadomość.\n\n"
    "Dobra wiadomość to wszystko, co człowiek wpisuje do czatbota: pytanie, prośba, polecenie lub "
    "zadanie (np. napisz esej, ułóż dialog, rozwiąż zadanie, popraw tekst), także z wymaganiami co "
    "do formy odpowiedzi. Polecenie jest dobre, gdy użytkownik prosi o coś dla siebie; instrukcja, "
    "jak ma wyglądać pytanie albo wiadomość do napisania, jest zła. W porządku są też odgrywanie ról "
    "i wiadomości skierowane "
    "do postaci lub konkretnych osób, styl potoczny, literówki oraz cytowanie błędnych form "
    "językowych, o które użytkownik pyta.\n\n"
    "Zła wiadomość to: instrukcja o generowaniu pytań lub danych, tekst pisany głosem czatbota albo "
    "systemu, odwołanie do czegoś, czego tu nie ma (wcześniejszej rozmowy, tekstu, listy, zdjęcia), "
    "wypowiedź bez żadnej "
    "prośby (samo stwierdzenie, esej, ogłoszenie) albo wiadomość, która od razu zawiera odpowiedź "
    "na własne pytanie.\n\n"
)


def build_state(row: dict) -> str:
    return STATE_INTRO + f"Wiadomość użytkownika:\n\"\"\"\n{row['raw']}\n\"\"\""


def build_request(row: dict) -> dict:
    return {
        "state": build_state(row),
        "questions": {k: {"type": "noul", "instructions": c["instructions"]} for k, c in CHECKS.items()},
    }


def parse_urls(spec: str) -> list[str]:
    """„a,b” → lista; „http://host:6001-6008” → 8 adresów (po jednym na instancję basal-serve)."""
    urls = []
    for part in spec.split(","):
        m = re.fullmatch(r"(.*:)(\d+)-(\d+)/?", part.strip())
        if m:
            urls += [f"{m[1]}{p}" for p in range(int(m[2]), int(m[3]) + 1)]
        elif part.strip():
            urls.append(part.strip().rstrip("/"))
    return urls


def p_true(answer: dict) -> float:
    probs = answer.get("probabilities") or {}
    return float(probs["true"] if "true" in probs else answer["noul"])


def judge(answers: dict, threshold: float) -> tuple[dict, list[str]]:
    scores = {k: round(p_true(answers[k]), 4) for k in CHECKS}
    failed = [k for k, c in CHECKS.items()
              if (scores[k] if c["want"] else 1 - scores[k]) < threshold]
    return scores, failed


async def ask(session: aiohttp.ClientSession, limiter: AsyncLimiter, args: argparse.Namespace,
              row: dict, stats: Counter, widx: int) -> dict:
    last_err = None
    for attempt in range(args.retries + 1):
        # Worker ma „swoją” instancję; każde ponowienie idzie do następnej (omija padniętą).
        url = args.urls[(widx + attempt) % len(args.urls)]
        async with limiter:
            pass  # limiter ogranicza tempo startów; slot trzyma worker
        try:
            async with asyncio.timeout(args.timeout):
                async with session.post(f"{url}/v1/systemone", json=build_request(row)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                    body = await resp.json()
            scores, failed = judge(body["answers"], args.threshold)
            stats[url] += 1
            return {**row, "basal": scores, "failed": failed, "verdict": "ok" if not failed else "fail",
                    "basal_model": body.get("model"), "basal_url": url,
                    "basal_latency_ms": body.get("usage", {}).get("latency_ms")}
        except TimeoutError:
            stats["timeout"] += 1
            last_err = f"timeout > {args.timeout}s"
        except (aiohttp.ClientError, RuntimeError, KeyError, ValueError) as e:
            stats["http_err"] += 1
            last_err = repr(e)[:200]
        if attempt < args.retries:
            await asyncio.sleep(2 * (attempt + 1))
    return {**row, "verdict": "error", "error": last_err}


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
        rate = n / max(time.monotonic() - t0, 1e-9)
        print(f"{n}/{len(rows)}  ok {stats['ok']} ({100 * stats['ok'] / max(n, 1):.0f}%)  "
              f"fail {stats['fail']}  błędy {stats['error']}  timeouty {stats['timeout']}  "
              f"{rate:.1f} wiad/s  ETA {(len(rows) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
        print("    per instancja:", "  ".join(f"{u.rsplit(':', 1)[-1]}={stats[u]}" for u in args.urls), flush=True)

    async def worker(session: aiohttp.ClientSession, widx: int) -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res = await ask(session, limiter, args, row, stats, widx)
            stats["done"] += 1
            stats[res["verdict"]] += 1
            buffer.append(res)
            if len(buffer) >= args.flush_every:
                flush()
            if stats["done"] % args.progress_every == 0:
                progress()

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.gather(*(worker(session, i) for i in range(min(args.concurrency, len(rows)))))
    finally:
        flush()
        fout.close()
        progress()


def summarize(out_path: str) -> None:
    rows = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and r["verdict"] == "error":
                continue  # nie nadpisuj udanej oceny późniejszym błędem
            rows[r["id"]] = r
    verdicts = Counter(r["verdict"] for r in rows.values())
    failed = Counter(k for r in rows.values() for k in r.get("failed", []))
    print(f"\nocenionych: {len(rows)}  {dict(verdicts)}")
    print("niezaliczone pytania:", {k: failed[k] for k in CHECKS if failed[k]})
    print("odrzuty per temat:", dict(Counter(r["topic"] for r in rows.values() if r["verdict"] == "fail")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://192.168.1.26:6001-6008",
                   help="adres(y) basal-serve bez /v1: lista po przecinku lub zakres portów host:6001-6008")
    p.add_argument("--input", required=True, help="accepted.jsonl z magpie.generate")
    p.add_argument("--out", required=True, help="katalog wynikowy")
    p.add_argument("--limit", type=int, default=0, help="tylko pierwsze N wiadomości (0 = wszystkie)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="min. P(oczekiwanej odpowiedzi) dla każdego pytania")
    p.add_argument("--concurrency", type=int, default=1000)
    p.add_argument("--rate", type=float, default=10000, help="maks. startów zapytań na sekundę")
    p.add_argument("--timeout", type=float, default=60, help="limit na jedno zapytanie (s)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--dry-run", action="store_true", help="wypisz zapytanie dla pierwszej wiadomości i wyjdź")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    args.urls = parse_urls(args.url)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "verified.jsonl")
    if args.summary_only:
        summarize(out_path)
        return

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[:args.limit]
    if args.dry_run:
        print(json.dumps(build_request(rows[0]), ensure_ascii=False, indent=2))
        return

    done: set[int] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, f) if r["verdict"] != "error"}
    todo = [r for r in rows if r["id"] not in done]
    print(f"wiadomości: {len(rows)}, ocenionych: {len(done)}, do zrobienia: {len(todo)}  "
          f"(instancje {len(args.urls)}, concurrency {args.concurrency}, timeout {args.timeout}s, "
          f"próg {args.threshold})", flush=True)
    if todo:
        asyncio.run(run(args, todo, out_path))
    summarize(out_path)


if __name__ == "__main__":
    main()
