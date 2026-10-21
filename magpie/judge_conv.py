"""Ocena całych rozmów SFT modelem basal (basal-serve, endpoint /v1/systemone).

Każda rozmowa (np. z polskie-sprawy-v8) = jeden stan z całą treścią (wszystkie tury użytkownika
i asystenta, razem z rozumowaniem asystenta; --without-reasoning je pomija) i pytania tak/nie
liczone na tym stanie:
  halucynacje           — czy odpowiedzi asystenta zawierają nieprawdziwe lub zmyślone informacje,
  dobry_przyklad_sft    — czy rozmowa jest dobrym przykładem do treningu imitacyjnego SFT,
  rozumowanie_sensowne  — tylko gdy rozmowa ma rozumowanie (i jest w stanie): czy jest sensowne,
  powtorzone_pytania    — tylko rozmowy wieloturowe: czy druga i kolejne wiadomości użytkownika
                          mają bardzo podobną treść do pierwszej,
  powtorzone_odpowiedzi — tylko rozmowy wieloturowe: to samo dla odpowiedzi asystenta.
Skrypt tylko ocenia: zapisuje P(tak) na pytanie, bez progów i werdyktów — filtrowanie osobno.

Wejście: sft.jsonl (id, messages[, meta]). Wyjście (w --out): scored.jsonl — id, liczba tur,
thinking, basal {pytanie: P(tak)}, status (scored / za_dlugie / error), model; bez kopii rozmów
(łączyć z sft.jsonl po id). Rozmowy dłuższe niż --max-chars nie są wysyłane (status "za_dlugie").
Wznowienie = ten sam wiersz poleceń (statusy "error" i "za_dlugie" są sprawdzane od nowa).

  python -m magpie.judge_conv --url http://192.168.1.26:6001-6008 \\
      --input data/polskie_sprawy_v8/sft.jsonl --out data/judge_v8 --limit 1000
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter

import aiohttp
from aiolimiter import AsyncLimiter

from .verify import p_true, parse_urls

CHECKS = {
    "halucynacje": {
        "instructions": "Czy odpowiedzi asystenta zawierają nieprawdziwe informacje albo zmyślone fakty, "
                        "nazwy, liczby, cytaty lub źródła?",
    },
    "dobry_przyklad_sft": {
        "instructions": "Czy ta rozmowa jest dobrym przykładem do treningu imitacyjnego SFT polskiego asystenta?",
    },
}
# Zadawane tylko rozmowom z co najmniej dwiema turami użytkownika.
MULTITURN_CHECKS = {
    "powtorzone_pytania": {
        "instructions": "Czy druga i kolejne wiadomości użytkownika mają bardzo podobną treść do jego "
                        "pierwszej wiadomości?",
    },
    "powtorzone_odpowiedzi": {
        "instructions": "Czy druga i kolejne odpowiedzi asystenta mają bardzo podobną treść do jego "
                        "pierwszej odpowiedzi?",
    },
}
# Zadawane tylko rozmowom z rozumowaniem (gdy rozumowanie jest w stanie).
REASONING_CHECK = {
    "rozumowanie_sensowne": {
        "instructions": "Czy rozumowanie asystenta jest sensowne?",
    },
}

STATE_INTRO = (
    "Budujemy zbiór danych do treningu imitacyjnego SFT polskiego asystenta AI (czatbota): model uczy się "
    "odpowiadać dokładnie tak, jak asystent w poniższej rozmowie. Oceniamy, czy ta rozmowa nadaje się "
    "do takiego zbioru.\n\n"
    "Dobry przykład to rozmowa, w której asystent odpowiada zgodnie z prawdą, rzeczowo i pomocnie, "
    "na temat pytania, poprawną polszczyzną i w formie, jakiej chcielibyśmy uczyć model. Zły przykład "
    "to rozmowa z błędami merytorycznymi lub zmyślonymi faktami, z odpowiedzią nie na temat, urwaną, "
    "powtarzającą się, niezrozumiałą, w złym języku albo z odmową bez powodu.\n\n"
)
REASONING_INTRO = (
    "Przed odpowiedzią asystent czasem rozumuje (fragment oznaczony jako rozumowanie asystenta); model "
    "uczy się także tego rozumowania. Sensowne rozumowanie jest logiczne, dotyczy pytania, prowadzi do "
    "udzielonej odpowiedzi i nie zawiera zapętleń, powtórzeń ani wewnętrznych sprzeczności.\n\n"
)
ROLES = {"user": "Użytkownik", "assistant": "Asystent", "system": "System"}


def render(row: dict, with_reasoning: bool) -> str:
    parts = []
    for m in row["messages"]:
        text = m["content"]
        if with_reasoning and m.get("reasoning_content"):
            text = f"(rozumowanie asystenta)\n{m['reasoning_content']}\n(odpowiedź)\n{text}"
        parts.append(f"[{ROLES.get(m['role'], m['role'])}]\n{text}")
    return "\n\n".join(parts)


def checks_for(row: dict, with_reasoning: bool) -> dict:
    has_reasoning = with_reasoning and any(m.get("reasoning_content") for m in row["messages"])
    multiturn = sum(m["role"] == "user" for m in row["messages"]) >= 2
    return {**CHECKS, **(REASONING_CHECK if has_reasoning else {}), **(MULTITURN_CHECKS if multiturn else {})}


def build_request(row: dict, with_reasoning: bool) -> dict:
    checks = checks_for(row, with_reasoning)
    intro = STATE_INTRO + (REASONING_INTRO if "rozumowanie_sensowne" in checks else "")
    return {
        "state": intro + f"Rozmowa:\n\"\"\"\n{render(row, with_reasoning)}\n\"\"\"",
        "questions": {k: {"type": "noul", "instructions": c["instructions"]} for k, c in checks.items()},
    }




async def ask(session, limiter, args, row, stats, widx) -> dict:
    base = {"id": row["id"], "turns": sum(m["role"] == "user" for m in row["messages"]),
            "thinking": (row.get("meta") or {}).get("thinking")}
    request = build_request(row, args.with_reasoning)
    if args.max_chars and len(request["state"]) > args.max_chars:
        return {**base, "status": "za_dlugie", "chars": len(request["state"])}
    last_err = None
    for attempt in range(args.retries + 1):
        url = args.urls[(widx + attempt) % len(args.urls)]
        async with limiter:
            pass
        try:
            async with asyncio.timeout(args.timeout):
                async with session.post(f"{url}/v1/systemone", json=request) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                    body = await resp.json()
            scores = {k: round(p_true(body["answers"][k]), 4) for k in request["questions"]}
            stats[url] += 1
            return {**base, "basal": scores, "status": "scored", "basal_model": body.get("model"),
                    "chars": len(request["state"])}
        except TimeoutError:
            stats["timeout"] += 1
            last_err = f"timeout > {args.timeout}s"
        except (aiohttp.ClientError, RuntimeError, KeyError, ValueError) as e:
            stats["http_err"] += 1
            last_err = repr(e)[:200]
        if attempt < args.retries:
            await asyncio.sleep(2 * (attempt + 1))
    return {**base, "status": "error", "error": last_err}


async def run(args, rows: list[dict], out_path: str) -> None:
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
        print(f"{n}/{len(rows)}  ocenione {stats['scored']}  za długie {stats['za_dlugie']}  "
              f"błędy {stats['error']}  timeouty {stats['timeout']}  "
              f"{rate:.1f} rozm/s  ETA {(len(rows) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    async def worker(session, widx: int) -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res = await ask(session, limiter, args, row, stats, widx)
            stats["done"] += 1
            stats[res["status"]] += 1
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


# Pytania, przy których „tak” oznacza zaletę (przy pozostałych „tak” = problem).
GOOD_IF_YES = {"dobry_przyklad_sft", "rozumowanie_sensowne"}


def table(header: list[str], rows: list[list[str]], right: set[int]) -> str:
    """Tabela z ramkami; kolumny o indeksach z `right` wyrównane do prawej."""
    width = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    cell = lambda r: "│ " + " │ ".join(str(x).rjust(w) if k in right else str(x).ljust(w)
                                       for k, (x, w) in enumerate(zip(r, width))) + " │"
    line = lambda a, b, c: a + b.join("─" * (w + 2) for w in width) + c
    return "\n".join([line("┌", "┬", "┐"), cell(header), line("├", "┼", "┤"),
                      *map(cell, rows), line("└", "┴", "┘")])


def summarize(out_path: str) -> None:
    rows = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and rows[r["id"]]["status"] == "scored":
                continue  # nie nadpisuj udanej oceny późniejszym błędem
            rows[r["id"]] = r
    status = Counter(r["status"] for r in rows.values())
    print(f"\nrozmów: {len(rows)}  " + "  ".join(f"{k}: {v}" for k, v in status.items()))
    scores: dict[str, list[float]] = {}
    for r in rows.values():
        for k, v in r.get("basal", {}).items():
            scores.setdefault(k, []).append(v)
    body = []
    for k, v in scores.items():
        q = statistics.quantiles(v, n=10) if len(v) > 1 else v * 9
        good = k in GOOD_IF_YES
        flagged = sum((x < 0.5) if good else (x > 0.5) for x in v)
        body.append([k, "zaleta" if good else "problem", f"{len(v):,}", f"{statistics.mean(v):.3f}",
                     f"{q[0]:.3f}", f"{q[4]:.3f}", f"{q[8]:.3f}", f"{flagged:,} ({100 * flagged / len(v):.1f}%)"])
    if body:
        print("P(tak) na pytanie; flaga = P > 0,5 przy problemie albo P < 0,5 przy zalecie")
        print(table(["pytanie", "„tak” =", "rozmów", "śr.", "p10", "p50", "p90", "flaga"], body,
                    right={2, 3, 4, 5, 6, 7}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://192.168.1.26:6001-6008",
                   help="adres(y) basal-serve bez /v1: lista po przecinku lub zakres portów host:6001-6008")
    p.add_argument("--input", required=True, help="sft.jsonl z rozmowami (id, messages)")
    p.add_argument("--out", required=True, help="katalog wynikowy")
    p.add_argument("--limit", type=int, default=0, help="tylko pierwsze N rozmów (0 = wszystkie)")
    p.add_argument("--without-reasoning", dest="with_reasoning", action="store_false",
                   help="pomiń rozumowanie asystenta (wtedy bez pytania rozumowanie_sensowne)")
    p.add_argument("--max-chars", type=int, default=30000, help="dłuższych stanów nie wysyłaj (0 = bez limitu)")
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--rate", type=float, default=10000, help="maks. startów zapytań na sekundę")
    p.add_argument("--timeout", type=float, default=120, help="limit na jedno zapytanie (s)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--dry-run", action="store_true", help="wypisz zapytanie dla pierwszej rozmowy i wyjdź")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    args.urls = parse_urls(args.url)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "scored.jsonl")
    if args.summary_only:
        summarize(out_path)
        return
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    if args.dry_run:
        print(json.dumps(build_request(rows[0], args.with_reasoning), ensure_ascii=False, indent=2))
        return
    done: set = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            # za_dlugie też od nowa: po zmianie --max-chars mogą się już zmieścić.
            done = {r["id"] for r in map(json.loads, f) if r["status"] == "scored"}
    todo = [r for r in rows if r["id"] not in done]
    print(f"rozmów: {len(rows)}, ocenionych: {len(done)}, do zrobienia: {len(todo)}  (instancje {len(args.urls)}, "
          f"concurrency {args.concurrency}, timeout {args.timeout}s, "
          f"rozumowanie {'tak' if args.with_reasoning else 'nie'}, max {args.max_chars} znaków)", flush=True)
    if todo:
        asyncio.run(run(args, todo, out_path))
    summarize(out_path)


if __name__ == "__main__":
    main()
