"""Rozmowy wieloturowe: kolejne tury użytkownika pisze Glimmer (Magpie z historią), odpowiada
magpie.answer innym modelem (Bielik 11B / Mistral / private). Runda po rundzie, bo oba modele nie
mieszczą się naraz na GPU:

  select  — losuje rozmowy z sft.jsonl (system, user, assistant) i przypisuje im docelową liczbę tur
            (domyślnie 80% → 2 tury, 20% → 5 tur); tylko rozmowy bez rozumowania (meta.thinking off),
            bo prepare_polskie_sprawy_v3.py wymaga zgodności trybu we wszystkich turach.
  ask     — dla rozmów, którym brakuje tur, generuje następną turę użytkownika: prompt glimmer8
            rozmowy + historia (bez rozumowania) + otwarta tura user. Wynik: accepted.jsonl
            w formacie wejścia magpie.answer (raw = nowe pytanie, history = dotychczasowe wiadomości).
  answer  — zwykłe magpie.answer na accepted.jsonl; jego sft.jsonl to wejście następnego `ask`.

  python -m magpie.multiturn select --input data/polskie_sprawy_v4_sys/sft.jsonl --out data/mt/round1.jsonl
  python -m magpie.multiturn ask --input data/mt/round1.jsonl --out data/mt/ask_t2
  python -m magpie.answer --thinking off --input data/mt/ask_t2/accepted.jsonl --out data/mt/answers_t2 ...
  python -m magpie.multiturn ask --input data/mt/answers_t2/sft.jsonl --out data/mt/ask_t3
  … (do tury 5; rozmowy 2-turowe kończą się po answers_t2)

  pipeline — jeden model (Bielik po SFT, szablon v13) pisze i pytania, i odpowiedzi: wszystkie
            brakujące tury rozmowy w jednym przebiegu, zapytanie po zapytaniu (/v1/completions):
            historia + otwarta tura user → pytanie; to samo + pytanie + otwarta tura asystenta →
            odpowiedź; to samo + odpowiedź + otwarta tura user → kolejne pytanie… Każde zapytanie
            przedłuża poprzednie, więc prefix caching (RadixAttention) trzyma całą rozmowę.
            Odrzucone pytanie/odpowiedź dostaje od razu kolejne próby (--attempts, nowy seed);
            rozmowa, która utknie, zostaje z turami zrobionymi do tej pory. Wynik: raw.jsonl i po
            zakończeniu sft.jsonl (rozmowy z ≥ 2 turami użytkownika; bez systemu, chyba że
            --system-in-sft).

  python -m magpie.multiturn pipeline --input data/mt_v6/round1.jsonl --out data/mt_v6/pipeline \\
      --model <model na serwerze> --context 8192 --answer-temperature 0.9 --answer-frequency-penalty 0.03

Filtry tury użytkownika: reguły pierwszej tury (filters.classify) poza „kontynuacją rozmowy”
(„Dzięki, a…”, „A jeśli…” są tu naturalne) + powtórzenie wcześniejszego pytania + echo odpowiedzi.
Wznowienie = ten sam wiersz poleceń; gotowe id (rozmowy) są pomijane. --retry-rejected N: rozmowa,
której pytanie odrzuciły filtry, dostaje do N kolejnych prób z nowym seedem (np. ponowne uruchomienie
po zakończonym runie); do accepted.jsonl trafia pierwsza przyjęta próba, raw.jsonl trzyma wszystkie.
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
from difflib import SequenceMatcher

from .filters import classify, clean
from .generate import run
from .pipeline import ASSISTANT_OPEN, answer_args, call, split_answer
from .prompt import TEMPLATES, TURNS, build_history_prompt


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def classify_followup(out: dict, job: dict, repeat_ratio: float) -> str | None:
    reason = classify(out, job["system"], skip=("kontynuacja rozmowy",))
    if reason:
        return reason
    new = normalize(out["raw"])
    for m in job["history"]:
        if m["role"] == "user" and SequenceMatcher(None, new, normalize(m["content"])).ratio() >= repeat_ratio:
            return "powtórzenie pytania"
    last_answer = next(m["content"] for m in reversed(job["history"]) if m["role"] == "assistant")
    if shingles(out["raw"]) & shingles(last_answer):
        return "echo odpowiedzi"
    return None


def cmd_select(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    pool = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            roles = [m["role"] for m in r["messages"]]
            if roles == ["system", "user", "assistant"] and r["meta"].get("thinking") == args.thinking:
                pool.append(r)
    if len(pool) < args.n:
        raise SystemExit(f"Tylko {len(pool)} rozmów spełnia warunki, potrzeba {args.n}")
    chosen = rng.sample(pool, args.n)
    long_n = round(args.n * args.long_share)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} istnieje — nie nadpisuję")
    with open(args.out, "x", encoding="utf-8") as f:
        for i, r in enumerate(chosen):
            r["target_turns"] = args.long_turns if i < long_n else args.short_turns
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pula ({args.thinking}): {len(pool)}, wybrano {args.n}: {long_n} × {args.long_turns} tur, "
          f"{args.n - long_n} × {args.short_turns} tur  (seed {args.seed})  →  {args.out}")


def build_jobs(args: argparse.Namespace, attempts: dict[str, int] | None = None) -> list[dict]:
    attempts = attempts or {}
    jobs = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r["messages"]
            turn = sum(m["role"] == "user" for m in msgs) + 1
            target = args.target_turns or r["target_turns"]
            if turn > target or msgs[-1]["role"] != "assistant" or msgs[0]["role"] != "system":
                continue
            history = [{"role": m["role"], "content": m["content"]} for m in msgs]  # bez reasoning_content
            prompt = build_history_prompt(history, args.template)
            if args.max_prompt_chars and len(prompt) > args.max_prompt_chars:
                continue
            jobs.append({"id": r["id"], "turn": turn, "target_turns": target,
                         "system": msgs[0]["content"], "history": history, "meta": r["meta"],
                         "topic": r["meta"].get("topic"), "persona": r["meta"].get("persona"),
                         # Próba 0 ma seed jak przed --retry-rejected; kolejne dostają nowy.
                         "attempt": attempts.get(r["id"], 0),
                         "seed": int(hashlib.sha256((f"{r['id']}:{turn}" + (
                             f":{attempts[r['id']]}" if attempts.get(r["id"]) else "")).encode()).hexdigest()[:8], 16),
                         "prompt": prompt})
    return jobs


def finalize(out_dir: str) -> None:
    rows = {}
    with open(os.path.join(out_dir, "raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            old = rows.get(r["id"])
            # pierwsza przyjęta próba wygrywa; błąd nie nadpisuje odrzutu
            if old and (old["ok"] or r.get("finish") == "error"):
                continue
            rows[r["id"]] = r
    accepted = [r for r in sorted(rows.values(), key=lambda r: r["id"]) if r["ok"]]
    with open(os.path.join(out_dir, "accepted.jsonl"), "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    reasons = Counter(r.get("reason") for r in rows.values() if not r["ok"])
    print(f"\nrozmów: {len(rows)}  przyjętych pytań: {len(accepted)}  "
          f"tury: {dict(Counter(r['turn'] for r in accepted))}  "
          f"przyjęte w próbie: {dict(sorted(Counter(r.get('attempt', 0) for r in accepted).items()))}")
    print("odrzuty:", dict(reasons.most_common(12)))


def cmd_ask(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.jsonl")
    if args.finalize_only:
        finalize(args.out)
        return
    accepted: set[str] = set()
    rejected: Counter = Counter()  # liczba odrzuconych prób (błędy HTTP/timeouty się nie liczą)
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                if r["ok"]:
                    accepted.add(r["id"])
                elif r.get("finish") != "error":
                    rejected[r["id"]] += 1
    jobs = build_jobs(args, attempts=rejected)
    todo = [j for j in jobs if j["id"] not in accepted and rejected[j["id"]] <= args.retry_rejected
            and (not args.retry_only or j["id"] in rejected)]
    print(f"rozmów z brakującą turą: {len(jobs)}  tury: {dict(Counter(j['turn'] for j in jobs))}  "
          f"gotowych: {len(jobs) - len(todo)}, do zrobienia: {len(todo)}  (model {args.model}, "
          f"szablon {args.template}, concurrency {args.concurrency}, rate {args.rate}/s)", flush=True)
    if todo:
        asyncio.run(run(args, todo, raw_path,
                        classifier=lambda out, job: classify_followup(out, job, args.repeat_ratio)))
    finalize(args.out)


def call_seed(conv_id: str, turn: int, stage: str, attempt: int) -> int:
    return int(hashlib.sha256(f"{conv_id}:{turn}:{stage}:{attempt}".encode()).hexdigest()[:8], 16)


async def pipeline_work(session, limiter, args, job, stats) -> dict:
    """Wszystkie brakujące tury jednej rozmowy; zwraca wiersz raw.jsonl z pełnymi messages."""
    msgs = [dict(m) for m in job["messages"]]
    meta = dict(job["meta"])
    meta["turn_models"] = meta.get("turn_models", [meta.get("model")])
    model = os.path.basename(args.model.rstrip("/"))
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]  # bez reasoning_content
    prompt = build_history_prompt(history, "v13")
    turn = sum(m["role"] == "user" for m in msgs) + 1
    row = {"id": job["id"], "target_turns": job["target_turns"]}
    tries: list[int] = []
    while turn <= job["target_turns"]:
        if args.context - len(prompt) // 2 < args.min_answer_tokens + args.max_tokens:
            return {**row, "messages": msgs, "meta": meta, "turns": turn - 1, "tries": tries,
                    "ok": False, "finish": "stop", "stage": "pytanie", "reason": "za długa historia"}
        question = reason = None
        for attempt in range(args.attempts):
            q = await call(session, limiter, args,
                           {"seed": call_seed(job["id"], turn, "q", attempt), "prompt": prompt}, stats)
            if q["finish"] == "error":
                return {**row, "turns": turn - 1, "ok": False, "finish": "error", "stage": "pytanie",
                        "reason": f"błąd: {q.get('error')}"}
            q = clean(q)
            reason = classify_followup(q, {"system": msgs[0]["content"], "history": history}, args.repeat_ratio)
            if not reason:
                question = q["raw"]
                tries.append(attempt + 1)
                break
            stats[f"odrzucone pytanie: {reason}"] += 1
        if question is None:
            return {**row, "messages": msgs, "meta": meta, "turns": turn - 1, "tries": tries,
                    "ok": False, "finish": "stop", "stage": "pytanie", "reason": reason}

        a_prompt = prompt + question + ASSISTANT_OPEN["off"]
        a_args = copy.copy(args.answer)
        budget = args.context - len(a_prompt) // 2 - 32
        a_args.max_tokens = min(args.answer_max_tokens, budget) if args.answer_max_tokens else budget
        answer = why = None
        for attempt in range(args.answer_attempts):
            a = await call(session, limiter, a_args,
                           {"seed": call_seed(job["id"], turn, "a", attempt), "prompt": a_prompt}, stats)
            if a["finish"] == "error":
                return {**row, "turns": turn - 1, "ok": False, "finish": "error", "stage": "odpowiedź",
                        "reason": f"błąd: {a.get('error')}"}
            _, text, why = split_answer(a["raw"], "off") if a["finish"] == "stop" else ("", "", "ucięte")
            if not why:
                answer = text
                break
            stats[f"odrzucona odpowiedź: {why}"] += 1
        if answer is None:
            return {**row, "messages": msgs, "meta": meta, "turns": turn - 1, "tries": tries,
                    "ok": False, "finish": "stop", "stage": "odpowiedź", "reason": why}

        msgs += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        meta["turn_models"] = meta["turn_models"] + [model]
        # Dokładnie to, co dopisałby build_history_prompt: koniec tury asystenta + otwarta tura user.
        prompt = a_prompt + answer + "<|im_end|>\n" + TURNS["v13"]["open_user"]
        turn += 1
    return {**row, "messages": msgs, "meta": meta, "turns": turn - 1, "tries": tries,
            "ok": True, "finish": "stop", "stage": "koniec", "reason": None}


def pipeline_finalize(out_dir: str, system_in_sft: bool) -> None:
    rows = {}
    with open(os.path.join(out_dir, "raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in rows and r.get("finish") == "error":
                continue
            rows[r["id"]] = r
    good = [r for r in sorted(rows.values(), key=lambda r: r["id"])
            if r.get("finish") != "error" and r.get("turns", 0) >= 2]
    with open(os.path.join(out_dir, "sft.jsonl"), "w", encoding="utf-8") as f:
        for r in good:
            msgs = r["messages"] if system_in_sft else [m for m in r["messages"] if m["role"] != "system"]
            f.write(json.dumps({"id": r["id"], "messages": msgs, "meta": r["meta"]}, ensure_ascii=False) + "\n")
    stopped = Counter(f"{r.get('stage')}: {r.get('reason')}" for r in rows.values() if not r["ok"])
    print(f"\nrozmów: {len(rows)}  do sft.jsonl: {len(good)}  tury użytkownika: "
          f"{dict(sorted(Counter(r['turns'] for r in good).items()))}  "
          f"próby na przyjęte pytanie: {dict(sorted(Counter(t for r in good for t in r.get('tries', [])).items()))}")
    print("zatrzymane:", dict(stopped.most_common(12)))


def cmd_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.jsonl")
    if args.finalize_only:
        pipeline_finalize(args.out, args.system_in_sft)
        return
    if not args.context:
        raise SystemExit("Podaj --context (= --context-length serwera; router SGLang go nie podaje)")
    args.url = args.url.rstrip("/")
    args.template = "v13"
    args.answer = answer_args(args)
    done: set[str] = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, f) if r.get("finish") != "error"}
    jobs = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r["messages"]
            target = args.target_turns or r["target_turns"]
            if r["id"] in done or msgs[0]["role"] != "system" or msgs[-1]["role"] != "assistant":
                continue
            if sum(m["role"] == "user" for m in msgs) >= target:
                continue
            jobs.append({"id": r["id"], "messages": msgs, "meta": r["meta"], "target_turns": target})
    print(f"rozmów do zrobienia: {len(jobs)}, gotowych: {len(done)}  (model {args.model}, kontekst {args.context}, "
          f"próby pytania/odpowiedzi {args.attempts}/{args.answer_attempts}, concurrency {args.concurrency}, "
          f"rate {args.rate}/s; pytanie T={args.temperature} top_p={args.top_p}; odpowiedź "
          f"T={args.answer_temperature} top_p={args.answer_top_p} freq={args.answer_frequency_penalty})", flush=True)
    if jobs:
        asyncio.run(run(args, jobs, raw_path, work_fn=pipeline_work))
    pipeline_finalize(args.out, args.system_in_sft)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="wylosuj rozmowy i docelowe liczby tur")
    s.add_argument("--input", required=True, help="sft.jsonl z rozmowami system, user, assistant")
    s.add_argument("--out", required=True, help="plik .jsonl (wejście pierwszego ask)")
    s.add_argument("--n", type=int, default=100000)
    s.add_argument("--long-share", type=float, default=0.2, help="udział długich rozmów")
    s.add_argument("--long-turns", type=int, default=5)
    s.add_argument("--short-turns", type=int, default=2)
    s.add_argument("--thinking", choices=["off", "on"], default="off", help="tylko rozmowy w tym trybie")
    s.add_argument("--seed", type=int, default=20261011)

    a = sub.add_parser("ask", help="wygeneruj następną turę użytkownika")
    a.add_argument("--input", required=True, help="wynik select albo sft.jsonl z magpie.answer")
    a.add_argument("--out", required=True, help="katalog wynikowy (raw.jsonl, accepted.jsonl)")
    a.add_argument("--url", default="http://192.168.1.26:2300/v1")
    a.add_argument("--model", default="/media/nvme_2tb/maked/models/glimmer")
    a.add_argument("--template", choices=sorted(TEMPLATES), default="glimmer")
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--top-p", type=float, default=0.95)
    a.add_argument("--max-tokens", type=int, default=1024)
    a.add_argument("--max-prompt-chars", type=int, default=60000,
                   help="pomijaj rozmowy z dłuższą historią (0 = bez limitu)")
    a.add_argument("--repeat-ratio", type=float, default=0.75,
                   help="podobieństwo (difflib) do wcześniejszego pytania, od którego odrzucamy")
    a.add_argument("--concurrency", type=int, default=1500)
    a.add_argument("--rate", type=float, default=200)
    a.add_argument("--retries", type=int, default=2)
    a.add_argument("--ttft-timeout", type=float, default=390)
    a.add_argument("--chunk-timeout", type=float, default=156)
    a.add_argument("--total-timeout", type=float, default=780)
    a.add_argument("--flush-every", type=int, default=100)
    a.add_argument("--progress-every", type=int, default=500)
    a.add_argument("--retry-rejected", type=int, default=0,
                   help="ile dodatkowych prób (nowy seed) dla rozmów z odrzuconym pytaniem")
    a.add_argument("--retry-only", action="store_true",
                   help="tylko rozmowy już obecne w raw.jsonl (np. po przerwanym runie) — nowych nie zaczynaj")
    a.add_argument("--target-turns", type=int, default=0,
                   help="nadpisz docelową liczbę tur z wierszy wejścia (0 = z wierszy)")
    a.add_argument("--finalize-only", action="store_true")

    m = sub.add_parser("pipeline", help="pytania i odpowiedzi jednym modelem (v13), cała rozmowa naraz")
    m.add_argument("--input", required=True, help="wynik select (system, user, assistant + target_turns)")
    m.add_argument("--out", required=True, help="katalog wynikowy (raw.jsonl, sft.jsonl)")
    m.add_argument("--url", default="http://192.168.1.26:2300/v1")
    m.add_argument("--model", required=True, help="nazwa modelu na serwerze (Bielik po SFT, szablon v13)")
    m.add_argument("--context", type=int, default=0, help="kontekst serwera w tokenach (wymagany)")
    m.add_argument("--target-turns", type=int, default=0, help="nadpisz target_turns z wejścia (0 = z wierszy)")
    m.add_argument("--attempts", type=int, default=3, help="prób na jedno pytanie (nowy seed po odrzucie)")
    m.add_argument("--answer-attempts", type=int, default=2, help="prób na jedną odpowiedź")
    m.add_argument("--repeat-ratio", type=float, default=0.75)
    m.add_argument("--min-answer-tokens", type=int, default=512,
                   help="nie zaczynaj tury, gdy po pytaniu zostałoby mniej kontekstu na odpowiedź")
    m.add_argument("--system-in-sft", action="store_true", help="zostaw prompt glimmer8 w sft.jsonl")
    # pytanie
    m.add_argument("--temperature", type=float, default=1.0)
    m.add_argument("--top-p", type=float, default=0.95)
    m.add_argument("--max-tokens", type=int, default=1024)
    m.add_argument("--ttft-timeout", type=float, default=600)
    m.add_argument("--chunk-timeout", type=float, default=300)
    m.add_argument("--total-timeout", type=float, default=900)
    # odpowiedź (jak magpie.pipeline)
    m.add_argument("--answer-temperature", type=float, default=0.7)
    m.add_argument("--answer-top-p", type=float, default=0.95)
    m.add_argument("--answer-top-k", type=int, default=None)
    m.add_argument("--answer-min-p", type=float, default=None)
    m.add_argument("--answer-presence-penalty", type=float, default=None)
    m.add_argument("--answer-frequency-penalty", type=float, default=None)
    m.add_argument("--answer-repetition-penalty", type=float, default=None)
    m.add_argument("--answer-max-tokens", type=int, default=0, help="0 = reszta kontekstu po prompcie")
    m.add_argument("--answer-ttft-timeout", type=float, default=3600)
    m.add_argument("--answer-chunk-timeout", type=float, default=600)
    m.add_argument("--answer-total-timeout", type=float, default=7200)
    # wspólne
    m.add_argument("--no-stream", action="store_true")
    m.add_argument("--concurrency", type=int, default=1500)
    m.add_argument("--rate", type=float, default=200, help="zapytań/s (każda tura to 2+ zapytania)")
    m.add_argument("--retries", type=int, default=2, help="ponowienia po błędzie HTTP/timeoucie")
    m.add_argument("--flush-every", type=int, default=100)
    m.add_argument("--progress-every", type=int, default=500)
    m.add_argument("--finalize-only", action="store_true")

    args = p.parse_args()
    {"select": cmd_select, "ask": cmd_ask, "pipeline": cmd_pipeline}[args.cmd](args)


if __name__ == "__main__":
    main()
