"""Dopisuje do rozmów z polskie-sprawy-v4 oryginalny prompt systemowy Magpie (glimmer8).

Każdy run magpie.generate zapisał w accepted.jsonl pole `system` — prompt, z którego Glimmer
wygenerował dane pytanie. id rozmowy w v4 to SHA256 treści pytania (magpie.merge_sft), więc
dopasowujemy po SHA256 pola `raw`. To samo pytanie bywa w kilku runach z różnymi promptami
(wszystkie są prawdziwe); bierzemy pierwszy w kolejności posortowanych ścieżek accepted.jsonl,
żeby wynik był deterministyczny. Brak dopasowania przerywa skrypt.

--sft: dodatkowe sft.jsonl, w których rozmowy już zaczynają się od promptu systemowego (np.
destylowanie-1, nauka-lotnictwa-2 — tam są prompty pytań instruktora); mają pierwszeństwo przed
--magpie. --thinking / --single-turn: bierz z wejścia tylko pasujące rozmowy (np. pula do
magpie.multiturn z polskie-sprawy-v6), resztę pomijaj zamiast przerywać.

Wynik: te same rozmowy z wiadomością {"role": "system"} na początku; reszta bez zmian.
"""
import argparse
import glob
import hashlib
import json
import os
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/hf_polskie_sprawy_v4/sft.jsonl")
    p.add_argument("--magpie", default="data/magpie_glimmer8*/accepted.jsonl", help="glob z wynikami magpie.generate")
    p.add_argument("--out", default="data/polskie_sprawy_v4_sys/sft.jsonl")
    p.add_argument("--sft", nargs="*", default=[], help="sft.jsonl z rozmowami system, user, … (pierwszeństwo)")
    p.add_argument("--thinking", choices=["off", "on"], help="tylko rozmowy z meta.thinking w tym trybie")
    p.add_argument("--single-turn", action="store_true", help="tylko rozmowy user, assistant")
    args = p.parse_args()

    systems: dict[str, str] = {}
    variants: Counter = Counter()
    for path in args.sft:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)["messages"]
                if m[0]["role"] != "system" or m[1]["role"] != "user":
                    continue
                h = hashlib.sha256(m[1]["content"].encode("utf-8")).hexdigest()
                if h not in systems:
                    systems[h] = m[0]["content"]
                elif systems[h] != m[0]["content"]:
                    variants[h] += 1
    for path in sorted(glob.glob(args.magpie)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                h = hashlib.sha256(r["raw"].encode("utf-8")).hexdigest()
                if h not in systems:
                    systems[h] = r["system"]
                elif systems[h] != r["system"]:
                    variants[h] += 1
    print(f"pytań z promptem: {len(systems)} (z plików {args.sft + [args.magpie]})")

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} istnieje — nie nadpisuję")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = ambiguous = skipped = 0
    with open(args.input, encoding="utf-8") as fin, open(args.out, "x", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if (args.thinking and r["meta"].get("thinking") != args.thinking) or \
                    (args.single_turn and [m["role"] for m in r["messages"]] != ["user", "assistant"]):
                skipped += 1
                continue
            if r["messages"][0]["role"] == "system":
                raise SystemExit(f"{r['id']}: rozmowa ma już prompt systemowy")
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            if hashlib.sha256(user.encode("utf-8")).hexdigest() != r["id"]:
                raise SystemExit(f"{r['id']}: id ≠ SHA256 pytania")
            if r["id"] not in systems:
                raise SystemExit(f"{r['id']}: brak promptu systemowego w {args.magpie}")
            ambiguous += r["id"] in variants
            r["messages"] = [{"role": "system", "content": systems[r["id"]]}] + r["messages"]
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"rozmów: {n}, w tym z kilkoma możliwymi promptami (wzięty pierwszy): {ambiguous}, "
          f"pominiętych filtrem: {skipped}  →  {args.out}")


if __name__ == "__main__":
    main()
