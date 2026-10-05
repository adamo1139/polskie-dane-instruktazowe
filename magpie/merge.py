"""Łączy kilka verified.jsonl (magpie.verify) w jeden zbiór pytań, deduplikowany po SHA256.

Bierzemy tylko verdict == "ok". id każdego pytania = SHA256 pełnego tekstu (raw) — unikalny
i stabilny między połączeniami (id z poszczególnych runów to numery zadań i powtarzają się).
Przy duplikacie zostaje pierwsze wystąpienie w kolejności --inputs; source / source_id mówią,
skąd pochodzi. Pola potrzebne magpie.answer (raw, topic, persona, subtopics, verdict) zostają.
"""
import argparse
import hashlib
import json
import os
from collections import Counter


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="pliki verified.jsonl (kolejność = priorytet)")
    p.add_argument("--out", required=True, help="plik wynikowy .jsonl")
    args = p.parse_args()

    seen: set[str] = set()
    stats: Counter = Counter()
    merged = []
    for path in args.inputs:
        source = os.path.basename(os.path.dirname(os.path.abspath(path)))
        with open(path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                if r.get("verdict") != "ok":
                    continue
                stats["ok"] += 1
                h = text_id(r["raw"])
                if h in seen:
                    stats["duplikat"] += 1
                    continue
                seen.add(h)
                merged.append({**r, "id": h, "source": source, "source_id": r["id"]})
                stats[f"z {source}"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"przyjętych na wejściu: {stats['ok']}  duplikatów (SHA256): {stats['duplikat']}  "
          f"→  w zbiorze: {len(merged)}  ({args.out})")
    print("pochodzenie:", {k: v for k, v in stats.items() if k.startswith("z ")})
    print("per temat:", dict(Counter(r["topic"] for r in merged).most_common()))
    print("persona:", dict(Counter(r["persona"] for r in merged)))


if __name__ == "__main__":
    main()
