"""Łączy pliki sft.jsonl (magpie.answer) w jeden, deduplikowany po SHA256 pytania i potasowany.

id każdego wiersza = SHA256 treści wiadomości użytkownika (id z poszczególnych runów to numery
zadań albo hashe i mogą się różnić typem). Przy duplikacie zostaje pierwsze wystąpienie w kolejności
--inputs; source / source_id mówią, skąd pochodzi. Potem tasowanie z --seed.
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="pliki sft.jsonl (kolejność = priorytet)")
    p.add_argument("--out", required=True, help="plik wynikowy .jsonl")
    p.add_argument("--seed", type=int, default=29382948, help="seed tasowania")
    args = p.parse_args()

    seen: set[str] = set()
    stats: Counter = Counter()
    merged = []
    for path in args.inputs:
        source = os.path.basename(os.path.dirname(os.path.abspath(path)))
        with open(path, encoding="utf-8") as f:
            for r in map(json.loads, f):
                stats["wejście"] += 1
                user = next(m["content"] for m in r["messages"] if m["role"] == "user")
                h = hashlib.sha256(user.encode("utf-8")).hexdigest()
                if h in seen:
                    stats["duplikat"] += 1
                    continue
                seen.add(h)
                merged.append({**r, "id": h, "source": source, "source_id": r["id"]})
                stats[source] += 1

    random.Random(args.seed).shuffle(merged)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"na wejściu: {stats['wejście']}  duplikatów (SHA256): {stats['duplikat']}  →  w zbiorze: "
          f"{len(merged)}  (potasowane, seed {args.seed})  {args.out}")
    print("pochodzenie:", {k: v for k, v in stats.items() if k not in ("wejście", "duplikat")})


if __name__ == "__main__":
    main()
