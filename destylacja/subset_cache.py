"""Pod-cache SFT (poziomka-sft-v2): część shardów train + cała walidacja, bez kopiowania danych.

Shardy train z prepare_polskie_sprawy_v3.py dostają rozmowy po kolei (round-robin) z potasowanego
sft.jsonl, więc każdy shard to losowa 1/N zbioru, a k pierwszych shardów to losowe k/N. Nowy katalog
ma symlinki do plików tych shardów, tokenizera i szablonu oraz własny manifest.json (te same pola,
tylko shards i totals przeliczone), więc teacher_topk.py i trening (MMapSFTDataset / MMapKDDataset)
widzą go jak zwykły cache.

  python destylacja/subset_cache.py --cache <cache 16384> --out <nowy katalog> --train-shards 4
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="katalog cache z manifest.json")
    p.add_argument("--out", required=True, help="nowy katalog (nie może istnieć)")
    p.add_argument("--train-shards", type=int, required=True, help="ile pierwszych shardów train wziąć")
    args = p.parse_args()

    cache, out = Path(args.cache).resolve(), Path(args.out).resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    train = sorted((s for s in manifest["shards"] if s["split"] == "train"), key=lambda s: s["prefix"])
    if not 0 < args.train_shards <= len(train):
        raise SystemExit(f"--train-shards musi być w [1, {len(train)}]")
    keep = train[:args.train_shards] + [s for s in manifest["shards"] if s["split"] != "train"]
    out.mkdir(parents=True, exist_ok=False)

    (out / "tokenizer").symlink_to(cache / "tokenizer", target_is_directory=True)
    (out / "chat_template.jinja").symlink_to(cache / "chat_template.jinja")
    for shard in keep:
        prefix = Path(shard["prefix"])
        (out / prefix.parent).mkdir(parents=True, exist_ok=True)
        files = sorted((cache / prefix.parent).glob(f"{prefix.name}.*"))
        if not {f.name.split(".", 1)[1] for f in files} >= {"tokens.bin", "masks.bin", "offsets.bin"}:
            raise SystemExit(f"Brak plików shardu {prefix} w {cache}")
        for f in files:
            (out / prefix.parent / f.name).symlink_to(f)

    totals = {}
    for split in ("train", "validation"):
        stats = Counter()
        for shard in keep:
            if shard["split"] == split:
                stats.update(shard["stats"])
        stats["discarded_supervised_tokens"] = stats["input_supervised_tokens"] - stats["supervised_tokens"]
        totals[split] = dict(stats)
    sub = dict(manifest, shards=keep, totals=totals,
               subset_of=dict(cache=str(cache), manifest_sha256=sha256(cache / "manifest.json"),
                              train_shards=[s["prefix"] for s in train[:args.train_shards]],
                              train_shards_total=len(train)))
    (out / "manifest.json").write_text(json.dumps(sub, ensure_ascii=False, indent=2) + "\n")
    full = manifest["totals"]["train"]
    print(f"train: {totals['train']['records']:,} rozmów / {full['records']:,} "
          f"({100 * totals['train']['records'] / full['records']:.1f}%), {totals['train']['tokens']:,} tokenów; "
          f"walidacja: {totals['validation']['records']:,} rozmów (cała)  →  {out}")


if __name__ == "__main__":
    main()
