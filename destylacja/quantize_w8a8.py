"""Kwantyzacja nauczyciela do W8A8 INT8 (llm-compressor), kalibrowana na rekordach cache SFT.

W8A8 = wagi int8 per kanał + aktywacje int8 dynamicznie per token; SGLang/vLLM liczą to
kernelami INT8 (na 3090 Ti ~1,7× szybciej niż bf16 przy samym prefillu, NLL +~1%). Przepis:
SmoothQuant (przenosi skalę aktywacji do wag, smoothing 0,8), potem GPTQ na wszystkich Linear
poza lm_head. Kalibracja na tokenach z cache (v13, ten sam tekst, który nauczyciel potem
ocenia) — bez ponownej tokenizacji i bez szablonu czatu modelu.

  python3 destylacja/quantize_w8a8.py --model <bf16 HF> --cache <cache SFT> --out <katalog W8A8>

Wymaga llmcompressor (pip install llmcompressor) i jednego GPU z ~24 GB dla 7B.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def calibration_records(cache: Path, n: int, max_len: int, seed: int) -> list[list[int]]:
    """n losowych rekordów train z cache (ID tokenów), ucięte do max_len."""
    manifest = json.loads((cache / "manifest.json").read_text())
    shards = [s for s in manifest["shards"] if s["split"] == "train" and s["records"]]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(sum(s["records"] for s in shards), size=n, replace=False):
        for shard in shards:
            if i < shard["records"]:
                break
            i -= shard["records"]
        prefix = cache / shard["prefix"]
        offsets = np.memmap(f"{prefix}.offsets.bin", dtype="<u8", mode="r")
        tokens = np.memmap(f"{prefix}.tokens.bin", dtype="<u2", mode="r")
        s, e = int(offsets[i]), int(offsets[i + 1])
        out.append(tokens[s:min(e, s + max_len)].astype(np.int64).tolist())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="katalog HF modelu bf16")
    p.add_argument("--cache", required=True, help="cache SFT (poziomka-sft-v2) do kalibracji")
    p.add_argument("--out", required=True, help="katalog wynikowy (nie może istnieć)")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--smoothing", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20261019)
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} istnieje — nie nadpisuję")
    records = calibration_records(Path(args.cache).resolve(), args.samples, args.max_len, args.seed)
    print(f"kalibracja: {len(records)} rekordów z {args.cache}, śr. {np.mean([len(r) for r in records]):.0f} "
          f"tokenów (max {args.max_len})", flush=True)
    ds = Dataset.from_dict({"input_ids": records, "attention_mask": [[1] * len(r) for r in records]})

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    recipe = [SmoothQuantModifier(smoothing_strength=args.smoothing),
              GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_len,
            num_calibration_samples=len(records))
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    print(f"Gotowe: {out}", flush=True)


if __name__ == "__main__":
    main()
