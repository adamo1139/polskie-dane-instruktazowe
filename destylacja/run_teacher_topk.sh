#!/usr/bin/env bash
set -euo pipefail

# Top-k logprobów nauczyciela dla całego cache SFT: jeden proces teacher_topk.py na GPU
# (logi w ${OUT}/logs, w terminalu jedna linia z sumą postępu i ETA), potem --finalize
# (sprawdza komplet kawałków i pisze manifest.json). Przerwany przebieg wystarczy puścić
# jeszcze raz — gotowe kawałki (.done) są pomijane.
#
#   CACHE=~/projects/pretrain/bielik-sft-cache-destylowanie-1-v13-8192 \
#   TEACHER_HF=<eksport HF nauczyciela, bf16> \
#   OUT=~/projects/pretrain/teacher-topk-destylowanie-1-k40 \
#   bash run_teacher_topk.sh
#
# Najpierw benchmark na jednym GPU (bez zapisu), żeby dobrać TOKENS_PER_BATCH:
#   CUDA_VISIBLE_DEVICES=0 python teacher_topk.py --cache $CACHE --model $TEACHER_HF --out $OUT --benchmark 2000

: "${CACHE:?Set CACHE to the SFT cache directory (manifest.json)}"
: "${TEACHER_HF:?Set TEACHER_HF to the HF teacher directory}"
: "${OUT:?Set OUT to the output directory}"
K="${K:-40}"
GPUS="${GPUS:-8}"
TOKENS_PER_BATCH="${TOKENS_PER_BATCH:-16384}"
ATTN="${ATTN:-flash_attention_2}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/teacher_topk.py"

mkdir -p "${OUT}/logs"
rm -f "${OUT}"/logs/rank*.progress
pids=()
for ((r = 0; r < GPUS; r++)); do
    CUDA_VISIBLE_DEVICES="${r}" python3 "${SCRIPT}" --cache "${CACHE}" --model "${TEACHER_HF}" \
        --out "${OUT}" --k "${K}" --rank "${r}" --world "${GPUS}" \
        --tokens-per-batch "${TOKENS_PER_BATCH}" --attn "${ATTN}" \
        > "${OUT}/logs/rank${r}.log" 2>&1 &
    pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null; echo; exit 130' INT TERM
echo "Uruchomiono ${GPUS} procesów; logi: ${OUT}/logs/rank*.log"

# Suma postępu wszystkich ranków z plików rank*.progress, w jednej odświeżanej linii.
PROGRESS_PY='
import json, sys
from pathlib import Path
logs, world = Path(sys.argv[1]), int(sys.argv[2])
parts = []
for r in range(world):
    try:
        parts.append(json.loads((logs / f"rank{r}.progress").read_text()))
    except (OSError, ValueError):
        pass
# Rank liczy, gdy ma już nowe pozycje albo miał wszystko z poprzedniego przebiegu.
ready = [p for p in parts if p["new"] or p["done"] >= p["total"]]
rate = sum(p["new"] / p["elapsed"] for p in parts if p["new"])
if len(parts) < world or not rate:
    print(f"\rładowanie modelu i pierwsze paczki: {len(ready)}/{world} ranków liczy   ", end="", flush=True)
    sys.exit()
done, total = sum(p["done"] for p in parts), sum(p["total"] for p in parts)
# ETA przy obecnym tempie; dopóki nie liczą wszystkie ranki, jest zaniżone.
eta = (total - done) / rate
print(f"\r{len(ready)}/{world} ranków, {done:,}/{total:,} pozycji ({100 * done / total:.1f}%), "
      f"{rate:,.0f} tok/s, ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m   ", end="", flush=True)
'
# Rank, który padnie, zatrzymuje cały przebieg od razu (gotowe kawałki zostają na dysku).
declare -A finished=()
check_ranks() {
    for r in "${!pids[@]}"; do
        [[ -z "${finished[$r]:-}" ]] || continue
        kill -0 "${pids[$r]}" 2>/dev/null && continue
        if wait "${pids[$r]}"; then
            finished[$r]=1
        else
            echo; echo "rank ${r} zakończył się błędem — ${OUT}/logs/rank${r}.log:" >&2
            tail -n 15 "${OUT}/logs/rank${r}.log" >&2
            kill "${pids[@]}" 2>/dev/null; wait; exit 1
        fi
    done
}
while (( ${#finished[@]} < GPUS )); do
    python3 -c "${PROGRESS_PY}" "${OUT}/logs" "${GPUS}" || true
    check_ranks
    (( ${#finished[@]} < GPUS )) && sleep 5
done
python3 -c "${PROGRESS_PY}" "${OUT}/logs" "${GPUS}" || true
echo

python3 "${SCRIPT}" --cache "${CACHE}" --out "${OUT}" --k "${K}" --world "${GPUS}" --finalize
