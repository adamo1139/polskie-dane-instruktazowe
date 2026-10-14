#!/usr/bin/env bash
set -euo pipefail

# Top-k logprobów nauczyciela dla całego cache SFT: jeden proces teacher_topk.py na GPU,
# potem --finalize (sprawdza komplet kawałków i pisze manifest.json). Przerwany przebieg
# wystarczy puścić jeszcze raz — gotowe kawałki (.done) są pomijane.
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
pids=()
for ((r = 0; r < GPUS; r++)); do
    CUDA_VISIBLE_DEVICES="${r}" python3 "${SCRIPT}" --cache "${CACHE}" --model "${TEACHER_HF}" \
        --out "${OUT}" --k "${K}" --rank "${r}" --world "${GPUS}" \
        --tokens-per-batch "${TOKENS_PER_BATCH}" --attn "${ATTN}" \
        > "${OUT}/logs/rank${r}.log" 2>&1 &
    pids+=($!)
done
echo "Uruchomiono ${GPUS} procesów; logi: ${OUT}/logs/rank*.log"

failed=0
for r in "${!pids[@]}"; do
    wait "${pids[$r]}" || { echo "rank ${r} zakończył się błędem — ${OUT}/logs/rank${r}.log" >&2; failed=1; }
done
[[ "${failed}" == 0 ]] || exit 1

python3 "${SCRIPT}" --cache "${CACHE}" --out "${OUT}" --k "${K}" --world "${GPUS}" --finalize
