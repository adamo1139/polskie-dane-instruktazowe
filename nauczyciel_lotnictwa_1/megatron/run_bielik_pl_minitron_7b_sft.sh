#!/usr/bin/env bash
set -euo pipefail

# Nauczyciel Lotnictwa 1 — pełny SFT Bielika-PL-Minitron-7B-v3.0-Instruct, Megatron, PP=8.
# Ten sam schemat co run_bielik_1p5b_sft_pilot.sh (train_poziomka_sft.py, pakowanie thd, loss na
# wszystkich tokenach poza BOS), model z bielik_pl_minitron_7b_model_args.sh (bez biasów, 40 warstw
# = 5 na etap). TRAIN_ITERS (1 epoka) i EVAL_ITERS liczone z cache, jeśli nie podane.
#
# Uruchomienie na riggu (przykład):
#   LING_DIR=~/projects/pretrain/Ling-V2 \
#   SFT_DATA=~/projects/pretrain/bielik-sft-cache-<NAME>-v13-16384 SEQ_LENGTH=16384 \
#   bash run_bielik_pl_minitron_7b_sft.sh
#
# Pamięć (szacunek, do sprawdzenia pierwszym krótkim runem): ~0,93 mld parametrów na etap; wagi bf16
# + gradienty fp32 + Adam fp32 (DP=1, więc rozproszony optymalizator nic nie dzieli) ~ 17 GB/kartę,
# plus aktywacje (recompute full) i na etapie 0 / 7 embedding / lm_head + logity. Na 24 GB ciasno.

: "${LING_DIR:?Set LING_DIR to the Ling-V2 checkout}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${LING_DIR}/examples/sft/megatron"
WORK_DIR="${WORK_DIR:-$(cd -- "${LING_DIR}/.." && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-${LING_DIR}/Megatron-LM-core_v0.13.0}"
source "${SCRIPT_DIR}/bielik_pl_minitron_7b_model_args.sh"

: "${SFT_DATA:?Set SFT_DATA to the cache built by build_nauka_lotnictwa_1_cache.sh}"
: "${SEQ_LENGTH:?Set SEQ_LENGTH to the cache length}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/media/nvme_2tb/maked/bielik_train/Bielik-PL-Minitron-7B-v3.0-Instruct-dcp}"
RUN_NAME="${RUN_NAME:-bielik_pl_minitron_7b_sft_$(basename "${SFT_DATA}")}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-/media/nvme_2tb/maked/bielik_train/${RUN_NAME}}"
RESUME="${RESUME:-0}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
LR="${LR:-3e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
# 7.5B w bf16 = ~15 GB na zapis (same wagi).
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
DATALOADER_WORKERS=2
# TRAIN_ITERS / EVAL_ITERS: puste = z cache (1 epoka; walidacja do 9 x GBS sekwencji).
TRAIN_ITERS="${TRAIN_ITERS:-}"
EVAL_ITERS="${EVAL_ITERS:-}"

WANDB_ENTITY="${WANDB_ENTITY:-adamo1139-no}"
WANDB_PROJECT="${WANDB_PROJECT:-nauczyciel-lotnictwa}"
export WANDB_ENTITY WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || { echo "Missing patched Megatron checkout" >&2; exit 1; }
[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "No cache at ${SFT_DATA}; run build_nauka_lotnictwa_1_cache.sh" >&2; exit 1; }
[[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || { echo "Missing DCP tracker in ${LOAD_CHECKPOINT}" >&2; exit 1; }

# Cache must match the training length; TRAIN_ITERS / EVAL_ITERS derived from its packed bins.
read -r TRAIN_ITERS EVAL_ITERS < <(PYTHONPATH="${SFT_DIR}" python3 - "${SFT_DATA}/manifest.json" \
    "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}" "${TRAIN_ITERS:-0}" "${EVAL_ITERS:-0}" <<'PY'
import json, math, sys
from poziomka_data import MMapSFTDataset
gbs, train_iters, eval_iters = map(int, sys.argv[3:6])
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
train_bins = len(MMapSFTDataset(sys.argv[1], "train", pack=True).bins)
valid_bins = len(MMapSFTDataset(sys.argv[1], "validation", pack=True, shuffle=False).bins)
train_iters = train_iters or math.ceil(train_bins / gbs)
eval_iters = eval_iters or min(9, valid_bins // gbs)
if eval_iters < 1 or eval_iters * gbs > valid_bins:
    sys.exit(f"EVAL_ITERS {eval_iters} x GBS {gbs} vs {valid_bins} validation bins")
print(f"Cache OK: {manifest['totals']['train']['records']:,} train records, "
      f"{manifest['totals']['train']['tokens']:,} tokens, loss roles {manifest['loss_roles']}; "
      f"bins train {train_bins:,} -> TRAIN_ITERS {train_iters} (1 epoch = {math.ceil(train_bins / gbs)}), "
      f"validation {valid_bins} -> EVAL_ITERS {eval_iters}", file=sys.stderr)
print(train_iters, eval_iters)
PY
)
# Process substitution hides the checker's exit code: an empty result means it refused.
[[ -n "${TRAIN_ITERS}" && -n "${EVAL_ITERS}" ]] || { echo "Cache check failed (see above)" >&2; exit 1; }
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}_gbs${GLOBAL_BATCH_SIZE}_lr${LR}_const_wu${WARMUP_ITERS}_${TRAIN_ITERS}steps}"

if [[ "${RESUME}" != 1 && -e "${SAVE_CHECKPOINT}" ]]; then
    echo "Refusing existing output; set RESUME=1 to continue that run" >&2; exit 1
fi
LOAD_ARGS=(--finetune --no-load-optim --no-load-rng --override-opt_param-scheduler)
LOAD_FROM="${LOAD_CHECKPOINT}"
if [[ "${RESUME}" == 1 ]]; then
    [[ -f "${SAVE_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || { echo "No SFT checkpoint to resume" >&2; exit 1; }
    LOAD_ARGS=(--no-load-optim --no-load-rng --override-opt_param-scheduler)
    LOAD_FROM="${SAVE_CHECKPOINT}"
fi

# Refuse a run whose checkpoints cannot fit: ~15 GB per save, weights only.
save_parent="$(dirname "${SAVE_CHECKPOINT}")"
[[ -d "${save_parent}" ]] || { echo "No such directory: ${save_parent}" >&2; exit 1; }
free_gb=$(df -BG --output=avail "${save_parent}" | tail -1 | tr -dc '0-9')
needed_gb=$(( (TRAIN_ITERS / SAVE_INTERVAL + 2) * 15 ))
if (( free_gb < needed_gb )); then
    echo "Checkpoints need ~${needed_gb} GB, ${free_gb} GB free on ${save_parent}." >&2
    exit 1
fi

# Same NCCL/TE settings as run_poziomka.sh (patched P2P driver; nothing forced off).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export PYTHONPATH="${SFT_DIR}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# --log-throughput (jak w pretreningu Poziomki): TFLOP/s/GPU w logu i w wandb. Wzór Megatrona liczy
# pełne sekwencje 8192 i uwagę na całym oknie, a przy pakowaniu (87,6% prawdziwych tokenów, uwaga thd
# tylko w obrębie rozmowy) zawyża wynik; MFU ~ TFLOP/s / 80 (3090 Ti, bf16 z akumulacją fp32, dense).
# Saves are synchronous and weight-only, as in Poziomka run 10 (async save zeroed weights
# in runs 5/6). Recompute full as in Poziomka: 1F1B keeps up to 8 microbatches of
# activations on stage 0.
exec torchrun --standalone --nproc_per_node=8 "${SFT_DIR}/train_poziomka_sft.py" \
    "${BIELIK_MODEL_ARGS[@]}" \
    --sft-manifest "${SFT_DATA}/manifest.json" --sft-packing \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${SFT_DATA}/tokenizer" \
    --seq-length "${SEQ_LENGTH}" --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" --bf16 --optimizer adam --use-distributed-optimizer \
    --calculate-per-token-loss \
    --lr "${LR}" --min-lr "${LR}" --lr-decay-style constant \
    --lr-warmup-iters "${WARMUP_ITERS}" --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --seed 42 \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --dataloader-type single --num-workers "${DATALOADER_WORKERS}" \
    --no-create-attention-mask-in-dataloader --attention-backend flash \
    --attention-softmax-in-fp32 --no-masked-softmax-fusion \
    --load "${LOAD_FROM}" --save "${SAVE_CHECKPOINT}" --ckpt-format torch_dist \
    --no-save-optim --no-save-rng \
    --save-interval "${SAVE_INTERVAL}" --eval-interval "${EVAL_INTERVAL}" \
    --eval-iters "${EVAL_ITERS}" --log-interval 1 --log-throughput --no-one-logger \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" \
    "${LOAD_ARGS[@]}" "$@"
