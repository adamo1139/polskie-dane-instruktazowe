#!/usr/bin/env bash
set -euo pipefail

# Nauczyciel Lotnictwa 1 — pilotaż: pełny SFT Bielika-1.5B-v3.0-Instruct, Megatron, PP=8.
# Na wzór Ling-V2 run_poziomka_sft_run10.sh + run_poziomka.sh, z tym samym train_poziomka_sft.py
# (pakowanie thd, loss z cache), ale model gęsty: bez argumentów MoE (router, z-loss, permute).
#
# Uruchomienie na riggu:
#   LING_DIR=~/projects/pretrain/Ling-V2 bash run_bielik_1p5b_sft_pilot.sh
#
# Dane: bielik-sft-cache-nauka-lotnictwa-1-v13-8192 (build_nauka_lotnictwa_1_cache.sh):
# 532,592 rozmów train (141 za długich odrzuconych), loss na wszystkich tokenach poza BOS
# (także na prompcie systemowym), 60,426 spakowanych sekwencji po 8192 (wypełnienie 87,6%).
#
# GBS 64 przy micro-batch 1 i DP 1 = 64 mikrobatche na krok -> bańka pipeline'u
# (PP-1)/(m+PP-1) = 7/71 ~ 10% (run 10 Poziomki przy GBS 16: ~30%).
# ~64 x 7,180 prawdziwych tokenów = ~460k tokenów/krok.
# Jedna epoka: ceil(60,426 / 64) = 945 kroków; ostatni batch zawija się o 54 sekwencje.
#
# LR 5e-5 stały (min-lr = lr), warmup 20 kroków (momenty Adama startują od zera).

: "${LING_DIR:?Set LING_DIR to the Ling-V2 checkout}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${LING_DIR}/examples/sft/megatron"
WORK_DIR="${WORK_DIR:-$(cd -- "${LING_DIR}/.." && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-${LING_DIR}/Megatron-LM-core_v0.13.0}"
source "${SCRIPT_DIR}/bielik_1p5b_model_args.sh"

SFT_DATA="${WORK_DIR}/bielik-sft-cache-nauka-lotnictwa-1-v13-8192"
LOAD_CHECKPOINT="/media/nvme_2tb/maked/bielik_train/Bielik-1.5B-v3.0-Instruct-dcp"
SAVE_CHECKPOINT="/media/nvme_2tb/maked/bielik_train/bielik_1p5b_sft_pilot_nauka_lotnictwa_1_v13_8192_gbs64"
RESUME="${RESUME:-0}"

SEQ_LENGTH=8192
GLOBAL_BATCH_SIZE=64
TRAIN_ITERS=945
LR=5e-5
WARMUP_ITERS=20
# 1.5B w bf16 = ~3 GB na zapis (same wagi): 9 zapisów ~ 30 GB.
SAVE_INTERVAL=100
EVAL_INTERVAL=100
# Walidacja: 5,344 rozmów ~ 606 spakowanych sekwencji; 9 x 64 = 576.
EVAL_ITERS=9
DATALOADER_WORKERS=2

WANDB_ENTITY="${WANDB_ENTITY:-adamo1139-no}"
WANDB_PROJECT="${WANDB_PROJECT:-nauczyciel-lotnictwa}"
WANDB_NAME="${WANDB_NAME:-bielik_1p5b_sft_pilot_nauka_lotnictwa_1_v13_8192_packed_gbs64_lr5e-5_const_wu20_945steps}"
export WANDB_ENTITY WANDB_MODE="${WANDB_MODE:-online}"

[[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || { echo "Missing patched Megatron checkout" >&2; exit 1; }
[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "No cache at ${SFT_DATA}; run build_nauka_lotnictwa_1_cache.sh" >&2; exit 1; }
[[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || { echo "Missing DCP tracker in ${LOAD_CHECKPOINT}" >&2; exit 1; }

# The cache must match the training length, record count, train bins (TRAIN_ITERS) and
# validation bins (EVAL_ITERS x GBS must not exceed them).
PYTHONPATH="${SFT_DIR}" python3 - "${SFT_DATA}/manifest.json" "${SEQ_LENGTH}" \
    "${TRAIN_ITERS}" "${GLOBAL_BATCH_SIZE}" "${EVAL_ITERS}" <<'PY'
import json, math, sys
from poziomka_data import MMapSFTDataset
train_iters, gbs, eval_iters = map(int, sys.argv[3:6])
manifest = json.load(open(sys.argv[1]))
if manifest["seq_length"] != int(sys.argv[2]):
    sys.exit(f"Cache is {manifest['seq_length']} tokens, training at {sys.argv[2]}")
if manifest["totals"]["train"]["records"] != 532592:
    sys.exit(f"Expected 532,592 train records, found {manifest['totals']['train']['records']:,}; "
             "recompute TRAIN_ITERS")
train_bins = len(MMapSFTDataset(sys.argv[1], "train", pack=True).bins)
valid_bins = len(MMapSFTDataset(sys.argv[1], "validation", pack=True, shuffle=False).bins)
if math.ceil(train_bins / gbs) != train_iters:
    sys.exit(f"{train_bins:,} train bins -> {math.ceil(train_bins / gbs)} iters at GBS {gbs}, not {train_iters}")
if eval_iters * gbs > valid_bins:
    sys.exit(f"EVAL_ITERS {eval_iters} x GBS {gbs} > {valid_bins} validation bins; use {valid_bins // gbs}")
print(f"Bins OK: train {train_bins:,} ({train_iters} iters), validation {valid_bins} ({eval_iters} x {gbs} used)")
print(f"Cache OK: {manifest['totals']['train']['records']:,} train records, "
      f"{manifest['totals']['train']['tokens']:,} tokens, loss roles {manifest['loss_roles']}")
PY

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

# Refuse a run whose checkpoints cannot fit: ~3 GB per save, weights only.
save_parent="$(dirname "${SAVE_CHECKPOINT}")"
[[ -d "${save_parent}" ]] || { echo "No such directory: ${save_parent}" >&2; exit 1; }
free_gb=$(df -BG --output=avail "${save_parent}" | tail -1 | tr -dc '0-9')
needed_gb=$(( (TRAIN_ITERS / SAVE_INTERVAL + 2) * 3 ))
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
    --eval-iters "${EVAL_ITERS}" --log-interval 1 --no-one-logger \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" \
    "${LOAD_ARGS[@]}" "$@"
