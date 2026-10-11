#!/usr/bin/env bash
# Bielik-PL-Minitron-7B-v3.0-Instruct (LlamaForCausalLM) for Ling-patched Megatron core_v0.13.0.
# Shared by SFT and HF -> DCP conversion (MODEL_ARGS_FILE in convert_bielik_hf_to_dcp.sh).
#
# Unlike Bielik 1.5B: NO bias in any linear layer (attention_bias = mlp_bias = false), so
# --disable-bias-linear; RMSNorm eps 1e-5; HF config declares max_position_embeddings 32768.
# 40 layers / PP 8 = 5 per stage. RoPE theta 1e6 over the full head_dim (128), no scaling.
BIELIK_MODEL_ARGS=(
    --num-layers 40 --hidden-size 4096 --ffn-hidden-size 11264
    --num-attention-heads 32 --num-query-groups 8 --group-query-attention --kv-channels 128
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-32768}"
    --vocab-size 32000 --make-vocab-size-divisible-by 128
    --position-embedding-type rope --rotary-base "${ROTARY_BASE:-1000000}" --rotary-percent 1.0
    --swiglu --untie-embeddings-and-output-weights --normalization RMSNorm
    --norm-epsilon 1e-5 --disable-bias-linear --transformer-impl transformer_engine
    --attention-dropout 0 --hidden-dropout 0
    --pipeline-model-parallel-size 8 --tensor-model-parallel-size 1
)
