#!/usr/bin/env bash
# Bielik-1.5B-v3.0-Instruct (LlamaForCausalLM) for Ling-patched Megatron core_v0.13.0.
# Shared by SFT and HF -> DCP conversion, like poziomka_model_args.sh.
#
# Dense Llama with Qwen2.5-1.5B widths but 32 layers, and bias in EVERY linear layer
# (attention_bias and mlp_bias: q/k/v/o and gate/up/down). Megatron's default
# add_bias_linear=True gives exactly qkv, proj, fc1, fc2 biases, so this file must
# NOT pass --disable-bias-linear.
#
# RoPE: theta 1e6 over the full head_dim (128), no scaling. HF config declares
# max_position_embeddings 8192.
BIELIK_MODEL_ARGS=(
    --num-layers 32 --hidden-size 1536 --ffn-hidden-size 8960
    --num-attention-heads 12 --num-query-groups 2 --group-query-attention --kv-channels 128
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-8192}"
    --vocab-size 32000 --make-vocab-size-divisible-by 128
    --position-embedding-type rope --rotary-base "${ROTARY_BASE:-1000000}" --rotary-percent 1.0
    --swiglu --untie-embeddings-and-output-weights --normalization RMSNorm
    --norm-epsilon 1e-6 --transformer-impl transformer_engine
    --attention-dropout 0 --hidden-dropout 0
    --pipeline-model-parallel-size 8 --tensor-model-parallel-size 1
)
