"""
Load an HF Llama checkpoint (Bielik v3, bias in every linear layer) into a Megatron
model and save it as DCP. Based on Ling-V2 tools/load_hf_save_dcp.py (BailingMoeV2).

Run with torchrun using the same parallelism as training:
    PYTHONPATH=Megatron-LM-core_v0.13.0:$PYTHONPATH \
    torchrun --nproc_per_node 8 load_hf_llama_save_dcp.py \
        --hf-path /path/to/hf/checkpoint \
        <all the same Megatron args as training>

Differences from the bailing importer:
- HF Llama keeps q/k/v separate (q_proj/k_proj/v_proj); they are concatenated and
  interleaved per query group for mcore, and their biases get the same reordering.
- Dense MLP only; gate/up weights AND biases are concatenated into linear_fc1.
- Every model parameter of this stage must be written exactly once, and every HF
  tensor of this stage must be used; anything left over aborts instead of saving a
  partially initialised checkpoint.
"""

import json
import os
import sys
import torch
from safetensors import safe_open


def validate_hf_config(hf_path, args):
    """Reject shape-compatible but semantically wrong imports."""
    with open(os.path.join(hf_path, 'config.json')) as stream:
        config = json.load(stream)
    comparisons = {
        'model_type': 'llama',
        'hidden_act': 'silu',
        'num_hidden_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'intermediate_size': args.ffn_hidden_size,
        'num_attention_heads': args.num_attention_heads,
        'num_key_value_heads': args.num_query_groups,
        'head_dim': args.kv_channels,
        'vocab_size': args.padded_vocab_size,
        'rope_theta': args.rotary_base,
        'rms_norm_eps': args.norm_epsilon,
        'tie_word_embeddings': not args.untie_embeddings_and_output_weights,
        # Megatron's add_bias_linear covers qkv, proj, fc1 and fc2 together.
        'attention_bias': args.add_bias_linear,
        'mlp_bias': args.add_bias_linear,
    }
    for key, expected in comparisons.items():
        value = config.get(key)
        if key == 'rope_theta' and value is not None:
            value = float(value)
            expected = float(expected)
        if value != expected:
            raise ValueError(f'HF/Megatron mismatch: {key}={config.get(key)!r}, expected {expected!r}')
    if args.rotary_percent != 1.0:
        raise ValueError('Llama rotates the full head_dim: --rotary-percent must be 1.0')
    if config.get('rope_scaling') is not None or args.use_rope_scaling:
        raise ValueError('This converter is configured for unscaled RoPE')
    if args.qk_layernorm or args.num_experts:
        raise ValueError('Bielik has no qk-norm and no MoE')
    if args.tensor_model_parallel_size != 1:
        raise ValueError('HF importer currently supports TP=1 only')
    if args.num_layers % args.pipeline_model_parallel_size:
        raise ValueError('Importer requires equally sized pipeline stages')


def load_hf_state_dict(hf_path):
    """Load all tensors from HF safetensors checkpoint."""
    index_path = os.path.join(hf_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index['weight_map'].values()))
    else:
        shard_files = ['model.safetensors']

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(hf_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def interleave_qkv(q, k, v, num_query_groups):
    """
    HF Llama: separate Q [heads*d], K [groups*d], V [groups*d] (weights or biases).
    Megatron (mcore): [Q0, K0, V0, Q1, K1, V1, ...] interleaved per query group.
    """
    groups = []
    for qi, ki, vi in zip(torch.chunk(q, num_query_groups, dim=0),
                          torch.chunk(k, num_query_groups, dim=0),
                          torch.chunk(v, num_query_groups, dim=0)):
        groups.append(torch.cat([qi, ki, vi], dim=0))
    return torch.cat(groups, dim=0)


class Copier:
    """Copy HF tensors into parameters, tracking what was used and what was written."""

    def __init__(self, model, hf_state_dict):
        self.hf = hf_state_dict
        self.used = set()
        self.written = set()
        self.names = {id(p): n for n, p in model.named_parameters()}

    def get(self, key):
        self.used.add(key)
        return self.hf[key]

    def copy(self, param, tensor):
        if param.shape != tensor.shape:
            raise ValueError(f'{self.names.get(id(param), "?")}: shape {tuple(param.shape)} '
                             f'≠ HF {tuple(tensor.shape)}')
        param.data.copy_(tensor)
        self.written.add(id(param))


def load_hf_into_model(model, hf_state_dict, args):
    """Copy HF weights into a Megatron model (single PP stage)."""
    from megatron.core import mpu

    num_query_groups = args.num_query_groups if args.group_query_attention else args.num_attention_heads

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    num_layers_per_stage = args.num_layers // args.pipeline_model_parallel_size
    layer_offset = pp_rank * num_layers_per_stage
    c = Copier(model, hf_state_dict)

    # Embeddings (only on first PP stage)
    if hasattr(model, 'embedding'):
        c.copy(model.embedding.word_embeddings.weight, c.get('model.embed_tokens.weight'))

    # Output layer and final layernorm (only on last PP stage)
    if hasattr(model, 'output_layer'):
        c.copy(model.output_layer.weight, c.get('lm_head.weight'))
    if hasattr(model, 'decoder') and getattr(model.decoder, 'final_layernorm', None) is not None:
        c.copy(model.decoder.final_layernorm.weight, c.get('model.norm.weight'))

    for local_idx, layer in enumerate(model.decoder.layers):
        layer_idx = layer_offset + local_idx
        prefix = f'model.layers.{layer_idx}'
        attn = layer.self_attention

        # Attention: q/k/v weights and biases, interleaved per query group
        c.copy(attn.linear_qkv.weight, interleave_qkv(
            c.get(f'{prefix}.self_attn.q_proj.weight'), c.get(f'{prefix}.self_attn.k_proj.weight'),
            c.get(f'{prefix}.self_attn.v_proj.weight'), num_query_groups))
        c.copy(attn.linear_qkv.bias, interleave_qkv(
            c.get(f'{prefix}.self_attn.q_proj.bias'), c.get(f'{prefix}.self_attn.k_proj.bias'),
            c.get(f'{prefix}.self_attn.v_proj.bias'), num_query_groups))
        c.copy(attn.linear_proj.weight, c.get(f'{prefix}.self_attn.o_proj.weight'))
        c.copy(attn.linear_proj.bias, c.get(f'{prefix}.self_attn.o_proj.bias'))

        # Input layernorm (fused into linear_qkv in TE)
        c.copy(attn.linear_qkv.layer_norm_weight, c.get(f'{prefix}.input_layernorm.weight'))

        # Post-attention layernorm (fused into linear_fc1 in TE)
        mlp = layer.mlp
        c.copy(mlp.linear_fc1.layer_norm_weight, c.get(f'{prefix}.post_attention_layernorm.weight'))

        # Dense SwiGLU MLP: fc1 = [gate; up] (Megatron applies act to the first half)
        c.copy(mlp.linear_fc1.weight, torch.cat(
            [c.get(f'{prefix}.mlp.gate_proj.weight'), c.get(f'{prefix}.mlp.up_proj.weight')], dim=0))
        c.copy(mlp.linear_fc1.bias, torch.cat(
            [c.get(f'{prefix}.mlp.gate_proj.bias'), c.get(f'{prefix}.mlp.up_proj.bias')], dim=0))
        c.copy(mlp.linear_fc2.weight, c.get(f'{prefix}.mlp.down_proj.weight'))
        c.copy(mlp.linear_fc2.bias, c.get(f'{prefix}.mlp.down_proj.bias'))

        print(f'  Loaded layer {layer_idx}')

    # Every parameter of this stage written, every HF tensor of this stage used.
    missing = [n for n, p in model.named_parameters() if id(p) not in c.written]
    if missing:
        raise ValueError(f'[pp {pp_rank}] parameters not loaded from HF: {missing}')
    stage_layers = {f'model.layers.{i}.' for i in range(layer_offset, layer_offset + num_layers_per_stage)}
    expected = {k for k in hf_state_dict if any(k.startswith(p) for p in stage_layers)}
    if hasattr(model, 'embedding'):
        expected.add('model.embed_tokens.weight')
    if hasattr(model, 'output_layer'):
        expected |= {'lm_head.weight', 'model.norm.weight'}
    unused = sorted(expected - c.used)
    if unused:
        raise ValueError(f'[pp {pp_rank}] HF tensors not used: {unused}')


def main():
    # Extract our own args before Megatron's parse_args eats sys.argv
    hf_path = None
    save_iteration = 8000
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--hf-path':
            hf_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--save-iteration':
            save_iteration = int(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = filtered_argv

    assert hf_path is not None, "Must specify --hf-path"

    from megatron.training.global_vars import get_args
    from megatron.training.initialize import initialize_megatron
    from megatron.training.checkpointing import save_checkpoint
    from megatron.core import mpu
    from pretrain_gpt import model_provider

    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
            'use_cpu_initialization': True,
        }
    )

    args = get_args()
    validate_hf_config(hf_path, args)
    rank = torch.distributed.get_rank()

    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    model = model_provider(pre_process, post_process).to(args.params_dtype)

    if rank == 0:
        print('Model built.')

    # Each rank loads HF weights sequentially to avoid all ranks holding the full
    # state dict simultaneously
    world_size = torch.distributed.get_world_size()
    for loading_rank in range(world_size):
        if rank == loading_rank:
            print(f'[rank {rank}] Loading HF weights from {hf_path}...')
            hf_state_dict = load_hf_state_dict(hf_path)
            load_hf_into_model(model, hf_state_dict, args)
            del hf_state_dict
            import gc; gc.collect()
            print(f'[rank {rank}] Done loading weights.')
        torch.distributed.barrier()

    if rank == 0:
        print(f'Saving DCP checkpoint at iteration {save_iteration}...')

    save_checkpoint(save_iteration, [model], None, None,
                    num_floating_point_operations_so_far=0)

    if rank == 0:
        print('Done! Checkpoint saved.')

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
