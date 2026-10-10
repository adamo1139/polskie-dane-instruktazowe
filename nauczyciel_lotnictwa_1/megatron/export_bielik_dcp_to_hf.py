#!/usr/bin/env python3
"""Export a Megatron DCP checkpoint of Bielik v3 (Llama, bias in every linear layer) to HF.

Odwrotność load_hf_llama_save_dcp.py; czytanie DCP jak w Ling-V2
tools/convert_dcp_to_safetensors_apt4.py (_load_state_dict_from_keys + planner z metadanych).

- q/k/v: mcore przeplata je per grupa zapytań [Q0 K0 V0 Q1 K1 V1 ...] -> HF q_proj/k_proj/v_proj
  (wagi i biasy); fc1 = [gate; up] -> gate_proj/up_proj (wagi i biasy).
- Warstwy obsługiwane w obu zapisach DCP: osobne klucze `decoder.layers.<i>.…` albo jeden tensor
  `decoder.layers.…` z osią warstw na początku.
- config.json i generation_config.json z oryginalnego modelu HF (--hf-reference), tokenizer
  i szablon czatu z cache SFT (--sft-cache: tokenizer/ z EOS <|im_end|> + chat_template.jinja v13).
- Kontrole: args z common.pt zgadzają się z configiem HF; zbiór kluczy i kształty wyniku są
  IDENTYCZNE jak w --hf-reference; każdy tensor DCP został użyty; dtype bf16.

  python3 export_bielik_dcp_to_hf.py \
      --checkpoint-path .../bielik_1p5b_sft_pilot_.../iter_0000945 \
      --hf-reference /media/nvme_2tb/maked/models/Bielik-1.5B-v3.0-Instruct \
      --sft-cache ~/projects/pretrain/bielik-sft-cache-nauka-lotnictwa-1-v13-8192 \
      --target-path /media/nvme_2tb/maked/models/bielik-1.5b-nauka-lotnictwa-1-pilot \
      --megatron-path ~/projects/pretrain/Ling-V2/Megatron-LM-core_v0.13.0
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


def set_up_planner(self, state_dict, metadata=None, is_coordinator=False) -> None:
    # Rebuild the state dict from the metadata (same patch as Ling-V2's exporter).
    assert not state_dict
    assert metadata is not None
    for k, v in metadata.state_dict_metadata.items():
        if k not in self.keys:
            continue
        if isinstance(v, TensorStorageMetadata):
            v = torch.empty(v.size, dtype=v.properties.dtype)
        state_dict[k] = v
    super(torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner, self
          ).set_up_planner(state_dict, metadata, is_coordinator)


torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner.set_up_planner = set_up_planner

SKIP = ("optimizer", "_extra_state", "rng_state", "chained_0", "rerun_state_machine_state")
TOP = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
}
LAYER_SIMPLE = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "self_attention.linear_proj.bias": "self_attn.o_proj.bias",
    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    "mlp.linear_fc2.bias": "mlp.down_proj.bias",
}


def split_qkv(t, kind, heads, groups, head_dim):
    """mcore [Q0 K0 V0 Q1 K1 V1 ...] (dim 0) -> HF q, k, v."""
    per_group_q = heads // groups * head_dim
    qs, ks, vs = [], [], []
    for g in torch.chunk(t, groups, dim=0):
        q, k, v = g.split([per_group_q, head_dim, head_dim], dim=0)
        qs.append(q); ks.append(k); vs.append(v)
    return {f"self_attn.q_proj.{kind}": torch.cat(qs), f"self_attn.k_proj.{kind}": torch.cat(ks),
            f"self_attn.v_proj.{kind}": torch.cat(vs)}


def map_layer(post, t, cfg):
    if post in LAYER_SIMPLE:
        return {LAYER_SIMPLE[post]: t}
    if post in ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"):
        return split_qkv(t, post.rsplit(".", 1)[1], cfg["num_attention_heads"],
                         cfg["num_key_value_heads"], cfg["head_dim"])
    if post in ("mlp.linear_fc1.weight", "mlp.linear_fc1.bias"):
        kind = post.rsplit(".", 1)[1]
        gate, up = torch.chunk(t, 2, dim=0)
        return {f"mlp.gate_proj.{kind}": gate, f"mlp.up_proj.{kind}": up}
    raise KeyError(f"Unknown layer tensor: {post}")


def reference_shapes(hf_reference):
    index = os.path.join(hf_reference, "model.safetensors.index.json")
    files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) \
        else ["model.safetensors"]
    shapes = {}
    for name in files:
        with safe_open(os.path.join(hf_reference, name), framework="pt") as f:
            for k in f.keys():
                shapes[k] = tuple(f.get_slice(k).get_shape())
    return shapes


def check_args(margs, cfg):
    pairs = {
        "num_layers": cfg["num_hidden_layers"], "hidden_size": cfg["hidden_size"],
        "ffn_hidden_size": cfg["intermediate_size"], "num_attention_heads": cfg["num_attention_heads"],
        "num_query_groups": cfg["num_key_value_heads"], "kv_channels": cfg["head_dim"],
        "norm_epsilon": cfg["rms_norm_eps"], "add_bias_linear": cfg["attention_bias"] and cfg["mlp_bias"],
        "untie_embeddings_and_output_weights": not cfg["tie_word_embeddings"],
    }
    for k, expected in pairs.items():
        if getattr(margs, k, None) != expected:
            raise ValueError(f"common.pt args.{k}={getattr(margs, k, None)!r}, HF config expects {expected!r}")
    if float(margs.rotary_base) != float(cfg["rope_theta"]) or margs.rotary_percent != 1.0:
        raise ValueError("RoPE in checkpoint differs from HF config")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-path", required=True, help="DCP iter_XXXXXXX directory")
    p.add_argument("--hf-reference", required=True, help="original HF Bielik (config + key/shape reference)")
    p.add_argument("--sft-cache", required=True, help="SFT cache with tokenizer/ and chat_template.jinja")
    p.add_argument("--target-path", required=True)
    p.add_argument("--megatron-path", help="needed to unpickle args from common.pt")
    a = p.parse_args()
    if a.megatron_path:
        sys.path.insert(0, a.megatron_path)
    if os.path.exists(a.target_path):
        raise SystemExit(f"Refusing existing output: {a.target_path}")

    cfg = json.load(open(os.path.join(a.hf_reference, "config.json")))
    margs = torch.load(os.path.join(a.checkpoint_path, "common.pt"), weights_only=False)["args"]
    check_args(margs, cfg)
    num_layers = cfg["num_hidden_layers"]

    meta = FileSystemReader(a.checkpoint_path).read_metadata().state_dict_metadata
    keys = {k for k, v in meta.items()
            if isinstance(v, TensorStorageMetadata) and not any(s in k for s in SKIP)}
    print(f"DCP: {len(keys)} model tensors")
    sd = _load_state_dict_from_keys(keys, checkpoint_id=a.checkpoint_path)

    out, used = {}, set()
    per_layer = re.compile(r"decoder\.layers\.(\d+)\.(.+)")
    for k, t in sd.items():
        if k in TOP:
            out[TOP[k]] = t
        elif m := per_layer.fullmatch(k):
            i, post = int(m[1]), m[2]
            out.update({f"model.layers.{i}.{n}": v for n, v in map_layer(post, t, cfg).items()})
        elif k.startswith("decoder.layers."):
            post = k[len("decoder.layers."):]
            if t.shape[0] != num_layers:
                raise ValueError(f"{k}: expected leading layer axis {num_layers}, got {tuple(t.shape)}")
            for i in range(num_layers):
                out.update({f"model.layers.{i}.{n}": v for n, v in map_layer(post, t[i], cfg).items()})
        else:
            raise KeyError(f"Unmapped DCP tensor: {k} {tuple(t.shape)}")
        used.add(k)
    if used != keys:
        raise ValueError(f"DCP tensors not used: {sorted(keys - used)}")

    ref = reference_shapes(a.hf_reference)
    if set(out) != set(ref):
        raise ValueError(f"Key mismatch vs reference. missing: {sorted(set(ref) - set(out))[:10]} "
                         f"extra: {sorted(set(out) - set(ref))[:10]}")
    bad = [(k, tuple(out[k].shape), ref[k]) for k in ref if tuple(out[k].shape) != ref[k]]
    if bad:
        raise ValueError(f"Shape mismatch vs reference: {bad[:10]}")
    if any(v.dtype != torch.bfloat16 for v in out.values()):
        raise ValueError(f"Non-bf16 tensors: {sorted({str(v.dtype) for v in out.values()})}")
    print(f"HF: {len(out)} tensors, keys and shapes identical to {a.hf_reference}")

    os.makedirs(a.target_path)
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.target_path, "model.safetensors"),
              metadata={"format": "pt"})
    for name in ("config.json", "generation_config.json"):
        shutil.copy2(os.path.join(a.hf_reference, name), a.target_path)
    tok_dir = os.path.join(a.sft_cache, "tokenizer")
    for name in os.listdir(tok_dir):
        shutil.copy2(os.path.join(tok_dir, name), a.target_path)
    shutil.copy2(os.path.join(a.sft_cache, "chat_template.jinja"), os.path.join(a.target_path, "chat_template.jinja"))
    tc_path = os.path.join(a.target_path, "tokenizer_config.json")
    tc = json.load(open(tc_path))
    tc["chat_template"] = open(os.path.join(a.sft_cache, "chat_template.jinja")).read()
    json.dump(tc, open(tc_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved {a.target_path}: model.safetensors, config/generation_config from reference, "
          f"tokenizer + v13 chat template from {a.sft_cache}")


if __name__ == "__main__":
    main()
