#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
TRANSFORMER_CONFIG = "transformer/config.json"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-single-block-inventory")
BLOCK_INDEX = 0
BLOCK_PREFIX = f"single_transformer_blocks.{BLOCK_INDEX}"


def tensor_shape_map_from_safetensors(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: list(f.get_tensor(k).shape) for k in f.keys()}


def lowbit_recovered_shapes(sd: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    q_prefixes = set(quantized_prefixes(sd))
    quantized_parts = {f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]}
    for key, value in sd.items():
        if key in quantized_parts:
            continue
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
    for prefix in q_prefixes:
        shapes[f"{prefix}.weight"] = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    return shapes


def block_related_module_keys(module_keys: list[str]) -> list[str]:
    return sorted(k for k in module_keys if k.startswith(BLOCK_PREFIX + "."))


def block_related_checkpoint_keys(keys: set[str]) -> list[str]:
    return sorted(k for k in keys if k.startswith(BLOCK_PREFIX + "."))


def classify_checkpoint_key(key: str, q_prefixes: set[str]) -> str:
    for prefix in q_prefixes:
        if key == f"{prefix}.weight":
            if prefix.endswith("to_qkv_mlp_proj"):
                return "quantized_fused_qkv_mlp_weight"
            return "quantized_linear_weight"
    if key.endswith(".bias"):
        return "bias_passthrough"
    if ".norm." in key or key.endswith(".scale") or key.endswith(".linear.weight") or key.endswith(".linear.bias"):
        return "norm_or_modulation_passthrough"
    return "other_passthrough"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=TRANSFORMER_CONFIG, repo_type="model"))
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    _lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    recovered_shapes = lowbit_recovered_shapes(lowbit_sd)
    q_prefixes = set(quantized_prefixes(lowbit_sd))

    config = FluxTransformer2DModel.load_config(str(config_path.parent))
    module = FluxTransformer2DModel.from_config(config)
    module_shapes = {k: list(v.shape) for k, v in module.state_dict().items()}
    reference_shapes = tensor_shape_map_from_safetensors(ref_path)

    module_block_keys = block_related_module_keys(list(module_shapes))
    checkpoint_block_keys = block_related_checkpoint_keys(set(reference_shapes))
    recovered_block_keys = block_related_checkpoint_keys(set(recovered_shapes))

    checkpoint_entries = []
    for key in checkpoint_block_keys:
        checkpoint_entries.append({
            "key": key,
            "shape": reference_shapes[key],
            "recovered_shape": recovered_shapes.get(key),
            "class": classify_checkpoint_key(key, q_prefixes),
            "available_from_lowbit": key in recovered_shapes,
        })

    # Diffusers module uses split keys; checkpoint may use fused keys. Summarize likely fused mappings.
    split_candidates = [k for k in module_block_keys if any(part in k for part in [".attn.to_q.", ".attn.to_k.", ".attn.to_v.", ".proj_mlp."])]
    fused_candidates = [k for k in checkpoint_block_keys if "to_qkv_mlp_proj" in k]

    missing_recovered = sorted(set(checkpoint_block_keys) - set(recovered_block_keys))
    unexpected_recovered = sorted(set(recovered_block_keys) - set(checkpoint_block_keys))
    shape_mismatches = [
        {"key": key, "reference_shape": reference_shapes[key], "recovered_shape": recovered_shapes.get(key)}
        for key in checkpoint_block_keys
        if recovered_shapes.get(key) != reference_shapes[key]
    ]

    report = {
        "source_model_ref": LOWBIT_REF,
        "reference_model_ref": UNPACKED_REF,
        "block_prefix": BLOCK_PREFIX,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "module_class": "FluxTransformer2DModel",
        "module_block_key_count": len(module_block_keys),
        "checkpoint_block_key_count": len(checkpoint_block_keys),
        "recovered_block_key_count": len(recovered_block_keys),
        "missing_recovered_count": len(missing_recovered),
        "unexpected_recovered_count": len(unexpected_recovered),
        "shape_mismatch_count": len(shape_mismatches),
        "all_checkpoint_block_keys_recoverable": not missing_recovered and not shape_mismatches,
        "checkpoint_entries": checkpoint_entries,
        "module_block_keys_sample": module_block_keys[:120],
        "checkpoint_block_keys": checkpoint_block_keys,
        "split_module_candidates": split_candidates,
        "fused_checkpoint_candidates": fused_candidates,
        "missing_recovered": missing_recovered,
        "unexpected_recovered": unexpected_recovered,
        "shape_mismatches": shape_mismatches,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "block_prefix": BLOCK_PREFIX,
        "module_block_key_count": report["module_block_key_count"],
        "checkpoint_block_key_count": report["checkpoint_block_key_count"],
        "recovered_block_key_count": report["recovered_block_key_count"],
        "all_checkpoint_block_keys_recoverable": report["all_checkpoint_block_keys_recoverable"],
        "fused_checkpoint_candidates": fused_candidates,
        "split_module_candidate_count": len(split_candidates),
    }, indent=2))
    return 0 if report["all_checkpoint_block_keys_recoverable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
