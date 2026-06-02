#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-double-block-inventory")
BLOCK_INDEX = 0
BLOCK_PREFIX = f"transformer_blocks.{BLOCK_INDEX}"


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


def block_keys(keys: set[str]) -> list[str]:
    return sorted(k for k in keys if k.startswith(BLOCK_PREFIX + "."))


def classify_key(key: str, q_prefixes: set[str]) -> str:
    for prefix in q_prefixes:
        if key == f"{prefix}.weight":
            if re.match(r"^transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)$", prefix):
                return "quantized_img_attention_linear"
            if re.match(r"^transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)$", prefix):
                return "quantized_txt_attention_linear"
            if re.match(r"^transformer_blocks\.\d+\.ff\.(linear_in|linear_out)$", prefix):
                return "quantized_img_ff_linear"
            if re.match(r"^transformer_blocks\.\d+\.ff_context\.(linear_in|linear_out)$", prefix):
                return "quantized_txt_ff_linear"
            return "quantized_other"
    if ".norm" in key:
        return "norm_or_modulation_passthrough"
    if key.endswith(".bias") or key.endswith(".weight"):
        return "passthrough_tensor"
    return "other_passthrough"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    _lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    reference_shapes = tensor_shape_map_from_safetensors(ref_path)
    recovered_shapes = lowbit_recovered_shapes(lowbit_sd)
    q_prefixes = set(quantized_prefixes(lowbit_sd))

    ref_block_keys = block_keys(set(reference_shapes))
    recovered_block_keys = block_keys(set(recovered_shapes))
    entries = []
    class_counts: dict[str, int] = {}
    for key in ref_block_keys:
        cls = classify_key(key, q_prefixes)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        entries.append({
            "key": key,
            "shape": reference_shapes[key],
            "recovered_shape": recovered_shapes.get(key),
            "class": cls,
            "available_from_lowbit": key in recovered_shapes,
        })

    missing = sorted(set(ref_block_keys) - set(recovered_block_keys))
    unexpected = sorted(set(recovered_block_keys) - set(ref_block_keys))
    shape_mismatches = [
        {"key": key, "reference_shape": reference_shapes[key], "recovered_shape": recovered_shapes.get(key)}
        for key in ref_block_keys
        if recovered_shapes.get(key) != reference_shapes[key]
    ]
    quantized_entries = [e for e in entries if e["class"].startswith("quantized_")]
    passthrough_entries = [e for e in entries if not e["class"].startswith("quantized_")]
    report = {
        "source_model_ref": LOWBIT_REF,
        "reference_model_ref": UNPACKED_REF,
        "block_prefix": BLOCK_PREFIX,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "static_inventory_only": True,
        "instantiates_full_transformer_module": False,
        "checkpoint_block_key_count": len(ref_block_keys),
        "recovered_block_key_count": len(recovered_block_keys),
        "quantized_key_count": len(quantized_entries),
        "passthrough_key_count": len(passthrough_entries),
        "class_counts": dict(sorted(class_counts.items())),
        "missing_recovered_count": len(missing),
        "unexpected_recovered_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "all_checkpoint_block_keys_recoverable": not missing and not shape_mismatches,
        "entries": entries,
        "missing_recovered": missing,
        "unexpected_recovered": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "block_prefix": BLOCK_PREFIX,
        "static_inventory_only": report["static_inventory_only"],
        "checkpoint_block_key_count": report["checkpoint_block_key_count"],
        "recovered_block_key_count": report["recovered_block_key_count"],
        "quantized_key_count": report["quantized_key_count"],
        "passthrough_key_count": report["passthrough_key_count"],
        "class_counts": report["class_counts"],
        "all_checkpoint_block_keys_recoverable": report["all_checkpoint_block_keys_recoverable"],
    }, indent=2))
    return 0 if report["all_checkpoint_block_keys_recoverable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
