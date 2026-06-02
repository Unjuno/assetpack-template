#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-full-transformer-inventory")


def tensor_shape_map_from_safetensors(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: list(f.get_tensor(k).shape) for k in f.keys()}


def recovered_shape_map_from_lowbit(sd: dict[str, Any]) -> dict[str, list[int]]:
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


def classify_key(key: str, q_prefixes: set[str]) -> str:
    for prefix in q_prefixes:
        if key == f"{prefix}.weight":
            if prefix.startswith("single_transformer_blocks."):
                return "quantized_single_block_linear"
            if prefix.startswith("transformer_blocks."):
                return "quantized_double_block_linear"
            return "quantized_other_linear"
    if key.startswith("single_transformer_blocks."):
        if ".attn.norm_" in key:
            return "single_block_attention_norm_passthrough"
        return "single_block_other_passthrough"
    if key.startswith("transformer_blocks."):
        if ".attn.norm" in key:
            return "double_block_attention_norm_passthrough"
        if ".norm" in key:
            return "double_block_modulation_or_norm_passthrough"
        return "double_block_other_passthrough"
    if key.startswith("x_embedder."):
        return "input_image_embedder_passthrough"
    if key.startswith("context_embedder."):
        return "input_text_context_embedder_passthrough"
    if key.startswith("time_text_embed."):
        return "time_text_embedding_passthrough"
    if key.startswith("norm_out."):
        return "final_norm_passthrough"
    if key.startswith("proj_out."):
        return "final_projection_passthrough"
    if key.startswith("pos_embed") or "pos_embed" in key or "rope" in key.lower():
        return "positional_embedding_passthrough"
    return "other_passthrough"


def family(key: str) -> str:
    if key.startswith("single_transformer_blocks."):
        return "single_transformer_blocks"
    if key.startswith("transformer_blocks."):
        return "transformer_blocks"
    return key.split(".", 1)[0]


def block_index(key: str) -> int | None:
    m = re.match(r"^(?:single_transformer_blocks|transformer_blocks)\.(\d+)\.", key)
    return None if m is None else int(m.group(1))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    q_prefixes = set(quantized_prefixes(lowbit_sd))
    reference_shapes = tensor_shape_map_from_safetensors(ref_path)
    recovered_shapes = recovered_shape_map_from_lowbit(lowbit_sd)

    reference_keys = set(reference_shapes)
    recovered_keys = set(recovered_shapes)
    missing = sorted(reference_keys - recovered_keys)
    unexpected = sorted(recovered_keys - reference_keys)
    shape_mismatches = [
        {"key": key, "reference_shape": reference_shapes[key], "recovered_shape": recovered_shapes.get(key)}
        for key in sorted(reference_keys & recovered_keys)
        if reference_shapes[key] != recovered_shapes[key]
    ]

    entries = []
    class_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    block_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(reference_keys):
        cls = classify_key(key, q_prefixes)
        fam = family(key)
        idx = block_index(key)
        class_counts[cls] += 1
        family_counts[fam] += 1
        if idx is not None:
            block_family_counts[f"{fam}.{idx}"][cls] += 1
        entries.append({
            "key": key,
            "family": fam,
            "block_index": idx,
            "class": cls,
            "reference_shape": reference_shapes[key],
            "recovered_shape": recovered_shapes.get(key),
            "available_from_lowbit": key in recovered_shapes,
            "shape_matches": recovered_shapes.get(key) == reference_shapes[key],
        })

    quantized_reference_keys = [e for e in entries if e["class"].startswith("quantized_")]
    passthrough_reference_keys = [e for e in entries if not e["class"].startswith("quantized_")]
    missing_entries = [e for e in entries if not e["available_from_lowbit"]]
    mismatch_entries = [e for e in entries if e["available_from_lowbit"] and not e["shape_matches"]]

    report = {
        "source_model_ref": LOWBIT_REF,
        "reference_model_ref": UNPACKED_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "static_inventory_only": True,
        "instantiates_full_transformer_module": False,
        "reference_key_count": len(reference_keys),
        "recovered_key_count": len(recovered_keys),
        "quantized_prefix_count": len(q_prefixes),
        "quantized_reference_key_count": len(quantized_reference_keys),
        "passthrough_reference_key_count": len(passthrough_reference_keys),
        "missing_recovered_count": len(missing),
        "unexpected_recovered_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "all_reference_keys_recoverable": not missing and not shape_mismatches,
        "class_counts": dict(sorted(class_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "block_family_counts": {k: dict(sorted(v.items())) for k, v in sorted(block_family_counts.items())},
        "missing_recovered": missing,
        "unexpected_recovered": unexpected[:200],
        "shape_mismatches": shape_mismatches,
        "missing_entries": missing_entries[:200],
        "mismatch_entries": mismatch_entries[:200],
        "entries": entries,
        "lowbit_path": str(lowbit_path),
        "reference_path": str(ref_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "reference_key_count": report["reference_key_count"],
        "recovered_key_count": report["recovered_key_count"],
        "quantized_prefix_count": report["quantized_prefix_count"],
        "quantized_reference_key_count": report["quantized_reference_key_count"],
        "passthrough_reference_key_count": report["passthrough_reference_key_count"],
        "missing_recovered_count": report["missing_recovered_count"],
        "shape_mismatch_count": report["shape_mismatch_count"],
        "all_reference_keys_recoverable": report["all_reference_keys_recoverable"],
        "class_counts": report["class_counts"],
    }, indent=2))
    return 0 if report["all_reference_keys_recoverable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
