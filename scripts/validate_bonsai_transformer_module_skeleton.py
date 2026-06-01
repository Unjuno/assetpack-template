#!/usr/bin/env python3
from __future__ import annotations

import json
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
OUT_DIR = Path("reports/bonsai-transformer-module-skeleton")


def recovered_keys(lowbit_sd: dict[str, Any]) -> set[str]:
    q_prefixes = set(quantized_prefixes(lowbit_sd))
    quantized_parts = {f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]}
    keys: set[str] = set()
    for key, value in lowbit_sd.items():
        if key in quantized_parts:
            continue
        if isinstance(value, torch.Tensor):
            keys.add(key)
    for prefix in q_prefixes:
        keys.add(f"{prefix}.weight")
    return keys


def recovered_shapes(lowbit_sd: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in lowbit_sd.items():
        if isinstance(value, torch.Tensor) and not any(key.endswith(f".{suffix}") for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]):
            shapes[key] = list(value.shape)
    for prefix in quantized_prefixes(lowbit_sd):
        orig_shape = [int(v) for v in lowbit_sd[f"{prefix}.orig_shape"].tolist()]
        shapes[f"{prefix}.weight"] = orig_shape
    return shapes


def ref_shapes(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return {key: list(f.get_tensor(key).shape) for key in f.keys()}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=TRANSFORMER_CONFIG, repo_type="model"))
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    _lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)

    model = FluxTransformer2DModel.load_config(str(config_path.parent))
    module = FluxTransformer2DModel.from_config(model)
    module_sd = module.state_dict()

    rec_keys = recovered_keys(lowbit_sd)
    rec_shapes = recovered_shapes(lowbit_sd)
    module_keys = set(module_sd.keys())
    reference_shapes = ref_shapes(ref_path)

    module_missing_vs_recovered = sorted(rec_keys - module_keys)
    module_unexpected_vs_recovered = sorted(module_keys - rec_keys)
    reference_missing_vs_module = sorted(set(reference_shapes) - module_keys)
    reference_unexpected_vs_module = sorted(module_keys - set(reference_shapes))

    shape_mismatches = []
    for key in sorted(rec_keys & module_keys):
        expected = rec_shapes.get(key)
        actual = list(module_sd[key].shape)
        if expected != actual:
            shape_mismatches.append({"key": key, "recovered_shape": expected, "module_shape": actual})

    ref_shape_mismatches = []
    for key in sorted(set(reference_shapes) & module_keys):
        if reference_shapes[key] != list(module_sd[key].shape):
            ref_shape_mismatches.append({"key": key, "reference_shape": reference_shapes[key], "module_shape": list(module_sd[key].shape)})

    report = {
        "lowbit_ref": LOWBIT_REF,
        "unpacked_ref": UNPACKED_REF,
        "module_class": "FluxTransformer2DModel",
        "config_path": str(config_path),
        "reference_path": str(ref_path),
        "recovered_key_count": len(rec_keys),
        "module_key_count": len(module_keys),
        "reference_key_count": len(reference_shapes),
        "module_missing_vs_recovered_count": len(module_missing_vs_recovered),
        "module_unexpected_vs_recovered_count": len(module_unexpected_vs_recovered),
        "reference_missing_vs_module_count": len(reference_missing_vs_module),
        "reference_unexpected_vs_module_count": len(reference_unexpected_vs_module),
        "shape_mismatch_count": len(shape_mismatches),
        "reference_shape_mismatch_count": len(ref_shape_mismatches),
        "all_module_keys_match_recovered": not module_missing_vs_recovered and not module_unexpected_vs_recovered,
        "all_module_shapes_match_recovered": not shape_mismatches,
        "all_module_keys_match_reference": not reference_missing_vs_module and not reference_unexpected_vs_module,
        "all_module_shapes_match_reference": not ref_shape_mismatches,
        "module_missing_vs_recovered_sample": module_missing_vs_recovered[:50],
        "module_unexpected_vs_recovered_sample": module_unexpected_vs_recovered[:50],
        "shape_mismatches_sample": shape_mismatches[:50],
        "reference_shape_mismatches_sample": ref_shape_mismatches[:50],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_module_keys_match_recovered": report["all_module_keys_match_recovered"],
        "all_module_shapes_match_recovered": report["all_module_shapes_match_recovered"],
        "module_key_count": report["module_key_count"],
        "recovered_key_count": report["recovered_key_count"],
        "shape_mismatch_count": report["shape_mismatch_count"],
    }, indent=2))
    if not report["all_module_keys_match_recovered"] or not report["all_module_shapes_match_recovered"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
