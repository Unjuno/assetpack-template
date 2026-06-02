#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-passthrough-value-equivalence")


def passthrough_keys(sd: dict[str, Any]) -> list[str]:
    q_prefixes = set(quantized_prefixes(sd))
    quantized_parts = {f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]}
    keys = []
    for key, value in sd.items():
        if key in quantized_parts:
            continue
        if key.endswith((".W_q", ".scales", ".zeros", ".orig_shape", ".metadata")):
            continue
        if isinstance(value, torch.Tensor):
            keys.append(key)
    return sorted(keys)


def tensor_stat(t: torch.Tensor) -> dict[str, Any]:
    y = t.detach().to(torch.float32)
    finite = torch.isfinite(y)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "finite": bool(finite.all()),
        "min": float(y[finite].min()) if bool(finite.any()) else None,
        "max": float(y[finite].max()) if bool(finite.any()) else None,
        "mean": float(y[finite].mean()) if bool(finite.any()) else None,
        "std": float(y[finite].std()) if int(finite.sum()) > 1 else None,
    }


def compare_tensor(key: str, lowbit_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> dict[str, Any]:
    shape_matches = list(lowbit_tensor.shape) == list(ref_tensor.shape)
    dtype_matches = str(lowbit_tensor.dtype) == str(ref_tensor.dtype)
    if not shape_matches:
        return {
            "key": key,
            "shape_matches": False,
            "dtype_matches": dtype_matches,
            "lowbit": tensor_stat(lowbit_tensor),
            "reference": tensor_stat(ref_tensor),
            "allclose_rtol_0_atol_0": False,
            "allclose_rtol_1e_6_atol_1e_7": False,
            "mean_abs_error": None,
            "max_abs_error": None,
        }
    low = lowbit_tensor.detach().cpu()
    ref = ref_tensor.detach().cpu()
    diff = low.to(torch.float32) - ref.to(torch.float32)
    return {
        "key": key,
        "shape_matches": True,
        "dtype_matches": dtype_matches,
        "lowbit": tensor_stat(low),
        "reference": tensor_stat(ref),
        "allclose_rtol_0_atol_0": bool(torch.equal(low, ref)),
        "allclose_rtol_1e_6_atol_1e_7": bool(torch.allclose(low.to(torch.float32), ref.to(torch.float32), rtol=1e-6, atol=1e-7)),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    keys = passthrough_keys(lowbit_sd)
    results = []
    failures = []
    with safe_open(ref_path, framework="pt", device="cpu") as ref_file:
        ref_keys = set(ref_file.keys())
        for key in keys:
            if key not in ref_keys:
                failure = {"key": key, "missing_in_reference": True}
                results.append(failure)
                failures.append(failure)
                continue
            result = compare_tensor(key, lowbit_sd[key], ref_file.get_tensor(key))
            results.append(result)
            if not result.get("shape_matches") or not result.get("allclose_rtol_0_atol_0"):
                failures.append(result)

    exact_match_count = sum(1 for r in results if r.get("allclose_rtol_0_atol_0"))
    tolerant_match_count = sum(1 for r in results if r.get("allclose_rtol_1e_6_atol_1e_7"))
    report = {
        "source_model_ref": LOWBIT_REF,
        "reference_model_ref": UNPACKED_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "non-quantized passthrough tensor value equivalence",
        "passthrough_key_count": len(keys),
        "failure_count": len(failures),
        "exact_match_count": exact_match_count,
        "tolerant_match_count": tolerant_match_count,
        "all_passthrough_tensors_exact": len(failures) == 0,
        "max_abs_error": max((r.get("max_abs_error") or 0.0 for r in results), default=0.0),
        "failures": failures[:50],
        "results": results,
        "lowbit_path": str(lowbit_path),
        "reference_path": str(ref_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "passthrough_key_count": report["passthrough_key_count"],
        "failure_count": report["failure_count"],
        "exact_match_count": report["exact_match_count"],
        "tolerant_match_count": report["tolerant_match_count"],
        "all_passthrough_tensors_exact": report["all_passthrough_tensors_exact"],
        "max_abs_error": report["max_abs_error"],
    }, indent=2))
    return 0 if report["all_passthrough_tensors_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
