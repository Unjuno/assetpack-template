#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import (
    LOWBIT_REF,
    load_lowbit_transformer_state_dict,
    quantized_prefixes,
    recover_quantized_weight,
)

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-recovered-transformer-state-dict")


def recovered_keys(lowbit_sd: dict[str, Any]) -> set[str]:
    q_prefixes = set(quantized_prefixes(lowbit_sd))
    out: set[str] = set()
    for key in lowbit_sd:
        if any(key == f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]):
            continue
        out.add(key)
    for prefix in q_prefixes:
        out.add(f"{prefix}.weight")
    return out


def passthrough_tensor_keys(lowbit_sd: dict[str, Any]) -> list[str]:
    q_prefixes = set(quantized_prefixes(lowbit_sd))
    skip = {f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]}
    keys = []
    for key in sorted(lowbit_sd):
        if key in skip:
            continue
        if isinstance(lowbit_sd[key], torch.Tensor):
            keys.append(key)
    return keys


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    tf = t.detach().to(torch.float32)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(tf.min()),
        "max": float(tf.max()),
        "mean": float(tf.mean()),
        "std": float(tf.std()),
    }


def compare_tensor(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if list(a.shape) != list(b.shape):
        return {"shape_match": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    diff = af - bf
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "a_dtype": str(a.dtype),
        "b_dtype": str(b.dtype),
        "exact_equal_float32": bool(torch.equal(af, bf)),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    unpacked_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))

    q_prefixes = quantized_prefixes(lowbit_sd)
    rec_keys = recovered_keys(lowbit_sd)

    with safe_open(unpacked_path, framework="pt", device="cpu") as ref_file:
        ref_keys = set(ref_file.keys())
        missing_vs_ref = sorted(ref_keys - rec_keys)
        unexpected_vs_ref = sorted(rec_keys - ref_keys)

        quantized_results = []
        quantized_failures = []
        for prefix in q_prefixes:
            ref_key = f"{prefix}.weight"
            if ref_key not in ref_keys:
                failure = {"key": ref_key, "error": "missing in unpacked reference"}
                quantized_failures.append(failure)
                quantized_results.append(failure)
                continue
            weight, meta = recover_quantized_weight(lowbit_sd, prefix, output_dtype=torch.float32)
            ref = ref_file.get_tensor(ref_key).cpu().to(torch.float32)
            if list(weight.shape) != list(ref.shape) and list(weight.t().shape) == list(ref.shape):
                weight = weight.t().contiguous()
                meta["transposed_for_compare"] = True
            cmp = compare_tensor(weight, ref)
            result = {"key": ref_key, "metadata": meta, **cmp}
            quantized_results.append(result)
            if not cmp.get("shape_match") or cmp.get("max_abs_error") != 0.0:
                quantized_failures.append(result)
            del weight, ref

        passthrough_results = []
        passthrough_failures = []
        for key in passthrough_tensor_keys(lowbit_sd):
            if key not in ref_keys:
                failure = {"key": key, "error": "missing in unpacked reference"}
                passthrough_failures.append(failure)
                passthrough_results.append(failure)
                continue
            cmp = compare_tensor(lowbit_sd[key].cpu(), ref_file.get_tensor(key).cpu())
            result = {"key": key, **cmp}
            passthrough_results.append(result)
            if not cmp.get("shape_match") or cmp.get("max_abs_error") != 0.0:
                passthrough_failures.append(result)

    report = {
        "lowbit_ref": LOWBIT_REF,
        "unpacked_ref": UNPACKED_REF,
        "lowbit_path": str(lowbit_path),
        "unpacked_path": str(unpacked_path),
        "lowbit_top_key_count": len(lowbit_sd),
        "quantized_prefix_count": len(q_prefixes),
        "recovered_key_count": len(rec_keys),
        "reference_key_count": len(ref_keys),
        "missing_vs_reference_count": len(missing_vs_ref),
        "unexpected_vs_reference_count": len(unexpected_vs_ref),
        "missing_vs_reference_sample": missing_vs_ref[:50],
        "unexpected_vs_reference_sample": unexpected_vs_ref[:50],
        "quantized_compared_count": len(quantized_results),
        "quantized_failure_count": len(quantized_failures),
        "passthrough_compared_count": len(passthrough_results),
        "passthrough_failure_count": len(passthrough_failures),
        "all_keys_match": not missing_vs_ref and not unexpected_vs_ref,
        "all_quantized_exact": not quantized_failures,
        "all_passthrough_exact": not passthrough_failures,
        "quantized_failures": quantized_failures[:20],
        "passthrough_failures": passthrough_failures[:20],
        "quantized_results_sample": quantized_results[:20],
        "passthrough_results_sample": passthrough_results[:20],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_keys_match": report["all_keys_match"],
        "quantized_prefix_count": report["quantized_prefix_count"],
        "recovered_key_count": report["recovered_key_count"],
        "reference_key_count": report["reference_key_count"],
        "quantized_failure_count": report["quantized_failure_count"],
        "passthrough_failure_count": report["passthrough_failure_count"],
    }, indent=2))
    if not report["all_keys_match"] or not report["all_quantized_exact"] or not report["all_passthrough_exact"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
