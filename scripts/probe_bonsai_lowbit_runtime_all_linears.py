#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from bonsai_lowbit_recover import (
    LOWBIT_REF,
    load_lowbit_transformer_state_dict,
    quantized_prefixes,
    recover_quantized_weight,
    unpack_cols_transposed,
    expand_col_groups,
)

OUT_DIR = Path("reports/bonsai-lowbit-runtime-linears")
LAYER_LIMIT = int(os.getenv("BONSAI_RUNTIME_LINEAR_LIMIT", "0"))
BATCH = int(os.getenv("BONSAI_RUNTIME_LINEAR_BATCH", "2"))


class LowBitLinearRuntime(torch.nn.Module):
    def __init__(self, wq_t: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, orig_shape: list[int], metadata: list[int]):
        super().__init__()
        self.register_buffer("wq_t", wq_t.detach().cpu().contiguous(), persistent=True)
        self.register_buffer("scales", scales.detach().cpu().contiguous(), persistent=True)
        self.register_buffer("zeros", zeros.detach().cpu().contiguous(), persistent=True)
        self.orig_shape = [int(v) for v in orig_shape]
        self.nbits = int(metadata[1])
        self.group_size = int(metadata[2])

    def recovered_weight(self) -> torch.Tensor:
        unpacked = unpack_cols_transposed(self.wq_t, self.nbits, self.orig_shape[1])
        scales = expand_col_groups(self.scales, unpacked.shape[1], self.group_size)
        zeros = expand_col_groups(self.zeros, unpacked.shape[1], self.group_size)
        return unpacked * scales + zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.recovered_weight().to(dtype=x.dtype, device=x.device)
        return torch.nn.functional.linear(x, weight)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def packed_nbytes_for_prefix(sd: dict[str, Any], prefix: str) -> int:
    return sum(
        tensor_nbytes(sd[f"{prefix}.{name}"])
        for name in ["W_q", "scales", "zeros", "orig_shape", "metadata"]
        if isinstance(sd.get(f"{prefix}.{name}"), torch.Tensor)
    )


def check_prefix(sd: dict[str, Any], prefix: str, index: int) -> dict[str, Any]:
    wq = sd[f"{prefix}.W_q"]
    scales = sd[f"{prefix}.scales"]
    zeros = sd[f"{prefix}.zeros"]
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]

    runtime = LowBitLinearRuntime(wq, scales, zeros, orig_shape, metadata).eval()
    ref_weight, ref_meta = recover_quantized_weight(sd, prefix, output_dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(10_000 + index)
    x = torch.randn((BATCH, orig_shape[1]), generator=generator, dtype=torch.float32) / 10.0

    started = time.time()
    with torch.inference_mode():
        y_runtime = runtime(x)
        y_ref = torch.nn.functional.linear(x, ref_weight)
    seconds = round(time.time() - started, 4)

    diff = y_runtime - y_ref
    result = {
        "prefix": prefix,
        "orig_shape": orig_shape,
        "metadata": ref_meta,
        "packed_nbytes": packed_nbytes_for_prefix(sd, prefix),
        "reference_weight_nbytes_fp32": tensor_nbytes(ref_weight),
        "input_shape": list(x.shape),
        "output_shape": list(y_runtime.shape),
        "seconds": seconds,
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(torch.allclose(y_runtime, y_ref, rtol=1e-4, atol=1e-5)),
    }
    del runtime, ref_weight, x, y_runtime, y_ref, diff
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    prefixes = quantized_prefixes(sd)
    if LAYER_LIMIT > 0:
        prefixes = prefixes[:LAYER_LIMIT]

    results = []
    failures = []
    started = time.time()
    for index, prefix in enumerate(prefixes):
        try:
            result = check_prefix(sd, prefix, index)
            results.append(result)
            if not result["allclose_rtol_1e_4_atol_1e_5"]:
                failures.append(result)
        except Exception as exc:
            failure = {"prefix": prefix, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results.append(failure)
            failures.append(failure)

    total_packed_nbytes = sum(r.get("packed_nbytes", 0) for r in results)
    total_reference_weight_nbytes_fp32 = sum(r.get("reference_weight_nbytes_fp32", 0) for r in results)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "comparison_target": "per-layer recovered FP32 linear output",
        "lowbit_path": str(lowbit_path),
        "batch": BATCH,
        "layer_limit": LAYER_LIMIT,
        "runtime_linear_count": len(results),
        "failure_count": len(failures),
        "allclose_all": len(failures) == 0,
        "max_mean_abs_error": max((r.get("mean_abs_error", 0.0) for r in results), default=None),
        "max_abs_error": max((r.get("max_abs_error", 0.0) for r in results), default=None),
        "total_seconds": round(time.time() - started, 3),
        "total_packed_nbytes_sampled": total_packed_nbytes,
        "total_reference_weight_nbytes_fp32_sampled": total_reference_weight_nbytes_fp32,
        "size_ratio_reference_fp32_to_packed": None if total_packed_nbytes == 0 else total_reference_weight_nbytes_fp32 / total_packed_nbytes,
        "failures": failures[:20],
        "results": results,
        "note": "Correctness baseline: packed buffers are persistent, and each runtime layer dequantizes inside forward. This is not yet an optimized packed matmul kernel.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "runtime_linear_count": report["runtime_linear_count"],
        "failure_count": report["failure_count"],
        "allclose_all": report["allclose_all"],
        "max_abs_error": report["max_abs_error"],
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return 0 if report["allclose_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
