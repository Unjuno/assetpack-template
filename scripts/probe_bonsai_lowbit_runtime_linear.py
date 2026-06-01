#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, recover_quantized_weight, unpack_cols_transposed, expand_col_groups

LAYER = "single_transformer_blocks.0.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-runtime-linear")


class LowBitLinearRuntime(torch.nn.Module):
    def __init__(self, wq_t: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, orig_shape: list[int], metadata: list[int]):
        super().__init__()
        self.register_buffer("wq_t", wq_t.cpu().contiguous(), persistent=True)
        self.register_buffer("scales", scales.cpu().contiguous(), persistent=True)
        self.register_buffer("zeros", zeros.cpu().contiguous(), persistent=True)
        self.orig_shape = [int(v) for v in orig_shape]
        self.nbits = int(metadata[1])
        self.group_size = int(metadata[2])

    def recovered_weight(self) -> torch.Tensor:
        unpacked = unpack_cols_transposed(self.wq_t, self.nbits, self.orig_shape[1])
        s = expand_col_groups(self.scales, unpacked.shape[1], self.group_size)
        z = expand_col_groups(self.zeros, unpacked.shape[1], self.group_size)
        return unpacked * s + z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Correctness baseline: dequantize inside forward, then matmul.
        # This avoids storing a persistent 150MB FP32/FP16 weight for the layer,
        # but it is not yet optimized.
        w = self.recovered_weight().to(dtype=x.dtype, device=x.device)
        return torch.nn.functional.linear(x, w)


def stat(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    wq = sd[f"{LAYER}.W_q"]
    scales = sd[f"{LAYER}.scales"]
    zeros = sd[f"{LAYER}.zeros"]
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]

    runtime = LowBitLinearRuntime(wq, scales, zeros, orig_shape, metadata).eval()
    ref_weight, meta = recover_quantized_weight(sd, LAYER, output_dtype=torch.float32)
    x = torch.randn(2, orig_shape[1], dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_runtime = runtime(x)
        y_ref = torch.nn.functional.linear(x, ref_weight)
    diff = y_runtime - y_ref
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "layer": LAYER,
        "lowbit_path": str(lowbit_path),
        "metadata": meta,
        "packed_buffers": {
            "W_q": stat(wq),
            "scales": stat(scales),
            "zeros": stat(zeros),
        },
        "input": stat(x),
        "output": stat(y_runtime),
        "mean_abs_error_vs_recovered_weight_linear": float(diff.abs().mean()),
        "max_abs_error_vs_recovered_weight_linear": float(diff.abs().max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(torch.allclose(y_runtime, y_ref, rtol=1e-4, atol=1e-5)),
        "note": "Correctness baseline for low-bit runtime: packed buffers are persistent; dequantization is done inside forward and is not optimized yet.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "layer": LAYER,
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error_vs_recovered_weight_linear"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
