#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes

OUT_DIR = Path("reports/bonsai-lowbit-runtime-fused-split-onnx")
FUSED_PATTERN = re.compile(r"^single_transformer_blocks\.\d+\.attn\.to_qkv_mlp_proj$")


class LowBitFusedSplitOnnx(torch.nn.Module):
    def __init__(self, wq_t: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, orig_shape: list[int], metadata: list[int]):
        super().__init__()
        self.register_buffer("wq_t", wq_t.detach().cpu().contiguous(), persistent=True)
        self.register_buffer("scales", scales.detach().cpu().contiguous(), persistent=True)
        self.register_buffer("zeros", zeros.detach().cpu().contiguous(), persistent=True)
        self.out_features = int(orig_shape[0])
        self.in_features = int(orig_shape[1])
        self.nbits = int(metadata[1])
        self.group_size = int(metadata[2])
        self.packed_cols = int(wq_t.shape[0])
        self.elements_per_sample = self.in_features // self.packed_cols
        self.hidden_size = self.in_features
        self.mlp_size = self.out_features - 3 * self.hidden_size
        if self.mlp_size <= 0:
            raise ValueError(f"invalid fused qkv_mlp shape: {orig_shape}")

    def recovered_weight(self) -> torch.Tensor:
        # ONNX-friendly unpack: floor(W_q / 2**shift) mod 2**nbits.
        wq = self.wq_t.t().contiguous().to(torch.float32)
        shifts = torch.arange(self.elements_per_sample, dtype=torch.float32, device=wq.device) * float(self.nbits)
        divisors = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=wq.device), shifts)
        shifted = torch.floor(wq.unsqueeze(-1) / divisors)
        unpacked = torch.remainder(shifted, float(1 << self.nbits)).reshape(self.out_features, self.in_features)
        scales = self.scales.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        zeros = self.zeros.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        return unpacked * scales + zeros

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = torch.nn.functional.linear(x, self.recovered_weight().to(dtype=x.dtype))
        return torch.split(fused, [self.hidden_size, self.hidden_size, self.hidden_size, self.mlp_size], dim=-1)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def safe_name(prefix: str) -> str:
    return prefix.replace(".", "_").replace("/", "_")


def initializer_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    total = 0
    max_init = 0
    count = 0
    for init in model.graph.initializer:
        nbytes = 1
        for dim in init.dims:
            nbytes *= int(dim)
        width = {1: 4, 2: 1, 6: 4, 7: 8, 10: 2, 16: 2}.get(init.data_type, 4)
        nbytes *= width
        total += nbytes
        max_init = max(max_init, nbytes)
        count += 1
    return {"initializer_count": count, "initializer_estimated_nbytes": total, "max_initializer_estimated_nbytes": max_init}


def probe_one(sd: dict[str, Any], prefix: str, index: int) -> dict[str, Any]:
    wq = sd[f"{prefix}.W_q"]
    scales = sd[f"{prefix}.scales"]
    zeros = sd[f"{prefix}.zeros"]
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]

    module = LowBitFusedSplitOnnx(wq, scales, zeros, orig_shape, metadata).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(40_000 + index)
    x = torch.randn(2, orig_shape[1], dtype=torch.float32, generator=generator) / 10.0
    with torch.inference_mode():
        y_pt = [t.detach().cpu().numpy() for t in module(x)]

    onnx_path = OUT_DIR / f"{index:03d}_{safe_name(prefix)}.onnx"
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=["q", "k", "v", "mlp"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={"x": {0: "batch"}, "q": {0: "batch"}, "k": {0: "batch"}, "v": {0: "batch"}, "mlp": {0: "batch"}},
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"x": x.cpu().numpy().astype(np.float32)})
    diffs = [ort_out - pt_out for ort_out, pt_out in zip(y_ort, y_pt)]
    max_abs_error = max(float(np.abs(d).max()) for d in diffs)
    mean_abs_error = max(float(np.abs(d).mean()) for d in diffs)
    allclose = all(np.allclose(ort_out, pt_out, rtol=1e-4, atol=1e-5) for ort_out, pt_out in zip(y_ort, y_pt))

    packed_nbytes = tensor_nbytes(wq) + tensor_nbytes(scales) + tensor_nbytes(zeros)
    expanded_fp32_weight_nbytes = int(orig_shape[0] * orig_shape[1] * 4)
    init = initializer_summary(onnx_path)
    ratio = init["initializer_estimated_nbytes"] / expanded_fp32_weight_nbytes if expanded_fp32_weight_nbytes else None
    return {
        "prefix": prefix,
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "orig_shape": orig_shape,
        "hidden_size": module.hidden_size,
        "mlp_size": module.mlp_size,
        "metadata": {"nbits": int(metadata[1]), "group_size": int(metadata[2]), "raw": metadata},
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_weight_nbytes,
        "initializer_summary": init,
        "initializer_to_expanded_fp32_ratio": ratio,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "allclose_rtol_1e_4_atol_1e_5": bool(allclose),
        "not_folded_to_expanded_weight": bool(ratio is not None and ratio < 0.5),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    prefixes = [p for p in quantized_prefixes(sd) if FUSED_PATTERN.match(p)]
    results = []
    failures = []
    for index, prefix in enumerate(prefixes):
        try:
            result = probe_one(sd, prefix, index)
            results.append(result)
            if not result["allclose_rtol_1e_4_atol_1e_5"] or not result["not_folded_to_expanded_weight"]:
                failures.append(result)
        except Exception as exc:
            failure = {"prefix": prefix, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results.append(failure)
            failures.append(failure)

    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "target": "single_transformer_blocks.*.attn.to_qkv_mlp_proj ONNX split into q/k/v/mlp",
        "lowbit_path": str(lowbit_path),
        "fused_projection_count": len(results),
        "failure_count": len(failures),
        "all_passed": len(failures) == 0,
        "max_abs_error": max((r.get("max_abs_error", 0.0) for r in results), default=None),
        "max_initializer_to_expanded_fp32_ratio": max((r.get("initializer_to_expanded_fp32_ratio", 0.0) for r in results), default=None),
        "failures": failures[:20],
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "fused_projection_count": report["fused_projection_count"],
        "failure_count": report["failure_count"],
        "all_passed": report["all_passed"],
        "max_abs_error": report["max_abs_error"],
        "max_initializer_to_expanded_fp32_ratio": report["max_initializer_to_expanded_fp32_ratio"],
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
