#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict

LAYER = "single_transformer_blocks.0.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-runtime-linear-onnx")


class LowBitLinearRuntimeOnnx(torch.nn.Module):
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

    def recovered_weight(self) -> torch.Tensor:
        # W_q is stored transposed: [packed_cols, out_features].
        # Avoid Tensor bitshift here because the current torch.onnx exporter does
        # not lower aten.__rshift__. Use arithmetic unpack instead:
        # floor(W_q / 2**shift) mod 2**nbits.
        wq = self.wq_t.t().contiguous().to(torch.float32)
        shifts = torch.arange(self.elements_per_sample, dtype=torch.float32, device=wq.device) * float(self.nbits)
        divisors = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=wq.device), shifts)
        base = float(1 << self.nbits)
        shifted = torch.floor(wq.unsqueeze(-1) / divisors)
        unpacked = torch.remainder(shifted, base).reshape(self.out_features, self.in_features)
        scales = self.scales.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        zeros = self.zeros.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        return unpacked * scales + zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.recovered_weight().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, weight)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def initializer_summary(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    initializers = []
    total = 0
    for init in model.graph.initializer:
        nbytes = 1
        for dim in init.dims:
            nbytes *= int(dim)
        # Conservative byte estimate by ONNX data_type.
        # FLOAT=1 -> 4, UINT8=2 -> 1, INT32=6 -> 4, INT64=7 -> 8.
        width = {1: 4, 2: 1, 6: 4, 7: 8, 10: 2, 16: 2}.get(init.data_type, 4)
        nbytes *= width
        total += nbytes
        initializers.append({"name": init.name, "dims": list(init.dims), "data_type": int(init.data_type), "estimated_nbytes": nbytes})
    return {"initializer_count": len(initializers), "initializer_estimated_nbytes": total, "initializers": initializers[:50]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    wq = sd[f"{LAYER}.W_q"]
    scales = sd[f"{LAYER}.scales"]
    zeros = sd[f"{LAYER}.zeros"]
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]

    module = LowBitLinearRuntimeOnnx(wq, scales, zeros, orig_shape, metadata).eval()
    x = torch.randn(2, orig_shape[1], dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = module(x).detach().cpu().numpy()

    onnx_path = OUT_DIR / "lowbit_runtime_linear.onnx"
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"x": x.cpu().numpy().astype(np.float32)})[0]
    diff = y_ort - y_pt

    init_summary = initializer_summary(onnx_path)
    packed_nbytes = tensor_nbytes(wq) + tensor_nbytes(scales) + tensor_nbytes(zeros)
    expanded_fp32_nbytes = int(orig_shape[0] * orig_shape[1] * 4)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "layer": LAYER,
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_nbytes,
        "initializer_summary": init_summary,
        "input_shape": list(x.shape),
        "output_shape": list(y_pt.shape),
        "mean_abs_error": float(np.abs(diff).mean()),
        "max_abs_error": float(np.abs(diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(y_ort, y_pt, rtol=1e-4, atol=1e-5)),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "constant_folding_disabled": report["constant_folding_disabled"],
        "unpack_lowering": report["unpack_lowering"],
        "onnx_size_bytes": report["onnx_size_bytes"],
        "packed_nbytes": report["packed_nbytes"],
        "expanded_fp32_weight_nbytes": report["expanded_fp32_weight_nbytes"],
        "initializer_estimated_nbytes": init_summary["initializer_estimated_nbytes"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
