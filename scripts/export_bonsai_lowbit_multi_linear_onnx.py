#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_runtime_linear_onnx import LowBitLinearRuntimeOnnx, initializer_summary, tensor_nbytes

LAYERS = [
    "single_transformer_blocks.0.attn.to_out",
    "single_transformer_blocks.1.attn.to_out",
]
OUT_DIR = Path("reports/bonsai-lowbit-multi-linear-onnx")


class MultiLowBitLinearBlock(torch.nn.Module):
    def __init__(self, modules: list[LowBitLinearRuntimeOnnx]):
        super().__init__()
        self.layers = torch.nn.ModuleList(modules)
        in_features = {layer.in_features for layer in self.layers}
        if len(in_features) != 1:
            raise ValueError(f"all layers must share input dim, got {sorted(in_features)}")
        self.in_features = int(next(iter(in_features)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(layer(x) for layer in self.layers)


def load_runtime_linear(sd: dict, prefix: str) -> LowBitLinearRuntimeOnnx:
    wq = sd[f"{prefix}.W_q"]
    scales = sd[f"{prefix}.scales"]
    zeros = sd[f"{prefix}.zeros"]
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
    return LowBitLinearRuntimeOnnx(wq, scales, zeros, orig_shape, metadata)


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    modules = [load_runtime_linear(sd, prefix) for prefix in LAYERS]
    block = MultiLowBitLinearBlock(modules).eval()

    x = torch.randn(2, block.in_features, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = [y.detach().cpu().numpy() for y in block(x)]

    onnx_path = OUT_DIR / "lowbit_multi_linear.onnx"
    output_names = [f"y{i}" for i in range(len(LAYERS))]
    dynamic_axes = {"x": {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    torch.onnx.export(
        block,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes=dynamic_axes,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"x": x.cpu().numpy().astype(np.float32)})
    per_output = []
    ok = True
    for index, (prefix, ort_out, pt_out) in enumerate(zip(LAYERS, y_ort, y_pt)):
        diff = ort_out - pt_out
        item = {
            "index": index,
            "layer": prefix,
            "output_shape": list(pt_out.shape),
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort_out, pt_out, rtol=1e-4, atol=1e-5)),
        }
        ok = ok and item["allclose_rtol_1e_4_atol_1e_5"]
        per_output.append(item)

    external_data_path = Path(str(onnx_path) + ".data")
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.W_q"]) + tensor_nbytes(sd[f"{prefix}.scales"]) + tensor_nbytes(sd[f"{prefix}.zeros"]) for prefix in LAYERS)
    expanded_fp32_nbytes = sum(module.out_features * module.in_features * 4 for module in modules)
    init_summary = initializer_summary(onnx_path)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "block_kind": "multi_runtime_lowbit_linear_bundle",
        "layers": LAYERS,
        "layer_count": len(LAYERS),
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_nbytes,
        "initializer_summary": init_summary,
        "input_shape": list(x.shape),
        "outputs": per_output,
        "max_abs_error": max((item["max_abs_error"] for item in per_output), default=None),
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": ok,
        "claim": "multi_runtime_lowbit_linear_bundle_onnxruntime_cpu_verified_not_transformer_block",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "block_kind": report["block_kind"],
        "layer_count": report["layer_count"],
        "onnx_size_bytes": report["onnx_size_bytes"],
        "external_data_size_bytes": report["external_data_size_bytes"],
        "packed_nbytes": report["packed_nbytes"],
        "expanded_fp32_weight_nbytes": report["expanded_fp32_weight_nbytes"],
        "max_abs_error": report["max_abs_error"],
        "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"],
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
