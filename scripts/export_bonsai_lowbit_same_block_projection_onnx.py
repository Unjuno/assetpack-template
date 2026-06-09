#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_runtime_linear_onnx import LowBitLinearRuntimeOnnx, initializer_summary, tensor_nbytes

BLOCK_INDEX = 0
QKV_MLP_PROJ = f"single_transformer_blocks.{BLOCK_INDEX}.attn.to_qkv_mlp_proj"
TO_OUT = f"single_transformer_blocks.{BLOCK_INDEX}.attn.to_out"
LAYERS = [QKV_MLP_PROJ, TO_OUT]
OUT_DIR = Path("reports/bonsai-lowbit-same-block-projection-onnx")


class SameBlockProjectionBundle(torch.nn.Module):
    def __init__(self, qkv_mlp_proj: LowBitLinearRuntimeOnnx, to_out: LowBitLinearRuntimeOnnx):
        super().__init__()
        self.qkv_mlp_proj = qkv_mlp_proj
        self.to_out = to_out
        self.hidden_in_features = int(qkv_mlp_proj.in_features)
        self.attn_out_in_features = int(to_out.in_features)

    def forward(self, hidden: torch.Tensor, attn_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qkv_mlp = self.qkv_mlp_proj(hidden)
        out = self.to_out(attn_context)
        return qkv_mlp, out


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
    qkv_mlp = load_runtime_linear(sd, QKV_MLP_PROJ)
    to_out = load_runtime_linear(sd, TO_OUT)
    bundle = SameBlockProjectionBundle(qkv_mlp, to_out).eval()

    hidden = torch.randn(2, bundle.hidden_in_features, dtype=torch.float32) / 10.0
    attn_context = torch.randn(2, bundle.attn_out_in_features, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        qkv_mlp_pt, out_pt = [y.detach().cpu().numpy() for y in bundle(hidden, attn_context)]

    onnx_path = OUT_DIR / "lowbit_same_block_projection_bundle.onnx"
    torch.onnx.export(
        bundle,
        (hidden, attn_context),
        str(onnx_path),
        input_names=["hidden", "attn_context"],
        output_names=["qkv_mlp", "out"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={
            "hidden": {0: "batch"},
            "attn_context": {0: "batch"},
            "qkv_mlp": {0: "batch"},
            "out": {0: "batch"},
        },
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    qkv_mlp_ort, out_ort = session.run(
        None,
        {
            "hidden": hidden.cpu().numpy().astype(np.float32),
            "attn_context": attn_context.cpu().numpy().astype(np.float32),
        },
    )
    outputs = []
    for name, layer, ort_out, pt_out in [
        ("qkv_mlp", QKV_MLP_PROJ, qkv_mlp_ort, qkv_mlp_pt),
        ("out", TO_OUT, out_ort, out_pt),
    ]:
        diff = ort_out - pt_out
        outputs.append({
            "name": name,
            "layer": layer,
            "output_shape": list(pt_out.shape),
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort_out, pt_out, rtol=1e-4, atol=1e-5)),
        })

    external_data_path = Path(str(onnx_path) + ".data")
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.W_q"]) + tensor_nbytes(sd[f"{prefix}.scales"]) + tensor_nbytes(sd[f"{prefix}.zeros"]) for prefix in LAYERS)
    expanded_fp32_nbytes = qkv_mlp.out_features * qkv_mlp.in_features * 4 + to_out.out_features * to_out.in_features * 4
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in outputs)
    init_summary = initializer_summary(onnx_path)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "block_index": BLOCK_INDEX,
        "bundle_kind": "same_block_qkv_mlp_proj_and_to_out_projection_bundle",
        "is_real_transformer_block": False,
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
        "inputs": {
            "hidden_shape": list(hidden.shape),
            "attn_context_shape": list(attn_context.shape),
        },
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "same_block_lowbit_projection_bundle_onnxruntime_cpu_verified_not_attention_or_transformer_block",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "bundle_kind": report["bundle_kind"],
        "block_index": report["block_index"],
        "layer_count": report["layer_count"],
        "onnx_size_bytes": report["onnx_size_bytes"],
        "external_data_size_bytes": report["external_data_size_bytes"],
        "packed_nbytes": report["packed_nbytes"],
        "expanded_fp32_weight_nbytes": report["expanded_fp32_weight_nbytes"],
        "max_abs_error": report["max_abs_error"],
        "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"],
    }, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
