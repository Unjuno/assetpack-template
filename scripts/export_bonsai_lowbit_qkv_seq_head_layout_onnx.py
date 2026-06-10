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
LAYER = f"single_transformer_blocks.{BLOCK_INDEX}.attn.to_qkv_mlp_proj"
OUT_DIR = Path("reports/bonsai-lowbit-qkv-seq-head-layout-onnx")
HEAD_DIM_CANDIDATES = [128, 64]
SEQ_LEN = 4


class QkvSeqHeadLayoutProjection(torch.nn.Module):
    def __init__(self, proj: LowBitLinearRuntimeOnnx):
        super().__init__()
        self.proj = proj
        self.hidden_dim = int(proj.in_features)
        self.out_features = int(proj.out_features)
        self.q_size = self.hidden_dim
        self.k_size = self.hidden_dim
        self.v_size = self.hidden_dim
        self.mlp_size = self.out_features - (self.q_size + self.k_size + self.v_size)
        if self.mlp_size <= 0:
            raise ValueError(f"invalid qkv/mlp split: hidden_dim={self.hidden_dim} out_features={self.out_features}")
        self.head_dim = next((candidate for candidate in HEAD_DIM_CANDIDATES if self.hidden_dim % candidate == 0), None)
        if self.head_dim is None:
            raise ValueError(f"hidden_dim={self.hidden_dim} is not divisible by head_dim candidates {HEAD_DIM_CANDIDATES}")
        self.num_heads = self.hidden_dim // self.head_dim

    def _to_seq_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_mlp = self.proj(hidden)
        q, k, v, mlp = torch.split(qkv_mlp, [self.q_size, self.k_size, self.v_size, self.mlp_size], dim=-1)
        q_heads = self._to_seq_heads(q)
        k_heads = self._to_seq_heads(k)
        v_heads = self._to_seq_heads(v)
        return q_heads, k_heads, v_heads, mlp


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
    proj = load_runtime_linear(sd, LAYER)
    module = QkvSeqHeadLayoutProjection(proj).eval()

    hidden = torch.randn(2, SEQ_LEN, module.hidden_dim, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = [y.detach().cpu().numpy() for y in module(hidden)]

    onnx_path = OUT_DIR / "lowbit_qkv_seq_head_layout.onnx"
    output_names = ["q_seq_heads", "k_seq_heads", "v_seq_heads", "mlp"]
    torch.onnx.export(
        module,
        (hidden,),
        str(onnx_path),
        input_names=["hidden"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={
            "hidden": {0: "batch", 1: "seq"},
            "q_seq_heads": {0: "batch", 2: "seq"},
            "k_seq_heads": {0: "batch", 2: "seq"},
            "v_seq_heads": {0: "batch", 2: "seq"},
            "mlp": {0: "batch", 1: "seq"},
        },
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden": hidden.cpu().numpy().astype(np.float32)})
    expected_shapes = {
        "q_seq_heads": [int(hidden.shape[0]), module.num_heads, int(hidden.shape[1]), module.head_dim],
        "k_seq_heads": [int(hidden.shape[0]), module.num_heads, int(hidden.shape[1]), module.head_dim],
        "v_seq_heads": [int(hidden.shape[0]), module.num_heads, int(hidden.shape[1]), module.head_dim],
        "mlp": [int(hidden.shape[0]), int(hidden.shape[1]), module.mlp_size],
    }
    outputs = []
    allclose = True
    for name, ort_out, pt_out in zip(output_names, y_ort, y_pt):
        diff = ort_out - pt_out
        item = {
            "name": name,
            "output_shape": list(pt_out.shape),
            "expected_shape": expected_shapes[name],
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort_out, pt_out, rtol=1e-4, atol=1e-5)),
        }
        allclose = allclose and item["allclose_rtol_1e_4_atol_1e_5"] and item["output_shape"] == item["expected_shape"]
        outputs.append(item)

    external_data_path = Path(str(onnx_path) + ".data")
    packed_nbytes = tensor_nbytes(sd[f"{LAYER}.W_q"]) + tensor_nbytes(sd[f"{LAYER}.scales"]) + tensor_nbytes(sd[f"{LAYER}.zeros"])
    expanded_fp32_nbytes = proj.out_features * proj.in_features * 4
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    init_summary = initializer_summary(onnx_path)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "block_index": BLOCK_INDEX,
        "layer": LAYER,
        "graph_kind": "qkv_projection_split_and_sequence_head_layout",
        "is_attention": False,
        "is_real_transformer_block": False,
        "hidden_dim": module.hidden_dim,
        "projection_output_dim": module.out_features,
        "sequence_length": SEQ_LEN,
        "head_schema": {
            "method": "reshape_transpose_qkv_to_batch_heads_seq_head_dim",
            "layout": "batch_heads_seq_head_dim",
            "num_heads": module.num_heads,
            "head_dim": module.head_dim,
            "candidate_head_dims": HEAD_DIM_CANDIDATES,
        },
        "split_schema": {
            "method": "shape_derived_q_k_v_hidden_dim_remainder_mlp",
            "sizes": {
                "q": module.q_size,
                "k": module.k_size,
                "v": module.v_size,
                "mlp": module.mlp_size,
            },
            "sum": module.q_size + module.k_size + module.v_size + module.mlp_size,
        },
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_nbytes,
        "initializer_summary": init_summary,
        "input_shape": list(hidden.shape),
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "qkv_projection_sequence_head_layout_onnxruntime_cpu_verified_not_attention_or_transformer_block",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "graph_kind": report["graph_kind"],
        "block_index": report["block_index"],
        "hidden_dim": report["hidden_dim"],
        "projection_output_dim": report["projection_output_dim"],
        "sequence_length": report["sequence_length"],
        "head_schema": report["head_schema"],
        "split_schema": report["split_schema"],
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
