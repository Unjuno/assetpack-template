#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_runtime_linear_onnx import LowBitLinearRuntimeOnnx, initializer_summary, tensor_nbytes

BLOCK_INDEX = 0
QKV_MLP_PROJ = f"single_transformer_blocks.{BLOCK_INDEX}.attn.to_qkv_mlp_proj"
TO_OUT = f"single_transformer_blocks.{BLOCK_INDEX}.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-attention-to-out-input-source-onnx")
HEAD_DIM_CANDIDATES = [128, 64]
SEQ_LEN = 4


class AttentionToOutInputSourceCandidate(torch.nn.Module):
    def __init__(self, proj: LowBitLinearRuntimeOnnx, to_out: LowBitLinearRuntimeOnnx):
        super().__init__()
        self.proj = proj
        self.hidden_dim = int(proj.in_features)
        self.out_features = int(proj.out_features)
        self.to_out_in_features = int(to_out.in_features)
        self.to_out_out_features = int(to_out.out_features)
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
        self.context_flat_width = self.num_heads * self.head_dim
        self.missing_width = self.to_out_in_features - self.context_flat_width
        if self.missing_width <= 0:
            raise ValueError(f"to_out input is not wider than context: context={self.context_flat_width} to_out={self.to_out_in_features}")
        if self.mlp_size < self.missing_width:
            raise ValueError(f"mlp slice is too small for missing width: mlp={self.mlp_size} missing={self.missing_width}")
        self.candidate_width = self.context_flat_width + self.missing_width
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

    def _to_seq_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_mlp = self.proj(hidden)
        q, k, v, mlp = torch.split(qkv_mlp, [self.q_size, self.k_size, self.v_size, self.mlp_size], dim=-1)
        q_heads = self._to_seq_heads(q)
        k_heads = self._to_seq_heads(k)
        v_heads = self._to_seq_heads(v)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        context_heads = torch.matmul(weights, v_heads)
        context_seq_heads = context_heads.transpose(1, 2)
        context_flat = context_seq_heads.reshape(context_seq_heads.shape[0], context_seq_heads.shape[1], self.context_flat_width)
        mlp_prefix_candidate = mlp[..., : self.missing_width]
        mlp_remainder = mlp[..., self.missing_width :]
        to_out_input_candidate = torch.cat([context_flat, mlp_prefix_candidate], dim=-1)
        return context_flat, mlp_prefix_candidate, mlp_remainder, to_out_input_candidate, weights, scores


def load_runtime_linear(sd: dict, prefix: str) -> LowBitLinearRuntimeOnnx:
    wq = sd[f"{prefix}.W_q"]
    scales = sd[f"{prefix}.scales"]
    zeros = sd[f"{prefix}.zeros"]
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
    return LowBitLinearRuntimeOnnx(wq, scales, zeros, orig_shape, metadata)


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def lowbit_tensor_nbytes(sd: dict, prefix: str) -> int:
    return tensor_nbytes(sd[f"{prefix}.W_q"]) + tensor_nbytes(sd[f"{prefix}.scales"]) + tensor_nbytes(sd[f"{prefix}.zeros"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    proj = load_runtime_linear(sd, QKV_MLP_PROJ)
    to_out = load_runtime_linear(sd, TO_OUT)
    module = AttentionToOutInputSourceCandidate(proj, to_out).eval()

    hidden = torch.randn(2, SEQ_LEN, module.hidden_dim, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = [y.detach().cpu().numpy() for y in module(hidden)]

    onnx_path = OUT_DIR / "lowbit_attention_to_out_input_source_candidate.onnx"
    output_names = ["context_flat", "mlp_prefix_candidate", "mlp_remainder", "to_out_input_candidate", "weights", "scores"]
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
            "context_flat": {0: "batch", 1: "seq"},
            "mlp_prefix_candidate": {0: "batch", 1: "seq"},
            "mlp_remainder": {0: "batch", 1: "seq"},
            "to_out_input_candidate": {0: "batch", 1: "seq"},
            "weights": {0: "batch", 2: "query_seq", 3: "key_seq"},
            "scores": {0: "batch", 2: "query_seq", 3: "key_seq"},
        },
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden": hidden.cpu().numpy().astype(np.float32)})
    bsz = int(hidden.shape[0])
    seq = int(hidden.shape[1])
    expected_shapes = {
        "context_flat": [bsz, seq, module.context_flat_width],
        "mlp_prefix_candidate": [bsz, seq, module.missing_width],
        "mlp_remainder": [bsz, seq, module.mlp_size - module.missing_width],
        "to_out_input_candidate": [bsz, seq, module.candidate_width],
        "weights": [bsz, module.num_heads, seq, seq],
        "scores": [bsz, module.num_heads, seq, seq],
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
    packed_nbytes = lowbit_tensor_nbytes(sd, QKV_MLP_PROJ)
    inspected_to_out_packed_nbytes = lowbit_tensor_nbytes(sd, TO_OUT)
    expanded_fp32_nbytes = proj.out_features * proj.in_features * 4
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "block_index": BLOCK_INDEX,
        "layer": QKV_MLP_PROJ,
        "inspected_to_out_layer": TO_OUT,
        "graph_kind": "qkv_projection_attention_to_out_input_source_candidate",
        "is_attention_math": True,
        "is_attention_to_out_input_source_probe": True,
        "is_attention_with_to_out": False,
        "to_out_connection_attempted": False,
        "is_real_transformer_block": False,
        "semantic_to_out_input_verified": False,
        "hidden_dim": module.hidden_dim,
        "projection_output_dim": module.out_features,
        "sequence_length": SEQ_LEN,
        "head_schema": {"method": "reshape_transpose_qkv_to_batch_heads_seq_head_dim", "layout": "batch_heads_seq_head_dim", "num_heads": module.num_heads, "head_dim": module.head_dim, "scale": module.scale, "candidate_head_dims": HEAD_DIM_CANDIDATES},
        "split_schema": {"method": "shape_derived_q_k_v_hidden_dim_remainder_mlp", "sizes": {"q": module.q_size, "k": module.k_size, "v": module.v_size, "mlp": module.mlp_size}, "sum": module.q_size + module.k_size + module.v_size + module.mlp_size},
        "source_candidate_schema": {"method": "concat_context_flat_with_mlp_prefix_candidate_width_only", "context_flat_width": module.context_flat_width, "mlp_size": module.mlp_size, "mlp_prefix_candidate_width": module.missing_width, "mlp_remainder_width": module.mlp_size - module.missing_width, "candidate_width": module.candidate_width, "to_out_expected_in_features": module.to_out_in_features, "candidate_width_matches_to_out": module.candidate_width == module.to_out_in_features, "semantic_verified": False, "reason": "This verifies an ONNX width/source candidate only. It does not prove Bonsai semantically feeds this exact concat to to_out."},
        "to_out_input_schema": {"layer": TO_OUT, "expected_in_features": module.to_out_in_features, "out_features": module.to_out_out_features, "candidate_context_width": module.context_flat_width, "missing_width": module.missing_width, "candidate_input_width": module.candidate_width, "connection_attempted": False},
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "packed_nbytes": packed_nbytes,
        "inspected_to_out_packed_nbytes": inspected_to_out_packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_nbytes,
        "initializer_summary": initializer_summary(onnx_path),
        "input_shape": list(hidden.shape),
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "attention_to_out_input_source_width_candidate_onnxruntime_cpu_verified_not_semantic_to_out_or_transformer_block",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_kind": report["graph_kind"], "source_candidate_schema": report["source_candidate_schema"], "to_out_input_schema": report["to_out_input_schema"], "max_abs_error": report["max_abs_error"], "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"]}, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
