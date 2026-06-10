#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_runtime_linear_onnx import LowBitLinearRuntimeOnnx, initializer_summary, tensor_nbytes

BLOCK_INDEX = 0
BLOCK_PREFIX = f"single_transformer_blocks.{BLOCK_INDEX}"
QKV_MLP_PROJ = f"{BLOCK_PREFIX}.attn.to_qkv_mlp_proj"
TO_OUT = f"{BLOCK_PREFIX}.attn.to_out"
NORM_Q_KEY = f"{BLOCK_PREFIX}.attn.norm_q.weight"
NORM_K_KEY = f"{BLOCK_PREFIX}.attn.norm_k.weight"
MODULATION_KEY = "single_stream_modulation.linear.weight"
OUT_DIR = Path("reports/bonsai-lowbit-single-block-modulated-onnx")
SEQ_LEN = 4


class SingleBlockModulatedCore(torch.nn.Module):
    def __init__(self, proj: LowBitLinearRuntimeOnnx, to_out: LowBitLinearRuntimeOnnx, norm_q: torch.Tensor, norm_k: torch.Tensor, modulation_weight: torch.Tensor):
        super().__init__()
        self.proj = proj
        self.to_out = to_out
        self.hidden_dim = int(proj.in_features)
        self.out_features = int(proj.out_features)
        self.to_out_in_features = int(to_out.in_features)
        self.to_out_out_features = int(to_out.out_features)
        self.q_size = self.hidden_dim
        self.k_size = self.hidden_dim
        self.v_size = self.hidden_dim
        self.mlp_size = self.out_features - (self.q_size + self.k_size + self.v_size)
        if self.mlp_size <= 0 or self.mlp_size % 2 != 0:
            raise ValueError(f"invalid mlp_size={self.mlp_size}")
        self.head_dim = int(norm_q.numel())
        if int(norm_k.numel()) != self.head_dim:
            raise ValueError(f"norm_q/norm_k mismatch: {norm_q.numel()} vs {norm_k.numel()}")
        if self.hidden_dim % self.head_dim != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} is not divisible by head_dim={self.head_dim}")
        if list(modulation_weight.shape) != [3 * self.hidden_dim, self.hidden_dim]:
            raise ValueError(f"unexpected modulation weight shape: {list(modulation_weight.shape)}")
        self.num_heads = self.hidden_dim // self.head_dim
        self.context_flat_width = self.hidden_dim
        self.mlp_swiglu_width = self.mlp_size // 2
        self.semantic_input_width = self.context_flat_width + self.mlp_swiglu_width
        if self.semantic_input_width != self.to_out_in_features:
            raise ValueError(
                f"semantic input width mismatch: context={self.context_flat_width} swiglu={self.mlp_swiglu_width} "
                f"candidate={self.semantic_input_width} to_out={self.to_out_in_features}"
            )
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.register_buffer("norm_q", norm_q.detach().cpu().to(torch.float32).contiguous(), persistent=True)
        self.register_buffer("norm_k", norm_k.detach().cpu().to(torch.float32).contiguous(), persistent=True)
        self.register_buffer("modulation_weight", modulation_weight.detach().cpu().to(torch.float32).contiguous(), persistent=True)

    def _rms_norm_per_head(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (y * weight.view(1, 1, 1, -1)).to(dtype=x.dtype)

    def _layer_norm_modulated(self, hidden: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = F.layer_norm(hidden.to(torch.float32), (hidden.shape[-1],), eps=eps)
        y = y * (1.0 + scale.to(torch.float32).unsqueeze(1)) + shift.to(torch.float32).unsqueeze(1)
        return y.to(dtype=hidden.dtype)

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = F.linear(temb, self.modulation_weight.to(dtype=temb.dtype)).chunk(3, dim=-1)
        hidden_in = self._layer_norm_modulated(hidden, shift, scale)
        qkv_mlp = self.proj(hidden_in)
        q, k, v, mlp = torch.split(qkv_mlp, [self.q_size, self.k_size, self.v_size, self.mlp_size], dim=-1)
        q = q.reshape(hidden.shape[0], hidden.shape[1], self.num_heads, self.head_dim)
        k = k.reshape(hidden.shape[0], hidden.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(hidden.shape[0], hidden.shape[1], self.num_heads, self.head_dim)
        q = self._rms_norm_per_head(q, self.norm_q)
        k = self._rms_norm_per_head(k, self.norm_k)
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        v_heads = v.transpose(1, 2)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        context_heads = torch.matmul(weights, v_heads)
        context_flat = context_heads.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], self.context_flat_width)
        a, b = torch.chunk(mlp, 2, dim=-1)
        mlp_swiglu = F.silu(a) * b
        semantic_to_out_input = torch.cat([context_flat, mlp_swiglu], dim=-1)
        to_out_output = self.to_out(semantic_to_out_input)
        gated_to_out = gate.unsqueeze(1) * to_out_output
        block_output = hidden + gated_to_out
        return hidden_in, semantic_to_out_input, to_out_output, gated_to_out, block_output, gate, weights, scores


def load_runtime_linear(sd: dict, prefix: str) -> LowBitLinearRuntimeOnnx:
    return LowBitLinearRuntimeOnnx(
        sd[f"{prefix}.W_q"],
        sd[f"{prefix}.scales"],
        sd[f"{prefix}.zeros"],
        [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()],
        [int(v) for v in sd[f"{prefix}.metadata"].tolist()],
    )


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def lowbit_tensor_nbytes(sd: dict, prefix: str) -> int:
    return tensor_nbytes(sd[f"{prefix}.W_q"]) + tensor_nbytes(sd[f"{prefix}.scales"]) + tensor_nbytes(sd[f"{prefix}.zeros"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    module = SingleBlockModulatedCore(
        load_runtime_linear(sd, QKV_MLP_PROJ),
        load_runtime_linear(sd, TO_OUT),
        sd[NORM_Q_KEY],
        sd[NORM_K_KEY],
        sd[MODULATION_KEY],
    ).eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(250_000)
    hidden = torch.randn(2, SEQ_LEN, module.hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, module.hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    with torch.inference_mode():
        y_pt = [y.detach().cpu().numpy() for y in module(hidden, temb)]

    onnx_path = OUT_DIR / "lowbit_single_block_modulated.onnx"
    output_names = ["hidden_in", "semantic_to_out_input", "to_out_output", "gated_to_out", "block_output", "gate", "weights", "scores"]
    torch.onnx.export(
        module,
        (hidden, temb),
        str(onnx_path),
        input_names=["hidden", "temb"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={
            "hidden": {0: "batch", 1: "seq"},
            "temb": {0: "batch"},
            "hidden_in": {0: "batch", 1: "seq"},
            "semantic_to_out_input": {0: "batch", 1: "seq"},
            "to_out_output": {0: "batch", 1: "seq"},
            "gated_to_out": {0: "batch", 1: "seq"},
            "block_output": {0: "batch", 1: "seq"},
            "gate": {0: "batch"},
            "weights": {0: "batch", 2: "query_seq", 3: "key_seq"},
            "scores": {0: "batch", 2: "query_seq", 3: "key_seq"},
        },
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden": hidden.cpu().numpy().astype(np.float32), "temb": temb.cpu().numpy().astype(np.float32)})
    bsz = int(hidden.shape[0])
    seq = int(hidden.shape[1])
    expected_shapes = {
        "hidden_in": [bsz, seq, module.hidden_dim],
        "semantic_to_out_input": [bsz, seq, module.semantic_input_width],
        "to_out_output": [bsz, seq, module.to_out_out_features],
        "gated_to_out": [bsz, seq, module.to_out_out_features],
        "block_output": [bsz, seq, module.hidden_dim],
        "gate": [bsz, module.hidden_dim],
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
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "block_index": BLOCK_INDEX,
        "layer": QKV_MLP_PROJ,
        "to_out_layer": TO_OUT,
        "modulation_layer": MODULATION_KEY,
        "graph_kind": "single_block_modulated_attention_to_out_residual",
        "is_single_block_modulated_core": True,
        "is_attention_with_to_out": True,
        "to_out_connection_attempted": True,
        "has_modulation": True,
        "has_gate_residual": True,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "hidden_dim": module.hidden_dim,
        "projection_output_dim": module.out_features,
        "sequence_length": SEQ_LEN,
        "head_schema": {"method": "norm_weight_derived_head_dim_then_batch_heads_seq_head_dim", "layout": "batch_heads_seq_head_dim", "num_heads": module.num_heads, "head_dim": module.head_dim, "scale": module.scale, "norm_q_key": NORM_Q_KEY, "norm_k_key": NORM_K_KEY},
        "split_schema": {"method": "shape_derived_q_k_v_hidden_dim_remainder_mlp", "sizes": {"q": module.q_size, "k": module.k_size, "v": module.v_size, "mlp": module.mlp_size}, "sum": module.q_size + module.k_size + module.v_size + module.mlp_size},
        "modulation_schema": {"method": "linear_temb_chunk_shift_scale_gate_then_layer_norm_modulation", "layer": MODULATION_KEY, "chunks": ["shift", "scale", "gate"], "input_width": module.hidden_dim, "output_width": 3 * module.hidden_dim},
        "semantic_input_schema": {"method": "concat_normed_attention_context_with_swiglu_mlp", "context_flat_width": module.context_flat_width, "mlp_size": module.mlp_size, "mlp_activation": "swiglu", "mlp_swiglu_width": module.mlp_swiglu_width, "semantic_input_width": module.semantic_input_width, "to_out_expected_in_features": module.to_out_in_features, "semantic_width_matches_to_out": module.semantic_input_width == module.to_out_in_features},
        "residual_schema": {"method": "hidden_plus_gate_times_to_out", "gate_width": module.hidden_dim, "to_out_output_width": module.to_out_out_features, "block_output_width": module.hidden_dim},
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "packed_nbytes": lowbit_tensor_nbytes(sd, QKV_MLP_PROJ),
        "to_out_packed_nbytes": lowbit_tensor_nbytes(sd, TO_OUT),
        "expanded_fp32_weight_nbytes": module.out_features * module.hidden_dim * 4,
        "initializer_summary": initializer_summary(onnx_path),
        "input_shape": list(hidden.shape),
        "temb_shape": list(temb.shape),
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "single_block_modulated_attention_to_out_residual_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_kind": report["graph_kind"], "modulation_schema": report["modulation_schema"], "residual_schema": report["residual_schema"], "max_abs_error": report["max_abs_error"], "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"]}, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
