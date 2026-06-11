#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_runtime_linear_onnx import initializer_summary, tensor_nbytes
from export_bonsai_lowbit_single_block_modulated_onnx import SingleBlockModulatedCore, load_runtime_linear

BLOCK_INDICES = [0, 1]
MODULATION_KEY = "single_stream_modulation.linear.weight"
OUT_DIR = Path("reports/bonsai-lowbit-two-single-blocks-modulated-onnx")
SEQ_LEN = 4
ONNX_OPSET_VERSION = 18


def qkv_prefix(index: int) -> str:
    return f"single_transformer_blocks.{index}.attn.to_qkv_mlp_proj"


def to_out_prefix(index: int) -> str:
    return f"single_transformer_blocks.{index}.attn.to_out"


def norm_q_key(index: int) -> str:
    return f"single_transformer_blocks.{index}.attn.norm_q.weight"


def norm_k_key(index: int) -> str:
    return f"single_transformer_blocks.{index}.attn.norm_k.weight"


class TwoSingleBlocksModulated(torch.nn.Module):
    def __init__(self, cores: list[SingleBlockModulatedCore]):
        super().__init__()
        self.cores = torch.nn.ModuleList(cores)
        self.hidden_dim = int(cores[0].hidden_dim)
        self.semantic_input_width = int(cores[0].semantic_input_width)
        self.num_heads = int(cores[0].num_heads)
        self.head_dim = int(cores[0].head_dim)
        self.sequence_block_count = len(cores)
        for core in cores:
            if int(core.hidden_dim) != self.hidden_dim:
                raise ValueError("hidden_dim mismatch across cores")
            if int(core.semantic_input_width) != self.semantic_input_width:
                raise ValueError("semantic input width mismatch across cores")

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Each core returns:
        # hidden_in, semantic_to_out_input, to_out_output, gated_to_out, block_output, gate, weights, scores
        out0 = self.cores[0](hidden, temb)
        block0_output = out0[4]
        out1 = self.cores[1](block0_output, temb)
        block1_output = out1[4]
        return block0_output, block1_output, out0[1], out1[1], out0[5], out1[5], out0[6], out1[6]


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def lowbit_tensor_nbytes(sd: dict, prefix: str) -> int:
    return tensor_nbytes(sd[f"{prefix}.W_q"]) + tensor_nbytes(sd[f"{prefix}.scales"]) + tensor_nbytes(sd[f"{prefix}.zeros"])


def build_core(sd: dict, index: int) -> SingleBlockModulatedCore:
    return SingleBlockModulatedCore(
        load_runtime_linear(sd, qkv_prefix(index)),
        load_runtime_linear(sd, to_out_prefix(index)),
        sd[norm_q_key(index)],
        sd[norm_k_key(index)],
        sd[MODULATION_KEY],
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    module = TwoSingleBlocksModulated([build_core(sd, index) for index in BLOCK_INDICES]).eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(260_000)
    hidden = torch.randn(2, SEQ_LEN, module.hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, module.hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    with torch.inference_mode():
        y_pt = [y.detach().cpu().numpy() for y in module(hidden, temb)]

    onnx_path = OUT_DIR / "lowbit_two_single_blocks_modulated.onnx"
    output_names = [
        "block0_output",
        "block1_output",
        "block0_semantic_to_out_input",
        "block1_semantic_to_out_input",
        "block0_gate",
        "block1_gate",
        "block0_weights",
        "block1_weights",
    ]
    torch.onnx.export(
        module,
        (hidden, temb),
        str(onnx_path),
        input_names=["hidden", "temb"],
        output_names=output_names,
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=False,
        dynamic_axes={
            "hidden": {0: "batch", 1: "seq"},
            "temb": {0: "batch"},
            "block0_output": {0: "batch", 1: "seq"},
            "block1_output": {0: "batch", 1: "seq"},
            "block0_semantic_to_out_input": {0: "batch", 1: "seq"},
            "block1_semantic_to_out_input": {0: "batch", 1: "seq"},
            "block0_gate": {0: "batch"},
            "block1_gate": {0: "batch"},
            "block0_weights": {0: "batch", 2: "query_seq", 3: "key_seq"},
            "block1_weights": {0: "batch", 2: "query_seq", 3: "key_seq"},
        },
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden": hidden.cpu().numpy().astype(np.float32), "temb": temb.cpu().numpy().astype(np.float32)})
    bsz = int(hidden.shape[0])
    seq = int(hidden.shape[1])
    expected_shapes = {
        "block0_output": [bsz, seq, module.hidden_dim],
        "block1_output": [bsz, seq, module.hidden_dim],
        "block0_semantic_to_out_input": [bsz, seq, module.semantic_input_width],
        "block1_semantic_to_out_input": [bsz, seq, module.semantic_input_width],
        "block0_gate": [bsz, module.hidden_dim],
        "block1_gate": [bsz, module.hidden_dim],
        "block0_weights": [bsz, module.num_heads, seq, seq],
        "block1_weights": [bsz, module.num_heads, seq, seq],
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
    packed_nbytes = sum(lowbit_tensor_nbytes(sd, qkv_prefix(index)) + lowbit_tensor_nbytes(sd, to_out_prefix(index)) for index in BLOCK_INDICES)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "graph_kind": "two_single_blocks_modulated_attention_to_out_residual_stack",
        "is_two_single_blocks_modulated_stack": True,
        "block_indices": BLOCK_INDICES,
        "sequence_block_count": module.sequence_block_count,
        "onnx_opset_version": ONNX_OPSET_VERSION,
        "is_attention_with_to_out": True,
        "to_out_connection_attempted": True,
        "has_modulation": True,
        "has_gate_residual": True,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "hidden_dim": module.hidden_dim,
        "sequence_length": SEQ_LEN,
        "semantic_input_width": module.semantic_input_width,
        "head_schema": {"method": "norm_weight_derived_head_dim_then_batch_heads_seq_head_dim", "layout": "batch_heads_seq_head_dim", "num_heads": module.num_heads, "head_dim": module.head_dim},
        "modulation_schema": {"method": "shared_single_stream_modulation_linear_temb_chunk_shift_scale_gate", "layer": MODULATION_KEY, "chunks": ["shift", "scale", "gate"], "input_width": module.hidden_dim, "output_width": 3 * module.hidden_dim},
        "residual_schema": {"method": "sequential_hidden_plus_gate_times_to_out", "block0_input": "hidden", "block1_input": "block0_output", "block_output_width": module.hidden_dim},
        "lowbit_layers": [{"block_index": index, "qkv_mlp_proj": qkv_prefix(index), "to_out": to_out_prefix(index)} for index in BLOCK_INDICES],
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "packed_nbytes": packed_nbytes,
        "initializer_summary": initializer_summary(onnx_path),
        "input_shape": list(hidden.shape),
        "temb_shape": list(temb.shape),
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_kind": report["graph_kind"], "block_indices": report["block_indices"], "onnx_opset_version": report["onnx_opset_version"], "residual_schema": report["residual_schema"], "max_abs_error": report["max_abs_error"], "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"]}, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
