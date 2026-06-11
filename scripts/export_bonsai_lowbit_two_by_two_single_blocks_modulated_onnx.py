#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_two_single_blocks_modulated_onnx import (
    MODULATION_KEY,
    SEQ_LEN,
    TwoSingleBlocksModulated,
    build_core,
    lowbit_tensor_nbytes,
    qkv_prefix,
    to_out_prefix,
)

SEGMENTS = [(0, 1), (2, 3)]
OUT_DIR = Path("reports/bonsai-lowbit-two-by-two-single-blocks-modulated-onnx")


def export_segment(module: TwoSingleBlocksModulated, hidden: torch.Tensor, temb: torch.Tensor, path: Path) -> list[str]:
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
        str(path),
        input_names=["hidden", "temb"],
        output_names=output_names,
        opset_version=17,
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
    return output_names


def compare_outputs(prefix: str, names: list[str], ort_outputs: list[np.ndarray], pt_outputs: list[np.ndarray]) -> list[dict]:
    items = []
    for name, ort_out, pt_out in zip(names, ort_outputs, pt_outputs):
        diff = ort_out - pt_out
        items.append({
            "name": f"{prefix}_{name}",
            "output_shape": list(pt_out.shape),
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort_out, pt_out, rtol=1e-4, atol=1e-5)),
        })
    return items


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    modules = [TwoSingleBlocksModulated([build_core(sd, index) for index in segment]).eval() for segment in SEGMENTS]
    hidden_dim = int(modules[0].hidden_dim)
    semantic_input_width = int(modules[0].semantic_input_width)
    num_heads = int(modules[0].num_heads)
    head_dim = int(modules[0].head_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(280_000)
    hidden = torch.randn(2, SEQ_LEN, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, hidden_dim, generator=generator, dtype=torch.float32) * 0.01

    with torch.inference_mode():
        pt0 = [y.detach().cpu().numpy() for y in modules[0](hidden, temb)]
        pt_hidden_after_01 = torch.from_numpy(pt0[1]).to(dtype=hidden.dtype)
        pt1 = [y.detach().cpu().numpy() for y in modules[1](pt_hidden_after_01, temb)]

    with tempfile.TemporaryDirectory(prefix="bonsai-two-by-two-") as tmp:
        tmp_dir = Path(tmp)
        segment_paths = [tmp_dir / "segment_0_1.onnx", tmp_dir / "segment_2_3.onnx"]
        output_names = []
        for module, path in zip(modules, segment_paths):
            output_names.append(export_segment(module, hidden, temb, path))

        session0 = ort.InferenceSession(str(segment_paths[0]), providers=["CPUExecutionProvider"])
        ort0 = session0.run(None, {"hidden": hidden.cpu().numpy().astype(np.float32), "temb": temb.cpu().numpy().astype(np.float32)})
        session1 = ort.InferenceSession(str(segment_paths[1]), providers=["CPUExecutionProvider"])
        ort1 = session1.run(None, {"hidden": ort0[1].astype(np.float32), "temb": temb.cpu().numpy().astype(np.float32)})
        segment_sizes = [path_size(path) for path in segment_paths]

    outputs = []
    outputs.extend(compare_outputs("segment0_1", output_names[0], ort0, pt0))
    outputs.extend(compare_outputs("segment2_3", output_names[1], ort1, pt1))
    final_diff = ort1[1] - pt1[1]
    outputs.append({
        "name": "chained_final_block3_output",
        "output_shape": list(pt1[1].shape),
        "mean_abs_error": float(np.abs(final_diff).mean()),
        "max_abs_error": float(np.abs(final_diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort1[1], pt1[1], rtol=1e-4, atol=1e-5)),
    })
    allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in outputs)
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    packed_nbytes = sum(lowbit_tensor_nbytes(sd, qkv_prefix(index)) + lowbit_tensor_nbytes(sd, to_out_prefix(index)) for segment in SEGMENTS for index in segment)

    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "graph_kind": "two_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "is_two_by_two_single_blocks_modulated_stack": True,
        "block_indices": [0, 1, 2, 3],
        "segment_block_indices": [list(segment) for segment in SEGMENTS],
        "sequence_block_count": 4,
        "onnx_segment_count": 2,
        "is_attention_with_to_out": True,
        "to_out_connection_attempted": True,
        "has_modulation": True,
        "has_gate_residual": True,
        "is_single_monolithic_onnx": False,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "hidden_dim": hidden_dim,
        "sequence_length": SEQ_LEN,
        "semantic_input_width": semantic_input_width,
        "head_schema": {"method": "norm_weight_derived_head_dim_then_batch_heads_seq_head_dim", "layout": "batch_heads_seq_head_dim", "num_heads": num_heads, "head_dim": head_dim},
        "modulation_schema": {"method": "shared_single_stream_modulation_linear_temb_chunk_shift_scale_gate", "layer": MODULATION_KEY, "chunks": ["shift", "scale", "gate"], "input_width": hidden_dim, "output_width": 3 * hidden_dim},
        "residual_schema": {"method": "sequential_hidden_plus_gate_times_to_out_across_two_onnx_segments", "segment0_input": "hidden", "segment1_input": "segment0_block1_output", "final_output": "block3_output", "block_output_width": hidden_dim},
        "lowbit_layers": [{"block_index": index, "qkv_mlp_proj": qkv_prefix(index), "to_out": to_out_prefix(index)} for segment in SEGMENTS for index in segment],
        "lowbit_path": str(lowbit_path),
        "onnx_paths_persisted_in_reports": False,
        "onnx_segment_size_bytes": segment_sizes,
        "total_onnx_segment_size_bytes": sum(segment_sizes),
        "packed_nbytes": packed_nbytes,
        "input_shape": list(hidden.shape),
        "temb_shape": list(temb.shape),
        "outputs": outputs,
        "max_abs_error": max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": allclose,
        "claim": "two_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_kind": report["graph_kind"], "segment_block_indices": report["segment_block_indices"], "max_abs_error": report["max_abs_error"], "all_outputs_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"]}, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
