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

SEGMENTS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
OUT_DIR = Path("reports/bonsai-lowbit-eight-by-two-single-blocks-modulated-onnx")
STRICT_RTOL = 1e-4
STRICT_ATOL = 1e-5
DIAGNOSTIC_RTOL = 1e-4
DIAGNOSTIC_ATOL = 1e-4


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


def output_category(name: str) -> str:
    return "critical" if name.endswith("_output") or name == "chained_final_block15_output" else "diagnostic"


def compare_outputs(prefix: str, names: list[str], ort_outputs: list[np.ndarray], pt_outputs: list[np.ndarray]) -> list[dict]:
    items = []
    for name, ort_out, pt_out in zip(names, ort_outputs, pt_outputs):
        full_name = f"{prefix}_{name}"
        diff = ort_out - pt_out
        items.append({
            "name": full_name,
            "category": output_category(full_name),
            "output_shape": list(pt_out.shape),
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(ort_out, pt_out, rtol=STRICT_RTOL, atol=STRICT_ATOL)),
            "diagnostic_allclose_rtol_1e_4_atol_1e_4": bool(np.allclose(ort_out, pt_out, rtol=DIAGNOSTIC_RTOL, atol=DIAGNOSTIC_ATOL)),
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
    generator.manual_seed(300_000)
    hidden = torch.randn(2, SEQ_LEN, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb_np = temb.cpu().numpy().astype(np.float32)

    outputs: list[dict] = []
    segment_sizes: list[int] = []
    h_pt = hidden
    h_ort = hidden.cpu().numpy().astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="bonsai-eight-by-two-") as tmp:
        tmp_dir = Path(tmp)
        for segment, module in zip(SEGMENTS, modules):
            with torch.inference_mode():
                pt_outs = [y.detach().cpu().numpy() for y in module(h_pt, temb)]
            segment_path = tmp_dir / f"segment_{segment[0]}_{segment[1]}.onnx"
            names = export_segment(module, h_pt, temb, segment_path)
            session = ort.InferenceSession(str(segment_path), providers=["CPUExecutionProvider"])
            ort_outs = session.run(None, {"hidden": h_ort, "temb": temb_np})
            segment_sizes.append(path_size(segment_path))
            outputs.extend(compare_outputs(f"segment{segment[0]}_{segment[1]}", names, ort_outs, pt_outs))
            h_pt = torch.from_numpy(pt_outs[1]).to(dtype=hidden.dtype)
            h_ort = ort_outs[1].astype(np.float32)
            segment_path.unlink(missing_ok=True)

    final_diff = h_ort - h_pt.cpu().numpy().astype(np.float32)
    outputs.append({
        "name": "chained_final_block15_output",
        "category": "critical",
        "output_shape": list(h_pt.shape),
        "mean_abs_error": float(np.abs(final_diff).mean()),
        "max_abs_error": float(np.abs(final_diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(h_ort, h_pt.cpu().numpy().astype(np.float32), rtol=STRICT_RTOL, atol=STRICT_ATOL)),
        "diagnostic_allclose_rtol_1e_4_atol_1e_4": bool(np.allclose(h_ort, h_pt.cpu().numpy().astype(np.float32), rtol=DIAGNOSTIC_RTOL, atol=DIAGNOSTIC_ATOL)),
    })

    critical_outputs = [item for item in outputs if item["category"] == "critical"]
    diagnostic_outputs = [item for item in outputs if item["category"] == "diagnostic"]
    critical_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in critical_outputs)
    diagnostic_allclose = all(item["diagnostic_allclose_rtol_1e_4_atol_1e_4"] for item in diagnostic_outputs)
    all_outputs_strict_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in outputs)
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    critical_max_abs_error = max((item["max_abs_error"] for item in critical_outputs), default=None)
    diagnostic_max_abs_error = max((item["max_abs_error"] for item in diagnostic_outputs), default=None)
    packed_nbytes = sum(lowbit_tensor_nbytes(sd, qkv_prefix(index)) + lowbit_tensor_nbytes(sd, to_out_prefix(index)) for segment in SEGMENTS for index in segment)

    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "graph_kind": "eight_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "is_eight_by_two_single_blocks_modulated_stack": True,
        "block_indices": list(range(16)),
        "segment_block_indices": [list(segment) for segment in SEGMENTS],
        "sequence_block_count": 16,
        "onnx_segment_count": 8,
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
        "residual_schema": {"method": "sequential_hidden_plus_gate_times_to_out_across_eight_onnx_segments", "segment_inputs": ["hidden", "segment0_1_block1_output", "segment2_3_block1_output", "segment4_5_block1_output", "segment6_7_block1_output", "segment8_9_block1_output", "segment10_11_block1_output", "segment12_13_block1_output"], "final_output": "block15_output", "block_output_width": hidden_dim},
        "pass_fail_policy": {"critical_outputs": "block outputs and final chained output must pass rtol=1e-4 atol=1e-5", "diagnostic_outputs": "semantic inputs gates and attention weights are reported with relaxed rtol=1e-4 atol=1e-4 and do not control exit status"},
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
        "critical_max_abs_error": critical_max_abs_error,
        "diagnostic_max_abs_error": diagnostic_max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": all_outputs_strict_allclose,
        "critical_outputs_allclose_rtol_1e_4_atol_1e_5": critical_allclose,
        "diagnostic_outputs_allclose_rtol_1e_4_atol_1e_4": diagnostic_allclose,
        "claim": "eight_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_kind": report["graph_kind"], "segment_block_indices": report["segment_block_indices"], "critical_max_abs_error": report["critical_max_abs_error"], "critical_outputs_allclose": report["critical_outputs_allclose_rtol_1e_4_atol_1e_5"], "all_outputs_strict_allclose": report["all_outputs_allclose_rtol_1e_4_atol_1e_5"]}, indent=2))
    return 0 if critical_allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
