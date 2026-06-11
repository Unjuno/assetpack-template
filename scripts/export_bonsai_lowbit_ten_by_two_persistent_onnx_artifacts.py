#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_ten_by_two_single_blocks_modulated_onnx import (
    DIAGNOSTIC_ATOL,
    DIAGNOSTIC_RTOL,
    SEGMENTS,
    STRICT_ATOL,
    STRICT_RTOL,
    compare_outputs,
    export_segment,
)
from export_bonsai_lowbit_two_single_blocks_modulated_onnx import (
    MODULATION_KEY,
    SEQ_LEN,
    TwoSingleBlocksModulated,
    build_core,
    lowbit_tensor_nbytes,
    qkv_prefix,
    to_out_prefix,
)

OUT_DIR = Path("reports/bonsai-lowbit-ten-by-two-persistent-onnx-artifacts")
ONNX_DIR = OUT_DIR / "onnx"
REFERENCE_PATH = OUT_DIR / "reference.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(OUT_DIR).as_posix()


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    modules = [TwoSingleBlocksModulated([build_core(sd, index) for index in segment]).eval() for segment in SEGMENTS]
    hidden_dim = int(modules[0].hidden_dim)
    semantic_input_width = int(modules[0].semantic_input_width)
    num_heads = int(modules[0].num_heads)
    head_dim = int(modules[0].head_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(310_000)
    hidden = torch.randn(2, SEQ_LEN, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb_np = temb.cpu().numpy().astype(np.float32)

    outputs: list[dict] = []
    reference_arrays: dict[str, np.ndarray] = {
        "initial_hidden": hidden.cpu().numpy().astype(np.float32),
        "temb": temb_np,
    }
    segment_files: list[dict] = []
    h_pt = hidden
    h_ort = reference_arrays["initial_hidden"]

    for segment, module in zip(SEGMENTS, modules):
        with torch.inference_mode():
            pt_outs = [y.detach().cpu().numpy().astype(np.float32) for y in module(h_pt, temb)]

        segment_path = ONNX_DIR / f"segment_{segment[0]}_{segment[1]}.onnx"
        output_names = export_segment(module, h_pt, temb, segment_path)
        segment_size = segment_path.stat().st_size
        segment_sha256 = sha256_file(segment_path)

        session = ort.InferenceSession(str(segment_path), providers=["CPUExecutionProvider"])
        ort_outs = session.run(None, {"hidden": h_ort, "temb": temb_np})
        ort_outs = [out.astype(np.float32) for out in ort_outs]

        prefix = f"segment{segment[0]}_{segment[1]}"
        outputs.extend(compare_outputs(prefix, output_names, ort_outs, pt_outs))

        reference_arrays[f"{prefix}_block0_output"] = pt_outs[0]
        reference_arrays[f"{prefix}_block1_output"] = pt_outs[1]

        segment_files.append({
            "segment_block_indices": list(segment),
            "path": rel(segment_path),
            "sha256": segment_sha256,
            "size_bytes": segment_size,
            "output_names": output_names,
        })

        h_pt = torch.from_numpy(pt_outs[1]).to(dtype=hidden.dtype)
        h_ort = ort_outs[1]

    h_pt_np = h_pt.cpu().numpy().astype(np.float32)
    final_diff = h_ort - h_pt_np
    outputs.append({
        "name": "chained_final_block19_output",
        "category": "critical",
        "output_shape": list(h_pt.shape),
        "mean_abs_error": float(np.abs(final_diff).mean()),
        "max_abs_error": float(np.abs(final_diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(h_ort, h_pt_np, rtol=STRICT_RTOL, atol=STRICT_ATOL)),
        "diagnostic_allclose_rtol_1e_4_atol_1e_4": bool(np.allclose(h_ort, h_pt_np, rtol=DIAGNOSTIC_RTOL, atol=DIAGNOSTIC_ATOL)),
    })
    reference_arrays["expected_final_block19_output"] = h_pt_np

    np.savez_compressed(REFERENCE_PATH, **reference_arrays)

    critical_outputs = [item for item in outputs if item["category"] == "critical"]
    diagnostic_outputs = [item for item in outputs if item["category"] == "diagnostic"]
    critical_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in critical_outputs)
    diagnostic_allclose = all(item["diagnostic_allclose_rtol_1e_4_atol_1e_4"] for item in diagnostic_outputs)
    all_outputs_strict_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in outputs)
    max_abs_error = max((item["max_abs_error"] for item in outputs), default=None)
    critical_max_abs_error = max((item["max_abs_error"] for item in critical_outputs), default=None)
    diagnostic_max_abs_error = max((item["max_abs_error"] for item in diagnostic_outputs), default=None)
    packed_nbytes = sum(
        lowbit_tensor_nbytes(sd, qkv_prefix(index)) + lowbit_tensor_nbytes(sd, to_out_prefix(index))
        for segment in SEGMENTS
        for index in segment
    )

    report = {
        "artifact_kind": "persistent_onnx_segment_bundle",
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "is_ten_by_two_single_blocks_modulated_stack": True,
        "block_indices": list(range(20)),
        "segment_block_indices": [list(segment) for segment in SEGMENTS],
        "sequence_block_count": 20,
        "onnx_segment_count": 10,
        "persistent_onnx_artifacts": True,
        "reusable_onnx_chain_artifact": True,
        "onnx_paths_persisted_in_reports": True,
        "onnx_segment_files": segment_files,
        "reference_path": rel(REFERENCE_PATH),
        "reference_sha256": sha256_file(REFERENCE_PATH),
        "reference_arrays": sorted(reference_arrays.keys()),
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
        "head_schema": {
            "method": "norm_weight_derived_head_dim_then_batch_heads_seq_head_dim",
            "layout": "batch_heads_seq_head_dim",
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "modulation_schema": {
            "method": "shared_single_stream_modulation_linear_temb_chunk_shift_scale_gate",
            "layer": MODULATION_KEY,
            "chunks": ["shift", "scale", "gate"],
            "input_width": hidden_dim,
            "output_width": 3 * hidden_dim,
        },
        "residual_schema": {
            "method": "sequential_hidden_plus_gate_times_to_out_across_ten_persisted_onnx_segments",
            "segment_inputs": ["hidden"] + [f"segment{a}_{b}_block1_output" for a, b in SEGMENTS[:-1]],
            "final_output": "block19_output",
            "block_output_width": hidden_dim,
        },
        "pass_fail_policy": {
            "critical_outputs": "block outputs and final chained output must pass rtol=1e-4 atol=1e-5",
            "diagnostic_outputs": "semantic inputs gates and attention weights are reported with relaxed rtol=1e-4 atol=1e-4 and do not control exit status",
        },
        "lowbit_layers": [
            {"block_index": index, "qkv_mlp_proj": qkv_prefix(index), "to_out": to_out_prefix(index)}
            for segment in SEGMENTS
            for index in segment
        ],
        "lowbit_path": str(lowbit_path),
        "onnx_segment_size_bytes": [item["size_bytes"] for item in segment_files],
        "total_onnx_segment_size_bytes": sum(item["size_bytes"] for item in segment_files),
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
        "claim": "ten_by_two_persistent_onnx_segments_reusable_chain_artifact_exported_and_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "graph_kind": report["graph_kind"],
        "artifact_kind": report["artifact_kind"],
        "onnx_segment_count": report["onnx_segment_count"],
        "persistent_onnx_artifacts": report["persistent_onnx_artifacts"],
        "critical_max_abs_error": report["critical_max_abs_error"],
        "critical_outputs_allclose": report["critical_outputs_allclose_rtol_1e_4_atol_1e_5"],
    }, indent=2))
    return 0 if critical_allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
