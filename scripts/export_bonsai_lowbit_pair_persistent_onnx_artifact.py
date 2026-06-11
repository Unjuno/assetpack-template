#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

ROOT = Path("reports")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one persistent two-block Bonsai low-bit ONNX segment artifact.")
    parser.add_argument("--start", type=int, required=True, help="Even block index; segment is start,start+1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = int(args.start)
    if start < 0 or start % 2 != 0 or start + 1 >= 20:
        raise ValueError(f"--start must be an even index in [0, 18], got {start}")
    segment = (start, start + 1)
    out_dir = ROOT / f"bonsai-lowbit-persistent-onnx-segment-{segment[0]}-{segment[1]}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    module = TwoSingleBlocksModulated([build_core(sd, index) for index in segment]).eval()
    hidden_dim = int(module.hidden_dim)
    semantic_input_width = int(module.semantic_input_width)
    num_heads = int(module.num_heads)
    head_dim = int(module.head_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(400_000 + start)
    hidden = torch.randn(2, SEQ_LEN, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(2, hidden_dim, generator=generator, dtype=torch.float32) * 0.01

    with torch.inference_mode():
        pt_outputs = [y.detach().cpu().numpy().astype(np.float32) for y in module(hidden, temb)]

    onnx_path = out_dir / f"segment_{segment[0]}_{segment[1]}.onnx"
    output_names = export_segment(module, hidden, temb, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = [out.astype(np.float32) for out in session.run(None, {
        "hidden": hidden.cpu().numpy().astype(np.float32),
        "temb": temb.cpu().numpy().astype(np.float32),
    })]
    outputs = compare_outputs(f"segment{segment[0]}_{segment[1]}", output_names, ort_outputs, pt_outputs)

    critical_outputs = [item for item in outputs if item["category"] == "critical"]
    diagnostic_outputs = [item for item in outputs if item["category"] == "diagnostic"]
    critical_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in critical_outputs)
    diagnostic_allclose = all(item["diagnostic_allclose_rtol_1e_4_atol_1e_4"] for item in diagnostic_outputs)
    all_outputs_strict_allclose = all(item["allclose_rtol_1e_4_atol_1e_5"] for item in outputs)
    critical_max_abs_error = max((item["max_abs_error"] for item in critical_outputs), default=None)
    diagnostic_max_abs_error = max((item["max_abs_error"] for item in diagnostic_outputs), default=None)
    packed_nbytes = sum(lowbit_tensor_nbytes(sd, qkv_prefix(index)) + lowbit_tensor_nbytes(sd, to_out_prefix(index)) for index in segment)

    report = {
        "artifact_kind": "persistent_onnx_segment",
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "graph_kind": "pair_segment_single_blocks_modulated_attention_to_out_residual_stack",
        "block_indices": list(segment),
        "segment_block_indices": [list(segment)],
        "sequence_block_count": 2,
        "onnx_segment_count": 1,
        "persistent_onnx_artifacts": True,
        "reusable_onnx_segment_artifact": True,
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
        "lowbit_layers": [{"block_index": index, "qkv_mlp_proj": qkv_prefix(index), "to_out": to_out_prefix(index)} for index in segment],
        "lowbit_path": str(lowbit_path),
        "onnx_path": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "packed_nbytes": packed_nbytes,
        "input_shape": list(hidden.shape),
        "temb_shape": list(temb.shape),
        "outputs": outputs,
        "critical_max_abs_error": critical_max_abs_error,
        "diagnostic_max_abs_error": diagnostic_max_abs_error,
        "all_outputs_allclose_rtol_1e_4_atol_1e_5": all_outputs_strict_allclose,
        "critical_outputs_allclose_rtol_1e_4_atol_1e_5": critical_allclose,
        "diagnostic_outputs_allclose_rtol_1e_4_atol_1e_4": diagnostic_allclose,
        "claim": "pair_segment_persistent_onnx_artifact_exported_and_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_kind": report["artifact_kind"],
        "block_indices": report["block_indices"],
        "onnx_sha256": report["onnx_sha256"],
        "critical_outputs_allclose": report["critical_outputs_allclose_rtol_1e_4_atol_1e_5"],
    }, indent=2))
    return 0 if critical_allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
