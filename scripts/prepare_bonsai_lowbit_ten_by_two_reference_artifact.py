#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_ten_by_two_single_blocks_modulated_onnx import SEGMENTS
from export_bonsai_lowbit_two_single_blocks_modulated_onnx import (
    MODULATION_KEY,
    SEQ_LEN,
    TwoSingleBlocksModulated,
    build_core,
)

OUT_DIR = Path("reports/bonsai-lowbit-ten-by-two-persistent-reference")
REFERENCE_PATH = OUT_DIR / "reference.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    arrays: dict[str, np.ndarray] = {
        "initial_hidden": hidden.cpu().numpy().astype(np.float32),
        "temb": temb.cpu().numpy().astype(np.float32),
    }
    h_pt = hidden
    for segment, module in zip(SEGMENTS, modules):
        with torch.inference_mode():
            pt_outs = [y.detach().cpu().numpy().astype(np.float32) for y in module(h_pt, temb)]
        prefix = f"segment{segment[0]}_{segment[1]}"
        arrays[f"{prefix}_block0_output"] = pt_outs[0]
        arrays[f"{prefix}_block1_output"] = pt_outs[1]
        h_pt = torch.from_numpy(pt_outs[1]).to(dtype=hidden.dtype)

    arrays["expected_final_block19_output"] = h_pt.cpu().numpy().astype(np.float32)
    np.savez_compressed(REFERENCE_PATH, **arrays)

    report = {
        "artifact_kind": "persistent_onnx_chain_reference",
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
        "block_indices": list(range(20)),
        "segment_block_indices": [list(segment) for segment in SEGMENTS],
        "sequence_block_count": 20,
        "onnx_segment_count": 10,
        "reference_path": "reference.npz",
        "reference_sha256": sha256_file(REFERENCE_PATH),
        "reference_arrays": sorted(arrays.keys()),
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
        "lowbit_path": str(lowbit_path),
        "claim_scope": "reference tensors only; ONNX files are provided by separate persistent segment artifacts",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_kind": report["artifact_kind"],
        "sequence_block_count": report["sequence_block_count"],
        "onnx_segment_count": report["onnx_segment_count"],
        "reference_sha256": report["reference_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
