#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_two_single_blocks_modulated_onnx import SEQ_LEN, TwoSingleBlocksModulated, build_core

OUT_DIR = Path("reports/bonsai-lowbit-single-stream-chain-runner")
SUPPORTED_BLOCK_COUNTS = (2, 4, 8, 16, 20)
SEGMENT_SIZE = 2


def build_segments(sd: dict[str, Any], block_count: int) -> list[TwoSingleBlocksModulated]:
    if block_count not in SUPPORTED_BLOCK_COUNTS:
        raise ValueError(f"block_count must be one of {SUPPORTED_BLOCK_COUNTS}, got {block_count}")
    segments = []
    for start in range(0, block_count, SEGMENT_SIZE):
        cores = [build_core(sd, index) for index in range(start, start + SEGMENT_SIZE)]
        segments.append(TwoSingleBlocksModulated(cores).eval())
    return segments


def run_torch_chain(segments: list[TwoSingleBlocksModulated], hidden: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    h = hidden
    with torch.inference_mode():
        for segment_index, segment in enumerate(segments):
            outputs = segment(h, temb)
            block0_output, block1_output = outputs[0], outputs[1]
            start = segment_index * SEGMENT_SIZE
            trace.append({
                "segment_index": segment_index,
                "block_indices": [start, start + 1],
                "input_shape": list(h.shape),
                "block0_output_shape": list(block0_output.shape),
                "block1_output_shape": list(block1_output.shape),
                "block1_output_mean_abs": float(block1_output.abs().mean().item()),
                "block1_output_max_abs": float(block1_output.abs().max().item()),
            })
            h = block1_output
    return h, trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bonsai low-bit single-stream modulated block chain in PyTorch.")
    parser.add_argument("--blocks", type=int, default=2, choices=SUPPORTED_BLOCK_COUNTS, help="Number of single-stream blocks to run")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    segments = build_segments(sd, args.blocks)
    hidden_dim = int(segments[0].hidden_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    hidden = torch.randn(args.batch, args.seq, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    temb = torch.randn(args.batch, hidden_dim, generator=generator, dtype=torch.float32) * 0.01

    final_hidden, trace = run_torch_chain(segments, hidden, temb)
    report = {
        "ok": True,
        "runner_kind": "torch_lowbit_single_stream_modulated_block_chain",
        "source_model_ref": LOWBIT_REF,
        "lowbit_path": str(lowbit_path),
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "backend": "torch_cpu",
        "block_count": args.blocks,
        "segment_size": SEGMENT_SIZE,
        "segment_count": len(segments),
        "block_indices": list(range(args.blocks)),
        "segment_block_indices": [[index, index + 1] for index in range(0, args.blocks, SEGMENT_SIZE)],
        "batch": args.batch,
        "sequence_length": args.seq,
        "hidden_dim": hidden_dim,
        "seed": args.seed,
        "input_shape": list(hidden.shape),
        "temb_shape": list(temb.shape),
        "final_hidden_shape": list(final_hidden.shape),
        "final_hidden_mean_abs": float(final_hidden.abs().mean().item()),
        "final_hidden_max_abs": float(final_hidden.abs().max().item()),
        "trace": trace,
        "is_real_transformer_block": False,
        "is_full_bonsai_pipeline": False,
        "is_prompt_to_image_pipeline": False,
        "claim": "lowbit_single_stream_modulated_block_chain_torch_cpu_executable_not_real_transformer_block_or_full_bonsai_pipeline",
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"],
        "runner_kind": report["runner_kind"],
        "backend": report["backend"],
        "block_count": report["block_count"],
        "segment_count": report["segment_count"],
        "final_hidden_shape": report["final_hidden_shape"],
        "report_path": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
