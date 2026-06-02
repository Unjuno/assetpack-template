#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_lowbit_double_block_cores import double_block_core_reference, double_block_core_runtime
from probe_bonsai_lowbit_single_block_cores import block_core_reference as single_block_core_reference
from probe_bonsai_lowbit_single_block_cores import block_core_runtime as single_block_core_runtime

OUT_DIR = Path("reports/bonsai-lowbit-transformer-core-stack")
DOUBLE_BLOCK_COUNT = 5
SINGLE_BLOCK_COUNT = 20


def stat(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


def runtime_stack(sd: dict, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_state = img
    txt_state = txt
    for index in range(DOUBLE_BLOCK_COUNT):
        img_delta, txt_delta = double_block_core_runtime(sd, index, img_state, txt_state)
        img_state = img_state + img_delta
        txt_state = txt_state + txt_delta
    hidden = torch.cat([txt_state, img_state], dim=1)
    for index in range(SINGLE_BLOCK_COUNT):
        hidden = hidden + single_block_core_runtime(sd, index, hidden)
    txt_out, img_out = hidden.split([txt_state.shape[1], img_state.shape[1]], dim=1)
    return img_out, txt_out, hidden


def reference_stack(sd: dict, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_state = img
    txt_state = txt
    for index in range(DOUBLE_BLOCK_COUNT):
        img_delta, txt_delta = double_block_core_reference(sd, index, img_state, txt_state)
        img_state = img_state + img_delta
        txt_state = txt_state + txt_delta
    hidden = torch.cat([txt_state, img_state], dim=1)
    for index in range(SINGLE_BLOCK_COUNT):
        hidden = hidden + single_block_core_reference(sd, index, hidden)
    txt_out, img_out = hidden.split([txt_state.shape[1], img_state.shape[1]], dim=1)
    return img_out, txt_out, hidden


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    hidden_size = [int(v) for v in sd["transformer_blocks.0.attn.to_q.orig_shape"].tolist()][1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(120_000)
    img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) / 10.0

    started = time.time()
    with torch.inference_mode():
        img_runtime, txt_runtime, hidden_runtime = runtime_stack(sd, img, txt)
        img_ref, txt_ref, hidden_ref = reference_stack(sd, img, txt)
    seconds = round(time.time() - started, 3)

    img_diff = img_runtime - img_ref
    txt_diff = txt_runtime - txt_ref
    hidden_diff = hidden_runtime - hidden_ref
    allclose = bool(
        torch.allclose(img_runtime, img_ref, rtol=1e-4, atol=1e-5)
        and torch.allclose(txt_runtime, txt_ref, rtol=1e-4, atol=1e-5)
        and torch.allclose(hidden_runtime, hidden_ref, rtol=1e-4, atol=1e-5)
    )
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "core stack: 5 double block cores + 20 single block cores",
        "not_full_diffusers_transformer": True,
        "missing_full_features": [
            "time/text conditioning embeddings",
            "AdaLayerNorm modulation",
            "full residual/gating semantics from Diffusers",
            "rotary positional embedding",
            "final projection/unpatchify pipeline",
        ],
        "double_block_count": DOUBLE_BLOCK_COUNT,
        "single_block_count": SINGLE_BLOCK_COUNT,
        "seconds": seconds,
        "inputs": {"img": stat(img), "txt": stat(txt)},
        "outputs": {"img": stat(img_runtime), "txt": stat(txt_runtime), "hidden": stat(hidden_runtime)},
        "mean_abs_error": max(float(img_diff.abs().mean()), float(txt_diff.abs().mean()), float(hidden_diff.abs().mean())),
        "max_abs_error": max(float(img_diff.abs().max()), float(txt_diff.abs().max()), float(hidden_diff.abs().max())),
        "allclose_rtol_1e_4_atol_1e_5": allclose,
        "lowbit_path": str(lowbit_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_transformer": report["not_full_diffusers_transformer"],
        "double_block_count": report["double_block_count"],
        "single_block_count": report["single_block_count"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "seconds": report["seconds"],
    }, indent=2))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
