#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
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
INPUT_SCALE = float(os.getenv("BONSAI_CORE_STACK_INPUT_SCALE", "0.01"))
RESIDUAL_SCALE = float(os.getenv("BONSAI_CORE_STACK_RESIDUAL_SCALE", "0.001"))


def stat(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    finite = torch.isfinite(y)
    if not bool(finite.all()):
        finite_y = y[finite]
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "finite": False,
            "finite_count": int(finite.sum()),
            "numel": int(y.numel()),
            "min": None if finite_y.numel() == 0 else float(finite_y.min()),
            "max": None if finite_y.numel() == 0 else float(finite_y.max()),
            "mean": None if finite_y.numel() == 0 else float(finite_y.mean()),
            "std": None if finite_y.numel() <= 1 else float(finite_y.std()),
        }
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": True,
        "finite_count": int(finite.sum()),
        "numel": int(y.numel()),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


def ensure_finite(name: str, *tensors: torch.Tensor, trace: list[dict]) -> None:
    for tensor_index, tensor in enumerate(tensors):
        if not bool(torch.isfinite(tensor).all()):
            trace.append({"stage": name, "tensor_index": tensor_index, "finite": False, "stat": stat(tensor)})
            raise FloatingPointError(f"non-finite tensor at {name}[{tensor_index}]")


def runtime_stack(sd: dict, img: torch.Tensor, txt: torch.Tensor, trace: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_state = img
    txt_state = txt
    ensure_finite("runtime.input", img_state, txt_state, trace=trace)
    for index in range(DOUBLE_BLOCK_COUNT):
        img_delta, txt_delta = double_block_core_runtime(sd, index, img_state, txt_state)
        ensure_finite(f"runtime.double_block.{index}.delta", img_delta, txt_delta, trace=trace)
        img_state = img_state + RESIDUAL_SCALE * img_delta
        txt_state = txt_state + RESIDUAL_SCALE * txt_delta
        ensure_finite(f"runtime.double_block.{index}.state", img_state, txt_state, trace=trace)
    hidden = torch.cat([txt_state, img_state], dim=1)
    ensure_finite("runtime.concat", hidden, trace=trace)
    for index in range(SINGLE_BLOCK_COUNT):
        delta = single_block_core_runtime(sd, index, hidden)
        ensure_finite(f"runtime.single_block.{index}.delta", delta, trace=trace)
        hidden = hidden + RESIDUAL_SCALE * delta
        ensure_finite(f"runtime.single_block.{index}.state", hidden, trace=trace)
    txt_out, img_out = hidden.split([txt_state.shape[1], img_state.shape[1]], dim=1)
    return img_out, txt_out, hidden


def reference_stack(sd: dict, img: torch.Tensor, txt: torch.Tensor, trace: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_state = img
    txt_state = txt
    ensure_finite("reference.input", img_state, txt_state, trace=trace)
    for index in range(DOUBLE_BLOCK_COUNT):
        img_delta, txt_delta = double_block_core_reference(sd, index, img_state, txt_state)
        ensure_finite(f"reference.double_block.{index}.delta", img_delta, txt_delta, trace=trace)
        img_state = img_state + RESIDUAL_SCALE * img_delta
        txt_state = txt_state + RESIDUAL_SCALE * txt_delta
        ensure_finite(f"reference.double_block.{index}.state", img_state, txt_state, trace=trace)
    hidden = torch.cat([txt_state, img_state], dim=1)
    ensure_finite("reference.concat", hidden, trace=trace)
    for index in range(SINGLE_BLOCK_COUNT):
        delta = single_block_core_reference(sd, index, hidden)
        ensure_finite(f"reference.single_block.{index}.delta", delta, trace=trace)
        hidden = hidden + RESIDUAL_SCALE * delta
        ensure_finite(f"reference.single_block.{index}.state", hidden, trace=trace)
    txt_out, img_out = hidden.split([txt_state.shape[1], img_state.shape[1]], dim=1)
    return img_out, txt_out, hidden


def finite_diff_stat(diff: torch.Tensor) -> dict:
    finite = torch.isfinite(diff)
    if not bool(finite.all()):
        return {"finite": False, "mean_abs_error": math.nan, "max_abs_error": math.nan}
    return {"finite": True, "mean_abs_error": float(diff.abs().mean()), "max_abs_error": float(diff.abs().max())}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    hidden_size = [int(v) for v in sd["transformer_blocks.0.attn.to_q.orig_shape"].tolist()][1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(120_000)
    img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE
    txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE

    trace: list[dict] = []
    started = time.time()
    error: dict | None = None
    try:
        with torch.inference_mode():
            img_runtime, txt_runtime, hidden_runtime = runtime_stack(sd, img, txt, trace)
            img_ref, txt_ref, hidden_ref = reference_stack(sd, img, txt, trace)
        img_diff = img_runtime - img_ref
        txt_diff = txt_runtime - txt_ref
        hidden_diff = hidden_runtime - hidden_ref
        diff_stats = [finite_diff_stat(img_diff), finite_diff_stat(txt_diff), finite_diff_stat(hidden_diff)]
        all_finite = all(d["finite"] for d in diff_stats) and all(
            stat(t)["finite"] for t in [img_runtime, txt_runtime, hidden_runtime, img_ref, txt_ref, hidden_ref]
        )
        allclose = bool(
            all_finite
            and torch.allclose(img_runtime, img_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(txt_runtime, txt_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(hidden_runtime, hidden_ref, rtol=1e-4, atol=1e-5)
        )
        outputs = {"img": stat(img_runtime), "txt": stat(txt_runtime), "hidden": stat(hidden_runtime)}
        mean_abs_error = max(float(d["mean_abs_error"]) for d in diff_stats)
        max_abs_error = max(float(d["max_abs_error"]) for d in diff_stats)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
        outputs = {}
        all_finite = False
        allclose = False
        mean_abs_error = math.nan
        max_abs_error = math.nan
    seconds = round(time.time() - started, 3)

    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "numerically damped core stack: 5 double block cores + 20 single block cores",
        "not_full_diffusers_transformer": True,
        "input_scale": INPUT_SCALE,
        "residual_scale": RESIDUAL_SCALE,
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
        "outputs": outputs,
        "all_finite": all_finite,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "allclose_rtol_1e_4_atol_1e_5": allclose,
        "failure": error,
        "trace_tail": trace[-20:],
        "lowbit_path": str(lowbit_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_transformer": report["not_full_diffusers_transformer"],
        "input_scale": report["input_scale"],
        "residual_scale": report["residual_scale"],
        "double_block_count": report["double_block_count"],
        "single_block_count": report["single_block_count"],
        "all_finite": report["all_finite"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "seconds": report["seconds"],
        "failure": report["failure"],
    }, indent=2, allow_nan=True))
    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
