#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes
from export_bonsai_lowbit_single_block_core_onnx import LowBitLinearOnnx, initializer_summary, tensor_nbytes

OUT_DIR = Path("reports/bonsai-lowbit-double-block-cores-onnx")
BLOCK_LIMIT = int(os.getenv("BONSAI_DOUBLE_BLOCK_CORE_ONNX_LIMIT", "0"))
BLOCK_PATTERN = re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_q$")


class LowBitDoubleBlockCoreOnnx(torch.nn.Module):
    def __init__(self, sd: dict[str, Any], block_index: int):
        super().__init__()
        self.block_index = block_index
        self.prefix = f"transformer_blocks.{block_index}"
        self.img_to_q = LowBitLinearOnnx(sd, f"{self.prefix}.attn.to_q")
        self.img_to_k = LowBitLinearOnnx(sd, f"{self.prefix}.attn.to_k")
        self.img_to_v = LowBitLinearOnnx(sd, f"{self.prefix}.attn.to_v")
        self.img_to_out = LowBitLinearOnnx(sd, f"{self.prefix}.attn.to_out.0")
        self.txt_to_q = LowBitLinearOnnx(sd, f"{self.prefix}.attn.add_q_proj")
        self.txt_to_k = LowBitLinearOnnx(sd, f"{self.prefix}.attn.add_k_proj")
        self.txt_to_v = LowBitLinearOnnx(sd, f"{self.prefix}.attn.add_v_proj")
        self.txt_to_out = LowBitLinearOnnx(sd, f"{self.prefix}.attn.to_add_out")
        self.img_ff_in = LowBitLinearOnnx(sd, f"{self.prefix}.ff.linear_in")
        self.img_ff_out = LowBitLinearOnnx(sd, f"{self.prefix}.ff.linear_out")
        self.txt_ff_in = LowBitLinearOnnx(sd, f"{self.prefix}.ff_context.linear_in")
        self.txt_ff_out = LowBitLinearOnnx(sd, f"{self.prefix}.ff_context.linear_out")
        self.register_buffer("norm_q_weight", sd[f"{self.prefix}.attn.norm_q.weight"].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_k_weight", sd[f"{self.prefix}.attn.norm_k.weight"].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_added_q_weight", sd[f"{self.prefix}.attn.norm_added_q.weight"].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_added_k_weight", sd[f"{self.prefix}.attn.norm_added_k.weight"].detach().cpu().to(torch.float32), persistent=True)
        self.hidden_size = self.img_to_q.in_features
        self.head_dim = int(self.norm_q_weight.numel())
        self.heads = self.hidden_size // self.head_dim

    def rms_norm_per_head(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
        return y * weight.view(1, 1, 1, -1)

    def explicit_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * float(1.0 / math.sqrt(self.head_dim))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, vh)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.hidden_size)

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=-1)
        return F.silu(a) * b

    def shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_img = self.rms_norm_per_head(self.shape_heads(self.img_to_q(img)), self.norm_q_weight)
        k_img = self.rms_norm_per_head(self.shape_heads(self.img_to_k(img)), self.norm_k_weight)
        v_img = self.shape_heads(self.img_to_v(img))
        q_txt = self.rms_norm_per_head(self.shape_heads(self.txt_to_q(txt)), self.norm_added_q_weight)
        k_txt = self.rms_norm_per_head(self.shape_heads(self.txt_to_k(txt)), self.norm_added_k_weight)
        v_txt = self.shape_heads(self.txt_to_v(txt))

        q = torch.cat([q_txt, q_img], dim=1)
        k = torch.cat([k_txt, k_img], dim=1)
        v = torch.cat([v_txt, v_img], dim=1)
        attended = self.explicit_attention(q, k, v)
        txt_attn, img_attn = torch.split(attended, [txt.shape[1], img.shape[1]], dim=1)

        img_out = self.img_to_out(img_attn) + self.img_ff_out(self.swiglu(self.img_ff_in(img)))
        txt_out = self.txt_to_out(txt_attn) + self.txt_ff_out(self.swiglu(self.txt_ff_in(txt)))
        return img_out, txt_out


def block_indices(sd: dict[str, Any]) -> list[int]:
    indices = []
    for prefix in quantized_prefixes(sd):
        match = BLOCK_PATTERN.match(prefix)
        if match:
            indices.append(int(match.group(1)))
    out = sorted(indices)
    if BLOCK_LIMIT > 0:
        out = out[:BLOCK_LIMIT]
    return out


def block_quantized_prefixes(block_index: int) -> list[str]:
    prefix = f"transformer_blocks.{block_index}"
    return [
        f"{prefix}.attn.to_q",
        f"{prefix}.attn.to_k",
        f"{prefix}.attn.to_v",
        f"{prefix}.attn.to_out.0",
        f"{prefix}.attn.add_q_proj",
        f"{prefix}.attn.add_k_proj",
        f"{prefix}.attn.add_v_proj",
        f"{prefix}.attn.to_add_out",
        f"{prefix}.ff.linear_in",
        f"{prefix}.ff.linear_out",
        f"{prefix}.ff_context.linear_in",
        f"{prefix}.ff_context.linear_out",
    ]


def export_and_run(sd: dict[str, Any], block_index: int) -> dict[str, Any]:
    module = LowBitDoubleBlockCoreOnnx(sd, block_index).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(110_000 + block_index)
    img = torch.randn((1, 2, module.hidden_size), generator=generator, dtype=torch.float32) / 10.0
    txt = torch.randn((1, 3, module.hidden_size), generator=generator, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        img_pt, txt_pt = [t.detach().cpu().numpy() for t in module(img, txt)]

    onnx_path = OUT_DIR / f"double_block_core_{block_index:02d}.onnx"
    started = time.time()
    torch.onnx.export(
        module,
        (img, txt),
        str(onnx_path),
        input_names=["img", "txt"],
        output_names=["img_out", "txt_out"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={
            "img": {0: "batch", 1: "img_seq"},
            "txt": {0: "batch", 1: "txt_seq"},
            "img_out": {0: "batch", 1: "img_seq"},
            "txt_out": {0: "batch", 1: "txt_seq"},
        },
    )
    export_seconds = round(time.time() - started, 3)

    started = time.time()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img_ort, txt_ort = session.run(None, {"img": img.cpu().numpy().astype(np.float32), "txt": txt.cpu().numpy().astype(np.float32)})
    ort_seconds = round(time.time() - started, 3)
    img_diff = img_ort - img_pt
    txt_diff = txt_ort - txt_pt
    init = initializer_summary(onnx_path)

    prefixes = block_quantized_prefixes(block_index)
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.{name}"]) for prefix in prefixes for name in ["W_q", "scales", "zeros"])
    expanded_fp32_weight_nbytes = sum(int(torch.prod(sd[f"{prefix}.orig_shape"]).item()) * 4 for prefix in prefixes)
    ratio = init["initializer_estimated_nbytes"] / expanded_fp32_weight_nbytes if expanded_fp32_weight_nbytes else None

    try:
        onnx_path.unlink()
    except OSError:
        pass

    allclose = bool(
        np.allclose(img_ort, img_pt, rtol=1e-4, atol=1e-5)
        and np.allclose(txt_ort, txt_pt, rtol=1e-4, atol=1e-5)
    )
    return {
        "block_index": block_index,
        "img_input_shape": list(img.shape),
        "txt_input_shape": list(txt.shape),
        "img_output_shape": list(img_pt.shape),
        "txt_output_shape": list(txt_pt.shape),
        "export_seconds": export_seconds,
        "ort_seconds": ort_seconds,
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_weight_nbytes,
        "initializer_summary": init,
        "initializer_to_expanded_fp32_ratio": ratio,
        "not_folded_to_expanded_weight": bool(ratio is not None and ratio < 0.5),
        "mean_abs_error": max(float(np.abs(img_diff).mean()), float(np.abs(txt_diff).mean())),
        "max_abs_error": max(float(np.abs(img_diff).max()), float(np.abs(txt_diff).max())),
        "allclose_rtol_1e_4_atol_1e_5": allclose,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    indices = block_indices(sd)
    results = []
    failures = []
    started = time.time()
    for block_index in indices:
        try:
            result = export_and_run(sd, block_index)
            results.append(result)
            if not result["allclose_rtol_1e_4_atol_1e_5"] or not result["not_folded_to_expanded_weight"]:
                failures.append(result)
        except Exception as exc:
            failure = {"block_index": block_index, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results.append(failure)
            failures.append(failure)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "attention_lowering": "explicit_matmul_softmax_matmul",
        "target": "all transformer_blocks.* core ONNX: image/text QKV attention + image/text FFN",
        "not_full_diffusers_block": True,
        "block_limit": BLOCK_LIMIT,
        "double_block_core_count": len(results),
        "failure_count": len(failures),
        "all_passed": len(failures) == 0,
        "max_abs_error": max((r.get("max_abs_error", 0.0) for r in results), default=None),
        "max_initializer_to_expanded_fp32_ratio": max((r.get("initializer_to_expanded_fp32_ratio", 0.0) for r in results), default=None),
        "total_seconds": round(time.time() - started, 3),
        "lowbit_path": str(lowbit_path),
        "failures": failures[:20],
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_block": report["not_full_diffusers_block"],
        "double_block_core_count": report["double_block_core_count"],
        "failure_count": report["failure_count"],
        "all_passed": report["all_passed"],
        "max_abs_error": report["max_abs_error"],
        "max_initializer_to_expanded_fp32_ratio": report["max_initializer_to_expanded_fp32_ratio"],
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
