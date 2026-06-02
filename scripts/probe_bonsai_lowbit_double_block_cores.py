#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import (
    LOWBIT_REF,
    load_lowbit_transformer_state_dict,
    quantized_prefixes,
    recover_quantized_weight,
    unpack_cols_transposed,
    expand_col_groups,
)

OUT_DIR = Path("reports/bonsai-lowbit-double-block-cores")
BLOCK_LIMIT = int(os.getenv("BONSAI_DOUBLE_BLOCK_CORE_LIMIT", "0"))
BLOCK_PATTERN = re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_q$")


class LowBitLinearRuntime(torch.nn.Module):
    def __init__(self, sd: dict[str, Any], prefix: str):
        super().__init__()
        self.prefix = prefix
        self.register_buffer("wq_t", sd[f"{prefix}.W_q"].detach().cpu().contiguous(), persistent=True)
        self.register_buffer("scales", sd[f"{prefix}.scales"].detach().cpu().contiguous(), persistent=True)
        self.register_buffer("zeros", sd[f"{prefix}.zeros"].detach().cpu().contiguous(), persistent=True)
        self.orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
        self.metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
        self.nbits = int(self.metadata[1])
        self.group_size = int(self.metadata[2])

    def recovered_weight(self) -> torch.Tensor:
        unpacked = unpack_cols_transposed(self.wq_t, self.nbits, self.orig_shape[1])
        scales = expand_col_groups(self.scales, unpacked.shape[1], self.group_size)
        zeros = expand_col_groups(self.zeros, unpacked.shape[1], self.group_size)
        return unpacked * scales + zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.recovered_weight().to(dtype=x.dtype, device=x.device))


def block_prefix(block_index: int) -> str:
    return f"transformer_blocks.{block_index}"


def lb(sd: dict[str, Any], block_index: int, suffix: str) -> LowBitLinearRuntime:
    return LowBitLinearRuntime(sd, f"{block_prefix(block_index)}.{suffix}")


def ref_linear(sd: dict[str, Any], block_index: int, suffix: str, x: torch.Tensor) -> torch.Tensor:
    weight, _ = recover_quantized_weight(sd, f"{block_prefix(block_index)}.{suffix}", output_dtype=torch.float32)
    return F.linear(x, weight)


def rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.to(torch.float32).view(1, 1, 1, -1)).to(dtype=x.dtype)


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hidden_size: int, head_dim: int) -> torch.Tensor:
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * float(head_dim ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], hidden_size)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def project_runtime(sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q_img = lb(sd, block_index, "attn.to_q")(img)
    k_img = lb(sd, block_index, "attn.to_k")(img)
    v_img = lb(sd, block_index, "attn.to_v")(img)
    q_txt = lb(sd, block_index, "attn.add_q_proj")(txt)
    k_txt = lb(sd, block_index, "attn.add_k_proj")(txt)
    v_txt = lb(sd, block_index, "attn.add_v_proj")(txt)
    return q_img, k_img, v_img, q_txt, k_txt, v_txt


def project_reference(sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q_img = ref_linear(sd, block_index, "attn.to_q", img)
    k_img = ref_linear(sd, block_index, "attn.to_k", img)
    v_img = ref_linear(sd, block_index, "attn.to_v", img)
    q_txt = ref_linear(sd, block_index, "attn.add_q_proj", txt)
    k_txt = ref_linear(sd, block_index, "attn.add_k_proj", txt)
    v_txt = ref_linear(sd, block_index, "attn.add_v_proj", txt)
    return q_img, k_img, v_img, q_txt, k_txt, v_txt


def double_block_core_from_projections(
    sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor, projs: tuple[torch.Tensor, ...], runtime: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = block_prefix(block_index)
    q_img, k_img, v_img, q_txt, k_txt, v_txt = projs
    hidden_size = img.shape[-1]
    head_dim = int(sd[f"{prefix}.attn.norm_q.weight"].numel())
    heads = hidden_size // head_dim

    def shape(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], heads, head_dim)

    q_img = rms_norm_per_head(shape(q_img), sd[f"{prefix}.attn.norm_q.weight"])
    k_img = rms_norm_per_head(shape(k_img), sd[f"{prefix}.attn.norm_k.weight"])
    v_img = shape(v_img)
    q_txt = rms_norm_per_head(shape(q_txt), sd[f"{prefix}.attn.norm_added_q.weight"])
    k_txt = rms_norm_per_head(shape(k_txt), sd[f"{prefix}.attn.norm_added_k.weight"])
    v_txt = shape(v_txt)

    q = torch.cat([q_txt, q_img], dim=1)
    k = torch.cat([k_txt, k_img], dim=1)
    v = torch.cat([v_txt, v_img], dim=1)
    attended = explicit_attention(q, k, v, hidden_size, head_dim)
    txt_attn, img_attn = attended.split([txt.shape[1], img.shape[1]], dim=1)

    if runtime:
        img_attn_out = lb(sd, block_index, "attn.to_out.0")(img_attn)
        txt_attn_out = lb(sd, block_index, "attn.to_add_out")(txt_attn)
        img_ff = lb(sd, block_index, "ff.linear_out")(swiglu(lb(sd, block_index, "ff.linear_in")(img)))
        txt_ff = lb(sd, block_index, "ff_context.linear_out")(swiglu(lb(sd, block_index, "ff_context.linear_in")(txt)))
    else:
        img_attn_out = ref_linear(sd, block_index, "attn.to_out.0", img_attn)
        txt_attn_out = ref_linear(sd, block_index, "attn.to_add_out", txt_attn)
        img_ff = ref_linear(sd, block_index, "ff.linear_out", swiglu(ref_linear(sd, block_index, "ff.linear_in", img)))
        txt_ff = ref_linear(sd, block_index, "ff_context.linear_out", swiglu(ref_linear(sd, block_index, "ff_context.linear_in", txt)))
    return img_attn_out + img_ff, txt_attn_out + txt_ff


def double_block_core_runtime(sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return double_block_core_from_projections(sd, block_index, img, txt, project_runtime(sd, block_index, img, txt), runtime=True)


def double_block_core_reference(sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return double_block_core_from_projections(sd, block_index, img, txt, project_reference(sd, block_index, img, txt), runtime=False)


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


def check_block(sd: dict[str, Any], block_index: int) -> dict[str, Any]:
    hidden_size = [int(v) for v in sd[f"{block_prefix(block_index)}.attn.to_q.orig_shape"].tolist()][1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100_000 + block_index)
    img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    started = time.time()
    with torch.inference_mode():
        img_runtime, txt_runtime = double_block_core_runtime(sd, block_index, img, txt)
        img_ref, txt_ref = double_block_core_reference(sd, block_index, img, txt)
    seconds = round(time.time() - started, 4)
    img_diff = img_runtime - img_ref
    txt_diff = txt_runtime - txt_ref
    return {
        "block_index": block_index,
        "img_input_shape": list(img.shape),
        "txt_input_shape": list(txt.shape),
        "img_output_shape": list(img_runtime.shape),
        "txt_output_shape": list(txt_runtime.shape),
        "seconds": seconds,
        "mean_abs_error": max(float(img_diff.abs().mean()), float(txt_diff.abs().mean())),
        "max_abs_error": max(float(img_diff.abs().max()), float(txt_diff.abs().max())),
        "allclose_rtol_1e_4_atol_1e_5": bool(
            torch.allclose(img_runtime, img_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(txt_runtime, txt_ref, rtol=1e-4, atol=1e-5)
        ),
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
            result = check_block(sd, block_index)
            results.append(result)
            if not result["allclose_rtol_1e_4_atol_1e_5"]:
                failures.append(result)
        except Exception as exc:
            failure = {"block_index": block_index, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results.append(failure)
            failures.append(failure)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "all transformer_blocks.* core: image/text QKV attention + image/text FFN",
        "not_full_diffusers_block": True,
        "block_limit": BLOCK_LIMIT,
        "double_block_core_count": len(results),
        "failure_count": len(failures),
        "allclose_all": len(failures) == 0,
        "max_abs_error": max((r.get("max_abs_error", 0.0) for r in results), default=None),
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
        "allclose_all": report["allclose_all"],
        "max_abs_error": report["max_abs_error"],
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return 0 if report["allclose_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
