#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import (
    LOWBIT_REF,
    load_lowbit_transformer_state_dict,
    recover_quantized_weight,
    unpack_cols_transposed,
    expand_col_groups,
)

OUT_DIR = Path("reports/bonsai-lowbit-double-block-core")
BLOCK_INDEX = 0
BLOCK_PREFIX = f"transformer_blocks.{BLOCK_INDEX}"


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


def rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.to(torch.float32).view(1, 1, 1, -1)).to(dtype=x.dtype)


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hidden_size: int, head_dim: int) -> torch.Tensor:
    # q/k/v: [B, S, H, D] -> [B, S, H*D]
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


def lb(sd: dict[str, Any], suffix: str) -> LowBitLinearRuntime:
    return LowBitLinearRuntime(sd, f"{BLOCK_PREFIX}.{suffix}")


def ref_linear(sd: dict[str, Any], suffix: str, x: torch.Tensor) -> torch.Tensor:
    weight, _ = recover_quantized_weight(sd, f"{BLOCK_PREFIX}.{suffix}", output_dtype=torch.float32)
    return F.linear(x, weight)


def project_runtime(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q_img = lb(sd, "attn.to_q")(img)
    k_img = lb(sd, "attn.to_k")(img)
    v_img = lb(sd, "attn.to_v")(img)
    q_txt = lb(sd, "attn.add_q_proj")(txt)
    k_txt = lb(sd, "attn.add_k_proj")(txt)
    v_txt = lb(sd, "attn.add_v_proj")(txt)
    return q_img, k_img, v_img, q_txt, k_txt, v_txt


def project_reference(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q_img = ref_linear(sd, "attn.to_q", img)
    k_img = ref_linear(sd, "attn.to_k", img)
    v_img = ref_linear(sd, "attn.to_v", img)
    q_txt = ref_linear(sd, "attn.add_q_proj", txt)
    k_txt = ref_linear(sd, "attn.add_k_proj", txt)
    v_txt = ref_linear(sd, "attn.add_v_proj", txt)
    return q_img, k_img, v_img, q_txt, k_txt, v_txt


def double_block_core_from_projections(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor, projs: tuple[torch.Tensor, ...], runtime: bool) -> tuple[torch.Tensor, torch.Tensor]:
    q_img, k_img, v_img, q_txt, k_txt, v_txt = projs
    hidden_size = img.shape[-1]
    head_dim = int(sd[f"{BLOCK_PREFIX}.attn.norm_q.weight"].numel())
    heads = hidden_size // head_dim

    def shape(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], heads, head_dim)

    q_img = rms_norm_per_head(shape(q_img), sd[f"{BLOCK_PREFIX}.attn.norm_q.weight"])
    k_img = rms_norm_per_head(shape(k_img), sd[f"{BLOCK_PREFIX}.attn.norm_k.weight"])
    v_img = shape(v_img)
    q_txt = rms_norm_per_head(shape(q_txt), sd[f"{BLOCK_PREFIX}.attn.norm_added_q.weight"])
    k_txt = rms_norm_per_head(shape(k_txt), sd[f"{BLOCK_PREFIX}.attn.norm_added_k.weight"])
    v_txt = shape(v_txt)

    q = torch.cat([q_txt, q_img], dim=1)
    k = torch.cat([k_txt, k_img], dim=1)
    v = torch.cat([v_txt, v_img], dim=1)
    attended = explicit_attention(q, k, v, hidden_size, head_dim)
    txt_attn, img_attn = attended.split([txt.shape[1], img.shape[1]], dim=1)

    if runtime:
        img_attn_out = lb(sd, "attn.to_out.0")(img_attn)
        txt_attn_out = lb(sd, "attn.to_add_out")(txt_attn)
        img_ff = lb(sd, "ff.linear_out")(swiglu(lb(sd, "ff.linear_in")(img)))
        txt_ff = lb(sd, "ff_context.linear_out")(swiglu(lb(sd, "ff_context.linear_in")(txt)))
    else:
        img_attn_out = ref_linear(sd, "attn.to_out.0", img_attn)
        txt_attn_out = ref_linear(sd, "attn.to_add_out", txt_attn)
        img_ff = ref_linear(sd, "ff.linear_out", swiglu(ref_linear(sd, "ff.linear_in", img)))
        txt_ff = ref_linear(sd, "ff_context.linear_out", swiglu(ref_linear(sd, "ff_context.linear_in", txt)))
    return img_attn_out + img_ff, txt_attn_out + txt_ff


def double_block_core_runtime(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return double_block_core_from_projections(sd, img, txt, project_runtime(sd, img, txt), runtime=True)


def double_block_core_reference(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return double_block_core_from_projections(sd, img, txt, project_reference(sd, img, txt), runtime=False)


def stat(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    hidden_size = [int(v) for v in sd[f"{BLOCK_PREFIX}.attn.to_q.orig_shape"].tolist()][1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(90_000)
    img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        img_runtime, txt_runtime = double_block_core_runtime(sd, img, txt)
        img_ref, txt_ref = double_block_core_reference(sd, img, txt)
    img_diff = img_runtime - img_ref
    txt_diff = txt_runtime - txt_ref
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "transformer_blocks.0 core: image/text QKV attention + image/text FFN",
        "not_full_diffusers_block": True,
        "lowbit_path": str(lowbit_path),
        "inputs": {"img": stat(img), "txt": stat(txt)},
        "outputs": {"img": stat(img_runtime), "txt": stat(txt_runtime)},
        "mean_abs_error": max(float(img_diff.abs().mean()), float(txt_diff.abs().mean())),
        "max_abs_error": max(float(img_diff.abs().max()), float(txt_diff.abs().max())),
        "allclose_rtol_1e_4_atol_1e_5": bool(
            torch.allclose(img_runtime, img_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(txt_runtime, txt_ref, rtol=1e-4, atol=1e-5)
        ),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_block": report["not_full_diffusers_block"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "outputs": report["outputs"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
