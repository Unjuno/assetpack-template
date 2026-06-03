#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
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

OUT_DIR = Path("reports/bonsai-lowbit-modulated-block-cores")
DOUBLE_BLOCK_COUNT = int(os.getenv("BONSAI_MODULATED_DOUBLE_BLOCKS", "5"))
SINGLE_BLOCK_COUNT = int(os.getenv("BONSAI_MODULATED_SINGLE_BLOCKS", "20"))
INPUT_SCALE = float(os.getenv("BONSAI_MODULATED_INPUT_SCALE", "0.01"))
TEMB_SCALE = float(os.getenv("BONSAI_MODULATED_TEMB_SCALE", "0.01"))


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


def lb(sd: dict[str, Any], prefix: str, x: torch.Tensor) -> torch.Tensor:
    return LowBitLinearRuntime(sd, prefix)(x)


def ref_linear(sd: dict[str, Any], prefix: str, x: torch.Tensor) -> torch.Tensor:
    weight, _ = recover_quantized_weight(sd, prefix, output_dtype=torch.float32)
    return F.linear(x, weight.to(dtype=x.dtype, device=x.device))


def pt_linear(sd: dict[str, Any], key: str, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, sd[key].to(dtype=x.dtype, device=x.device))


def rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.to(torch.float32).view(1, 1, 1, -1)).to(dtype=x.dtype)


def layer_norm_modulated(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = F.layer_norm(x.to(torch.float32), (x.shape[-1],), eps=eps)
    y = y * (1.0 + scale.to(torch.float32).unsqueeze(1)) + shift.to(torch.float32).unsqueeze(1)
    return y.to(dtype=x.dtype)


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


def modulation_chunks(sd: dict[str, Any], key: str, temb: torch.Tensor, chunks: int) -> tuple[torch.Tensor, ...]:
    return pt_linear(sd, key, temb).chunk(chunks, dim=-1)


def double_block_modulated(
    sd: dict[str, Any], block_index: int, img: torch.Tensor, txt: torch.Tensor, temb: torch.Tensor, runtime: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"transformer_blocks.{block_index}"
    img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = modulation_chunks(
        sd, "double_stream_modulation_img.linear.weight", temb, 6
    )
    txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = modulation_chunks(
        sd, "double_stream_modulation_txt.linear.weight", temb, 6
    )

    img_attn_in = layer_norm_modulated(img, img_shift_msa, img_scale_msa)
    txt_attn_in = layer_norm_modulated(txt, txt_shift_msa, txt_scale_msa)
    lin = lb if runtime else ref_linear
    q_img = lin(sd, f"{prefix}.attn.to_q", img_attn_in)
    k_img = lin(sd, f"{prefix}.attn.to_k", img_attn_in)
    v_img = lin(sd, f"{prefix}.attn.to_v", img_attn_in)
    q_txt = lin(sd, f"{prefix}.attn.add_q_proj", txt_attn_in)
    k_txt = lin(sd, f"{prefix}.attn.add_k_proj", txt_attn_in)
    v_txt = lin(sd, f"{prefix}.attn.add_v_proj", txt_attn_in)

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
    img = img + img_gate_msa.unsqueeze(1) * lin(sd, f"{prefix}.attn.to_out.0", img_attn)
    txt = txt + txt_gate_msa.unsqueeze(1) * lin(sd, f"{prefix}.attn.to_add_out", txt_attn)

    img_ff_in = layer_norm_modulated(img, img_shift_mlp, img_scale_mlp)
    txt_ff_in = layer_norm_modulated(txt, txt_shift_mlp, txt_scale_mlp)
    img = img + img_gate_mlp.unsqueeze(1) * lin(sd, f"{prefix}.ff.linear_out", swiglu(lin(sd, f"{prefix}.ff.linear_in", img_ff_in)))
    txt = txt + txt_gate_mlp.unsqueeze(1) * lin(
        sd, f"{prefix}.ff_context.linear_out", swiglu(lin(sd, f"{prefix}.ff_context.linear_in", txt_ff_in))
    )
    return img, txt


def single_block_modulated(sd: dict[str, Any], block_index: int, hidden: torch.Tensor, temb: torch.Tensor, runtime: bool) -> torch.Tensor:
    prefix = f"single_transformer_blocks.{block_index}"
    shift, scale, gate = modulation_chunks(sd, "single_stream_modulation.linear.weight", temb, 3)
    hidden_in = layer_norm_modulated(hidden, shift, scale)
    lin = lb if runtime else ref_linear
    fused = lin(sd, f"{prefix}.attn.to_qkv_mlp_proj", hidden_in)
    hidden_size = hidden.shape[-1]
    mlp_size = fused.shape[-1] - 3 * hidden_size
    q, k, v, mlp = torch.split(fused, [hidden_size, hidden_size, hidden_size, mlp_size], dim=-1)
    head_dim = int(sd[f"{prefix}.attn.norm_q.weight"].numel())
    heads = hidden_size // head_dim
    q = rms_norm_per_head(q.view(q.shape[0], q.shape[1], heads, head_dim), sd[f"{prefix}.attn.norm_q.weight"])
    k = rms_norm_per_head(k.view(k.shape[0], k.shape[1], heads, head_dim), sd[f"{prefix}.attn.norm_k.weight"])
    v = v.view(v.shape[0], v.shape[1], heads, head_dim)
    attn = explicit_attention(q, k, v, hidden_size, head_dim)
    joined = torch.cat([attn, swiglu(mlp)], dim=-1)
    return hidden + gate.unsqueeze(1) * lin(sd, f"{prefix}.attn.to_out", joined)


def stack(sd: dict[str, Any], img: torch.Tensor, txt: torch.Tensor, temb: torch.Tensor, runtime: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for index in range(DOUBLE_BLOCK_COUNT):
        img, txt = double_block_modulated(sd, index, img, txt, temb, runtime)
    hidden = torch.cat([txt, img], dim=1)
    for index in range(SINGLE_BLOCK_COUNT):
        hidden = single_block_modulated(sd, index, hidden, temb, runtime)
    txt_out, img_out = hidden.split([txt.shape[1], img.shape[1]], dim=1)
    return img_out, txt_out, hidden


def stat(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().to(torch.float32)
    finite = torch.isfinite(y)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(finite.all()),
        "min": float(y[finite].min()) if bool(finite.any()) else None,
        "max": float(y[finite].max()) if bool(finite.any()) else None,
        "mean": float(y[finite].mean()) if bool(finite.any()) else None,
        "std": float(y[finite].std()) if int(finite.sum()) > 1 else None,
    }


def failure_report(error: BaseException, lowbit_path: str | None = None) -> dict[str, Any]:
    return {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "modulated gated transformer block cores: 5 double + 20 single",
        "not_full_diffusers_transformer": True,
        "double_block_count": DOUBLE_BLOCK_COUNT,
        "single_block_count": SINGLE_BLOCK_COUNT,
        "input_scale": INPUT_SCALE,
        "temb_scale": TEMB_SCALE,
        "all_finite": False,
        "allclose_rtol_1e_4_atol_1e_5": False,
        "failure": {"error_type": type(error).__name__, "error": str(error)[:2000]},
        "lowbit_path": lowbit_path,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path: str | None = None
    try:
        loaded_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
        lowbit_path = str(loaded_path)
        hidden_size = [int(v) for v in sd["transformer_blocks.0.attn.to_q.orig_shape"].tolist()][1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(150_000)
        img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE
        txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE
        temb = torch.randn((1, hidden_size), generator=generator, dtype=torch.float32) * TEMB_SCALE
        started = time.time()
        with torch.inference_mode():
            img_runtime, txt_runtime, hidden_runtime = stack(sd, img, txt, temb, runtime=True)
            img_ref, txt_ref, hidden_ref = stack(sd, img, txt, temb, runtime=False)
        seconds = round(time.time() - started, 3)
        img_diff = img_runtime - img_ref
        txt_diff = txt_runtime - txt_ref
        hidden_diff = hidden_runtime - hidden_ref
        all_finite = all(bool(torch.isfinite(t).all()) for t in [img_runtime, txt_runtime, hidden_runtime, img_ref, txt_ref, hidden_ref])
        allclose = bool(
            all_finite
            and torch.allclose(img_runtime, img_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(txt_runtime, txt_ref, rtol=1e-4, atol=1e-5)
            and torch.allclose(hidden_runtime, hidden_ref, rtol=1e-4, atol=1e-5)
        )
        report = {
            "source_model_ref": LOWBIT_REF,
            "uses_lowbit_source": True,
            "writes_expanded_checkpoint": False,
            "target": "modulated gated transformer block cores: 5 double + 20 single",
            "not_full_diffusers_transformer": True,
            "missing_full_features": ["rotary positional embedding", "exact Diffusers forward wrapper", "patchify/unpatchify pipeline"],
            "double_block_count": DOUBLE_BLOCK_COUNT,
            "single_block_count": SINGLE_BLOCK_COUNT,
            "input_scale": INPUT_SCALE,
            "temb_scale": TEMB_SCALE,
            "seconds": seconds,
            "inputs": {"img": stat(img), "txt": stat(txt), "temb": stat(temb)},
            "outputs": {"img": stat(img_runtime), "txt": stat(txt_runtime), "hidden": stat(hidden_runtime)},
            "all_finite": all_finite,
            "mean_abs_error": max(float(img_diff.abs().mean()), float(txt_diff.abs().mean()), float(hidden_diff.abs().mean())),
            "max_abs_error": max(float(img_diff.abs().max()), float(txt_diff.abs().max()), float(hidden_diff.abs().max())),
            "allclose_rtol_1e_4_atol_1e_5": allclose,
            "failure": None,
            "lowbit_path": lowbit_path,
        }
    except Exception as exc:
        report = failure_report(exc, lowbit_path=lowbit_path)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_transformer": report["not_full_diffusers_transformer"],
        "double_block_count": report["double_block_count"],
        "single_block_count": report["single_block_count"],
        "all_finite": report["all_finite"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report.get("max_abs_error"),
        "seconds": report.get("seconds"),
        "failure": report.get("failure"),
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
