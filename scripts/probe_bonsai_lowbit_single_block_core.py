#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, recover_quantized_weight, unpack_cols_transposed, expand_col_groups

OUT_DIR = Path("reports/bonsai-lowbit-single-block-core")
BLOCK_PREFIX = "single_transformer_blocks.0"
FUSED_PREFIX = f"{BLOCK_PREFIX}.attn.to_qkv_mlp_proj"
OUT_PREFIX = f"{BLOCK_PREFIX}.attn.to_out"
NORM_Q_KEY = f"{BLOCK_PREFIX}.attn.norm_q.weight"
NORM_K_KEY = f"{BLOCK_PREFIX}.attn.norm_k.weight"


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
    # x: [batch, seq, heads, head_dim], weight: [head_dim]
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.to(torch.float32).view(1, 1, 1, -1)).to(dtype=x.dtype)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def block_core_runtime(sd: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    fused = LowBitLinearRuntime(sd, FUSED_PREFIX)(x)
    out_features = fused.shape[-1]
    hidden_size = x.shape[-1]
    mlp_size = out_features - 3 * hidden_size
    q, k, v, mlp = torch.split(fused, [hidden_size, hidden_size, hidden_size, mlp_size], dim=-1)
    head_dim = int(sd[NORM_Q_KEY].numel())
    heads = hidden_size // head_dim
    q = q.view(x.shape[0], x.shape[1], heads, head_dim)
    k = k.view(x.shape[0], x.shape[1], heads, head_dim)
    v = v.view(x.shape[0], x.shape[1], heads, head_dim)
    q = rms_norm_per_head(q, sd[NORM_Q_KEY])
    k = rms_norm_per_head(k, sd[NORM_K_KEY])
    # scaled_dot_product_attention expects [batch, heads, seq, head_dim]
    attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], hidden_size)
    mlp_act = swiglu(mlp)
    joined = torch.cat([attn, mlp_act], dim=-1)
    return LowBitLinearRuntime(sd, OUT_PREFIX)(joined)


def block_core_reference(sd: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    fused_w, _ = recover_quantized_weight(sd, FUSED_PREFIX, output_dtype=torch.float32)
    out_w, _ = recover_quantized_weight(sd, OUT_PREFIX, output_dtype=torch.float32)
    fused = F.linear(x, fused_w)
    out_features = fused.shape[-1]
    hidden_size = x.shape[-1]
    mlp_size = out_features - 3 * hidden_size
    q, k, v, mlp = torch.split(fused, [hidden_size, hidden_size, hidden_size, mlp_size], dim=-1)
    head_dim = int(sd[NORM_Q_KEY].numel())
    heads = hidden_size // head_dim
    q = q.view(x.shape[0], x.shape[1], heads, head_dim)
    k = k.view(x.shape[0], x.shape[1], heads, head_dim)
    v = v.view(x.shape[0], x.shape[1], heads, head_dim)
    q = rms_norm_per_head(q, sd[NORM_Q_KEY])
    k = rms_norm_per_head(k, sd[NORM_K_KEY])
    attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], hidden_size)
    joined = torch.cat([attn, swiglu(mlp)], dim=-1)
    return F.linear(joined, out_w)


def stat(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(50_000)
    hidden_size = [int(v) for v in sd[f"{FUSED_PREFIX}.orig_shape"].tolist()][1]
    x = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_runtime = block_core_runtime(sd, x)
        y_ref = block_core_reference(sd, x)
    diff = y_runtime - y_ref
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "single_transformer_blocks.0 core: qkv_mlp -> norm_q/k -> SDPA -> SwiGLU -> to_out",
        "not_full_diffusers_block": True,
        "lowbit_path": str(lowbit_path),
        "input": stat(x),
        "output": stat(y_runtime),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(torch.allclose(y_runtime, y_ref, rtol=1e-4, atol=1e-5)),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_block": report["not_full_diffusers_block"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "output": report["output"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
