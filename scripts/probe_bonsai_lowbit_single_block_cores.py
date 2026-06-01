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

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes, recover_quantized_weight, unpack_cols_transposed, expand_col_groups

OUT_DIR = Path("reports/bonsai-lowbit-single-block-cores")
BLOCK_LIMIT = int(os.getenv("BONSAI_SINGLE_BLOCK_CORE_LIMIT", "0"))
BLOCK_PATTERN = re.compile(r"^single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj$")


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


def block_core_runtime(sd: dict[str, Any], block_index: int, x: torch.Tensor) -> torch.Tensor:
    block_prefix = f"single_transformer_blocks.{block_index}"
    fused_prefix = f"{block_prefix}.attn.to_qkv_mlp_proj"
    out_prefix = f"{block_prefix}.attn.to_out"
    norm_q_key = f"{block_prefix}.attn.norm_q.weight"
    norm_k_key = f"{block_prefix}.attn.norm_k.weight"

    fused = LowBitLinearRuntime(sd, fused_prefix)(x)
    hidden_size = x.shape[-1]
    mlp_size = fused.shape[-1] - 3 * hidden_size
    q, k, v, mlp = torch.split(fused, [hidden_size, hidden_size, hidden_size, mlp_size], dim=-1)
    head_dim = int(sd[norm_q_key].numel())
    heads = hidden_size // head_dim
    q = q.view(x.shape[0], x.shape[1], heads, head_dim)
    k = k.view(x.shape[0], x.shape[1], heads, head_dim)
    v = v.view(x.shape[0], x.shape[1], heads, head_dim)
    q = rms_norm_per_head(q, sd[norm_q_key])
    k = rms_norm_per_head(k, sd[norm_k_key])
    attn = explicit_attention(q, k, v, hidden_size, head_dim)
    joined = torch.cat([attn, swiglu(mlp)], dim=-1)
    return LowBitLinearRuntime(sd, out_prefix)(joined)


def block_core_reference(sd: dict[str, Any], block_index: int, x: torch.Tensor) -> torch.Tensor:
    block_prefix = f"single_transformer_blocks.{block_index}"
    fused_prefix = f"{block_prefix}.attn.to_qkv_mlp_proj"
    out_prefix = f"{block_prefix}.attn.to_out"
    norm_q_key = f"{block_prefix}.attn.norm_q.weight"
    norm_k_key = f"{block_prefix}.attn.norm_k.weight"

    fused_w, _ = recover_quantized_weight(sd, fused_prefix, output_dtype=torch.float32)
    out_w, _ = recover_quantized_weight(sd, out_prefix, output_dtype=torch.float32)
    fused = F.linear(x, fused_w)
    hidden_size = x.shape[-1]
    mlp_size = fused.shape[-1] - 3 * hidden_size
    q, k, v, mlp = torch.split(fused, [hidden_size, hidden_size, hidden_size, mlp_size], dim=-1)
    head_dim = int(sd[norm_q_key].numel())
    heads = hidden_size // head_dim
    q = q.view(x.shape[0], x.shape[1], heads, head_dim)
    k = k.view(x.shape[0], x.shape[1], heads, head_dim)
    v = v.view(x.shape[0], x.shape[1], heads, head_dim)
    q = rms_norm_per_head(q, sd[norm_q_key])
    k = rms_norm_per_head(k, sd[norm_k_key])
    attn = explicit_attention(q, k, v, hidden_size, head_dim)
    joined = torch.cat([attn, swiglu(mlp)], dim=-1)
    return F.linear(joined, out_w)


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
    fused_prefix = f"single_transformer_blocks.{block_index}.attn.to_qkv_mlp_proj"
    hidden_size = [int(v) for v in sd[f"{fused_prefix}.orig_shape"].tolist()][1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(70_000 + block_index)
    x = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) / 10.0
    started = time.time()
    with torch.inference_mode():
        y_runtime = block_core_runtime(sd, block_index, x)
        y_ref = block_core_reference(sd, block_index, x)
    seconds = round(time.time() - started, 4)
    diff = y_runtime - y_ref
    return {
        "block_index": block_index,
        "input_shape": list(x.shape),
        "output_shape": list(y_runtime.shape),
        "seconds": seconds,
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(torch.allclose(y_runtime, y_ref, rtol=1e-4, atol=1e-5)),
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
        "target": "all single_transformer_blocks.* core: qkv_mlp -> norm_q/k -> explicit attention -> SwiGLU -> to_out",
        "not_full_diffusers_block": True,
        "block_limit": BLOCK_LIMIT,
        "block_core_count": len(results),
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
        "block_core_count": report["block_core_count"],
        "failure_count": report["failure_count"],
        "allclose_all": report["allclose_all"],
        "max_abs_error": report["max_abs_error"],
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return 0 if report["allclose_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
