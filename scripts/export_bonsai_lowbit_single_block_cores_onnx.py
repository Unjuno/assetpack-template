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
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes
from export_bonsai_lowbit_single_block_core_onnx import LowBitLinearOnnx, initializer_summary, tensor_nbytes

OUT_DIR = Path("reports/bonsai-lowbit-single-block-cores-onnx")
BLOCK_LIMIT = int(os.getenv("BONSAI_SINGLE_BLOCK_CORE_ONNX_LIMIT", "0"))
BLOCK_PATTERN = re.compile(r"^single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj$")


class LowBitSingleBlockCoreOnnx(torch.nn.Module):
    def __init__(self, sd: dict[str, Any], block_index: int):
        super().__init__()
        self.block_index = block_index
        block_prefix = f"single_transformer_blocks.{block_index}"
        self.fused_prefix = f"{block_prefix}.attn.to_qkv_mlp_proj"
        self.out_prefix = f"{block_prefix}.attn.to_out"
        self.norm_q_key = f"{block_prefix}.attn.norm_q.weight"
        self.norm_k_key = f"{block_prefix}.attn.norm_k.weight"
        self.fused = LowBitLinearOnnx(sd, self.fused_prefix)
        self.to_out = LowBitLinearOnnx(sd, self.out_prefix)
        self.register_buffer("norm_q_weight", sd[self.norm_q_key].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_k_weight", sd[self.norm_k_key].detach().cpu().to(torch.float32), persistent=True)
        self.hidden_size = self.fused.in_features
        self.head_dim = int(self.norm_q_weight.numel())
        self.heads = self.hidden_size // self.head_dim
        self.mlp_size = self.fused.out_features - 3 * self.hidden_size
        if self.mlp_size <= 0:
            raise ValueError(f"invalid block core shape: block={block_index} fused_out={self.fused.out_features} hidden={self.hidden_size}")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fused(x)
        q, k, v, mlp = torch.split(fused, [self.hidden_size, self.hidden_size, self.hidden_size, self.mlp_size], dim=-1)
        bsz = x.shape[0]
        seq = x.shape[1]
        q = q.reshape(bsz, seq, self.heads, self.head_dim)
        k = k.reshape(bsz, seq, self.heads, self.head_dim)
        v = v.reshape(bsz, seq, self.heads, self.head_dim)
        q = self.rms_norm_per_head(q, self.norm_q_weight)
        k = self.rms_norm_per_head(k, self.norm_k_weight)
        attn = self.explicit_attention(q, k, v)
        mlp_a, mlp_b = torch.chunk(mlp, 2, dim=-1)
        mlp_act = F.silu(mlp_a) * mlp_b
        joined = torch.cat([attn, mlp_act], dim=-1)
        return self.to_out(joined)


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


def export_and_run(sd: dict[str, Any], block_index: int) -> dict[str, Any]:
    module = LowBitSingleBlockCoreOnnx(sd, block_index).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(80_000 + block_index)
    x = torch.randn((1, 2, module.hidden_size), generator=generator, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = module(x).detach().cpu().numpy()

    onnx_path = OUT_DIR / f"single_block_core_{block_index:02d}.onnx"
    started = time.time()
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        input_names=["hidden_states"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={"hidden_states": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )
    export_seconds = round(time.time() - started, 3)

    started = time.time()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden_states": x.cpu().numpy().astype(np.float32)})[0]
    ort_seconds = round(time.time() - started, 3)
    diff = y_ort - y_pt
    init = initializer_summary(onnx_path)

    prefixes = [module.fused_prefix, module.out_prefix]
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.{name}"]) for prefix in prefixes for name in ["W_q", "scales", "zeros"])
    expanded_fp32_weight_nbytes = sum(int(torch.prod(sd[f"{prefix}.orig_shape"]).item()) * 4 for prefix in prefixes)
    ratio = init["initializer_estimated_nbytes"] / expanded_fp32_weight_nbytes if expanded_fp32_weight_nbytes else None

    # Keep the artifact small and disk usage bounded; the report is the product.
    try:
        onnx_path.unlink()
    except OSError:
        pass

    return {
        "block_index": block_index,
        "input_shape": list(x.shape),
        "output_shape": list(y_pt.shape),
        "export_seconds": export_seconds,
        "ort_seconds": ort_seconds,
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_weight_nbytes,
        "initializer_summary": init,
        "initializer_to_expanded_fp32_ratio": ratio,
        "not_folded_to_expanded_weight": bool(ratio is not None and ratio < 0.5),
        "mean_abs_error": float(np.abs(diff).mean()),
        "max_abs_error": float(np.abs(diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(y_ort, y_pt, rtol=1e-4, atol=1e-5)),
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
        "target": "all single_transformer_blocks.* core ONNX: qkv_mlp -> norm_q/k -> explicit attention -> SwiGLU -> to_out",
        "not_full_diffusers_block": True,
        "block_limit": BLOCK_LIMIT,
        "block_core_count": len(results),
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
        "block_core_count": report["block_core_count"],
        "failure_count": report["failure_count"],
        "all_passed": report["all_passed"],
        "max_abs_error": report["max_abs_error"],
        "max_initializer_to_expanded_fp32_ratio": report["max_initializer_to_expanded_fp32_ratio"],
        "total_seconds": report["total_seconds"],
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
