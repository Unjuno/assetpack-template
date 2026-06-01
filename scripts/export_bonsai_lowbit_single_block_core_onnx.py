#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict

OUT_DIR = Path("reports/bonsai-lowbit-single-block-core-onnx")
BLOCK_PREFIX = "single_transformer_blocks.0"
FUSED_PREFIX = f"{BLOCK_PREFIX}.attn.to_qkv_mlp_proj"
OUT_PREFIX = f"{BLOCK_PREFIX}.attn.to_out"
NORM_Q_KEY = f"{BLOCK_PREFIX}.attn.norm_q.weight"
NORM_K_KEY = f"{BLOCK_PREFIX}.attn.norm_k.weight"


class LowBitLinearOnnx(torch.nn.Module):
    def __init__(self, sd: dict[str, Any], prefix: str):
        super().__init__()
        self.register_buffer("wq_t", sd[f"{prefix}.W_q"].detach().cpu().contiguous(), persistent=True)
        self.register_buffer("scales", sd[f"{prefix}.scales"].detach().cpu().contiguous(), persistent=True)
        self.register_buffer("zeros", sd[f"{prefix}.zeros"].detach().cpu().contiguous(), persistent=True)
        self.orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
        self.metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
        self.out_features = int(self.orig_shape[0])
        self.in_features = int(self.orig_shape[1])
        self.nbits = int(self.metadata[1])
        self.group_size = int(self.metadata[2])
        self.packed_cols = int(self.wq_t.shape[0])
        self.elements_per_sample = self.in_features // self.packed_cols

    def recovered_weight(self) -> torch.Tensor:
        # ONNX-friendly unpack: floor(W_q / 2**shift) mod 2**nbits.
        wq = self.wq_t.t().contiguous().to(torch.float32)
        shifts = torch.arange(self.elements_per_sample, dtype=torch.float32, device=wq.device) * float(self.nbits)
        divisors = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=wq.device), shifts)
        shifted = torch.floor(wq.unsqueeze(-1) / divisors)
        unpacked = torch.remainder(shifted, float(1 << self.nbits)).reshape(self.out_features, self.in_features)
        scales = self.scales.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        zeros = self.zeros.t().contiguous().to(torch.float32).repeat_interleave(self.group_size, dim=1)
        return unpacked * scales + zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.recovered_weight().to(dtype=x.dtype))


class LowBitSingleBlockCoreOnnx(torch.nn.Module):
    def __init__(self, sd: dict[str, Any]):
        super().__init__()
        self.fused = LowBitLinearOnnx(sd, FUSED_PREFIX)
        self.to_out = LowBitLinearOnnx(sd, OUT_PREFIX)
        self.register_buffer("norm_q_weight", sd[NORM_Q_KEY].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_k_weight", sd[NORM_K_KEY].detach().cpu().to(torch.float32), persistent=True)
        self.hidden_size = self.fused.in_features
        self.head_dim = int(self.norm_q_weight.numel())
        self.heads = self.hidden_size // self.head_dim
        self.mlp_size = self.fused.out_features - 3 * self.hidden_size
        if self.mlp_size <= 0:
            raise ValueError(f"invalid block core shape: fused_out={self.fused.out_features} hidden={self.hidden_size}")

    def rms_norm_per_head(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
        return y * weight.view(1, 1, 1, -1)

    def explicit_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q/k/v: [B, S, H, D] -> output [B, S, H*D]
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


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def initializer_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    total = 0
    max_init = 0
    count = 0
    for init in model.graph.initializer:
        nbytes = 1
        for dim in init.dims:
            nbytes *= int(dim)
        width = {1: 4, 2: 1, 6: 4, 7: 8, 10: 2, 16: 2}.get(init.data_type, 4)
        nbytes *= width
        total += nbytes
        max_init = max(max_init, nbytes)
        count += 1
    return {"initializer_count": count, "initializer_estimated_nbytes": total, "max_initializer_estimated_nbytes": max_init}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    module = LowBitSingleBlockCoreOnnx(sd).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(60_000)
    x = torch.randn((1, 2, module.hidden_size), generator=generator, dtype=torch.float32) / 10.0
    with torch.inference_mode():
        y_pt = module(x).detach().cpu().numpy()

    onnx_path = OUT_DIR / "single_block_core.onnx"
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

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = session.run(None, {"hidden_states": x.cpu().numpy().astype(np.float32)})[0]
    diff = y_ort - y_pt
    init = initializer_summary(onnx_path)

    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.{name}"]) for prefix in [FUSED_PREFIX, OUT_PREFIX] for name in ["W_q", "scales", "zeros"])
    expanded_fp32_weight_nbytes = sum(
        int(torch.prod(sd[f"{prefix}.orig_shape"]).item()) * 4 for prefix in [FUSED_PREFIX, OUT_PREFIX]
    )
    ratio = init["initializer_estimated_nbytes"] / expanded_fp32_weight_nbytes if expanded_fp32_weight_nbytes else None
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "unpack_lowering": "arithmetic_floor_div_mod_no_bitshift",
        "attention_lowering": "explicit_matmul_softmax_matmul",
        "target": "single_transformer_blocks.0 core ONNX: qkv_mlp -> norm_q/k -> explicit attention -> SwiGLU -> to_out",
        "not_full_diffusers_block": True,
        "lowbit_path": str(lowbit_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "input_shape": list(x.shape),
        "output_shape": list(y_pt.shape),
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_weight_nbytes,
        "initializer_summary": init,
        "initializer_to_expanded_fp32_ratio": ratio,
        "not_folded_to_expanded_weight": bool(ratio is not None and ratio < 0.5),
        "mean_abs_error": float(np.abs(diff).mean()),
        "max_abs_error": float(np.abs(diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(y_ort, y_pt, rtol=1e-4, atol=1e-5)),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_block": report["not_full_diffusers_block"],
        "attention_lowering": report["attention_lowering"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "initializer_to_expanded_fp32_ratio": report["initializer_to_expanded_fp32_ratio"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] and report["not_folded_to_expanded_weight"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
