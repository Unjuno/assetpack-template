#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from export_bonsai_lowbit_double_block_cores_onnx import LowBitDoubleBlockCoreOnnx, block_quantized_prefixes as double_block_quantized_prefixes
from export_bonsai_lowbit_single_block_core_onnx import LowBitSingleBlockCoreOnnx, initializer_summary, tensor_nbytes

OUT_DIR = Path("reports/bonsai-lowbit-transformer-core-stack-onnx")
DOUBLE_BLOCK_COUNT = int(os.getenv("BONSAI_CORE_STACK_ONNX_DOUBLE_BLOCKS", "1"))
SINGLE_BLOCK_COUNT = int(os.getenv("BONSAI_CORE_STACK_ONNX_SINGLE_BLOCKS", "2"))
INPUT_SCALE = float(os.getenv("BONSAI_CORE_STACK_INPUT_SCALE", "0.01"))
RESIDUAL_SCALE = float(os.getenv("BONSAI_CORE_STACK_RESIDUAL_SCALE", "0.001"))


class LowBitTransformerCoreStackOnnx(torch.nn.Module):
    def __init__(self, sd: dict[str, Any], double_blocks: int, single_blocks: int, residual_scale: float):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([LowBitDoubleBlockCoreOnnx(sd, index) for index in range(double_blocks)])
        self.single_blocks = torch.nn.ModuleList([LowBitSingleBlockCoreOnnx(sd) for _ in range(single_blocks)])
        # Recreate each single block with its own index. ModuleList comprehension above cannot pass index
        # through imported class, so replace it explicitly.
        self.single_blocks = torch.nn.ModuleList([LowBitSingleBlockCoreOnnxForIndex(sd, index) for index in range(single_blocks)])
        self.double_block_count = int(double_blocks)
        self.single_block_count = int(single_blocks)
        self.residual_scale = float(residual_scale)

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_state = img
        txt_state = txt
        for block in self.double_blocks:
            img_delta, txt_delta = block(img_state, txt_state)
            img_state = img_state + img_delta * self.residual_scale
            txt_state = txt_state + txt_delta * self.residual_scale
        hidden = torch.cat([txt_state, img_state], dim=1)
        for block in self.single_blocks:
            hidden = hidden + block(hidden) * self.residual_scale
        txt_out, img_out = torch.split(hidden, [txt_state.shape[1], img_state.shape[1]], dim=1)
        return img_out, txt_out, hidden


class LowBitSingleBlockCoreOnnxForIndex(LowBitSingleBlockCoreOnnx):
    def __init__(self, sd: dict[str, Any], block_index: int):
        # The imported class is fixed to block 0 by construction, so reproduce the same logic
        # by temporarily mapping the expected block-0 keys is not safe. Instead this subclass
        # creates the needed components directly.
        torch.nn.Module.__init__(self)
        from export_bonsai_lowbit_single_block_core_onnx import LowBitLinearOnnx

        block_prefix = f"single_transformer_blocks.{block_index}"
        fused_prefix = f"{block_prefix}.attn.to_qkv_mlp_proj"
        out_prefix = f"{block_prefix}.attn.to_out"
        norm_q_key = f"{block_prefix}.attn.norm_q.weight"
        norm_k_key = f"{block_prefix}.attn.norm_k.weight"
        self.fused = LowBitLinearOnnx(sd, fused_prefix)
        self.to_out = LowBitLinearOnnx(sd, out_prefix)
        self.register_buffer("norm_q_weight", sd[norm_q_key].detach().cpu().to(torch.float32), persistent=True)
        self.register_buffer("norm_k_weight", sd[norm_k_key].detach().cpu().to(torch.float32), persistent=True)
        self.hidden_size = self.fused.in_features
        self.head_dim = int(self.norm_q_weight.numel())
        self.heads = self.hidden_size // self.head_dim
        self.mlp_size = self.fused.out_features - 3 * self.hidden_size
        if self.mlp_size <= 0:
            raise ValueError(f"invalid single block core shape: block={block_index}")


def single_block_quantized_prefixes(block_index: int) -> list[str]:
    prefix = f"single_transformer_blocks.{block_index}"
    return [f"{prefix}.attn.to_qkv_mlp_proj", f"{prefix}.attn.to_out"]


def included_quantized_prefixes(double_blocks: int, single_blocks: int) -> list[str]:
    prefixes: list[str] = []
    for index in range(double_blocks):
        prefixes.extend(double_block_quantized_prefixes(index))
    for index in range(single_blocks):
        prefixes.extend(single_block_quantized_prefixes(index))
    return prefixes


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    hidden_size = [int(v) for v in sd["transformer_blocks.0.attn.to_q.orig_shape"].tolist()][1]
    module = LowBitTransformerCoreStackOnnx(sd, DOUBLE_BLOCK_COUNT, SINGLE_BLOCK_COUNT, RESIDUAL_SCALE).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(130_000)
    img = torch.randn((1, 2, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE
    txt = torch.randn((1, 3, hidden_size), generator=generator, dtype=torch.float32) * INPUT_SCALE
    with torch.inference_mode():
        img_pt, txt_pt, hidden_pt = [t.detach().cpu().numpy() for t in module(img, txt)]

    onnx_path = OUT_DIR / "transformer_core_stack_sample.onnx"
    started = time.time()
    torch.onnx.export(
        module,
        (img, txt),
        str(onnx_path),
        input_names=["img", "txt"],
        output_names=["img_out", "txt_out", "hidden"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes={
            "img": {0: "batch", 1: "img_seq"},
            "txt": {0: "batch", 1: "txt_seq"},
            "img_out": {0: "batch", 1: "img_seq"},
            "txt_out": {0: "batch", 1: "txt_seq"},
            "hidden": {0: "batch", 1: "joint_seq"},
        },
    )
    export_seconds = round(time.time() - started, 3)

    started = time.time()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img_ort, txt_ort, hidden_ort = session.run(None, {"img": img.cpu().numpy().astype(np.float32), "txt": txt.cpu().numpy().astype(np.float32)})
    ort_seconds = round(time.time() - started, 3)

    init = initializer_summary(onnx_path)
    prefixes = included_quantized_prefixes(DOUBLE_BLOCK_COUNT, SINGLE_BLOCK_COUNT)
    packed_nbytes = sum(tensor_nbytes(sd[f"{prefix}.{name}"]) for prefix in prefixes for name in ["W_q", "scales", "zeros"])
    expanded_fp32_weight_nbytes = sum(int(torch.prod(sd[f"{prefix}.orig_shape"]).item()) * 4 for prefix in prefixes)
    ratio = init["initializer_estimated_nbytes"] / expanded_fp32_weight_nbytes if expanded_fp32_weight_nbytes else None

    img_diff = img_ort - img_pt
    txt_diff = txt_ort - txt_pt
    hidden_diff = hidden_ort - hidden_pt
    allclose = bool(
        np.allclose(img_ort, img_pt, rtol=1e-4, atol=1e-5)
        and np.allclose(txt_ort, txt_pt, rtol=1e-4, atol=1e-5)
        and np.allclose(hidden_ort, hidden_pt, rtol=1e-4, atol=1e-5)
    )
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "constant_folding_disabled": True,
        "target": "sampled transformer core stack ONNX",
        "not_full_diffusers_transformer": True,
        "double_block_count": DOUBLE_BLOCK_COUNT,
        "single_block_count": SINGLE_BLOCK_COUNT,
        "input_scale": INPUT_SCALE,
        "residual_scale": RESIDUAL_SCALE,
        "onnx_size_bytes": onnx_path.stat().st_size,
        "export_seconds": export_seconds,
        "ort_seconds": ort_seconds,
        "packed_nbytes": packed_nbytes,
        "expanded_fp32_weight_nbytes": expanded_fp32_weight_nbytes,
        "initializer_summary": init,
        "initializer_to_expanded_fp32_ratio": ratio,
        "not_folded_to_expanded_weight": bool(ratio is not None and ratio < 0.5),
        "mean_abs_error": max(float(np.abs(img_diff).mean()), float(np.abs(txt_diff).mean()), float(np.abs(hidden_diff).mean())),
        "max_abs_error": max(float(np.abs(img_diff).max()), float(np.abs(txt_diff).max()), float(np.abs(hidden_diff).max())),
        "allclose_rtol_1e_4_atol_1e_5": allclose,
        "lowbit_path": str(lowbit_path),
    }
    try:
        onnx_path.unlink()
    except OSError:
        pass
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "not_full_diffusers_transformer": report["not_full_diffusers_transformer"],
        "double_block_count": report["double_block_count"],
        "single_block_count": report["single_block_count"],
        "allclose": report["allclose_rtol_1e_4_atol_1e_5"],
        "max_abs_error": report["max_abs_error"],
        "initializer_to_expanded_fp32_ratio": report["initializer_to_expanded_fp32_ratio"],
        "export_seconds": report["export_seconds"],
        "ort_seconds": report["ort_seconds"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_4_atol_1e_5"] and report["not_folded_to_expanded_weight"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
