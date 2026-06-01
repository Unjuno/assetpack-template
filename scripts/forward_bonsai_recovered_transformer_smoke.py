#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel

from load_bonsai_recovered_transformer_checkpoint import CHECKPOINT_DIR, OUT_DIR, build_checkpoint

FORWARD_REPORT = OUT_DIR / "forward_report.json"


def cfg_get(config, name: str, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def tensor_stats(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()) if y.numel() > 1 else 0.0,
    }


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            out = first_tensor(item)
            if out is not None:
                return out
    if hasattr(value, "sample") and isinstance(value.sample, torch.Tensor):
        return value.sample
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_checkpoint()

    model = FluxTransformer2DModel.from_pretrained(
        str(CHECKPOINT_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    config = model.config
    in_channels = int(cfg_get(config, "in_channels"))
    joint_attention_dim = int(cfg_get(config, "joint_attention_dim"))
    pooled_projection_dim = int(cfg_get(config, "pooled_projection_dim"))
    guidance_embeds = bool(cfg_get(config, "guidance_embeds", False))

    batch = 1
    image_seq_len = 1
    text_seq_len = 1
    dtype = torch.float16

    hidden_states = torch.zeros((batch, image_seq_len, in_channels), dtype=dtype)
    encoder_hidden_states = torch.zeros((batch, text_seq_len, joint_attention_dim), dtype=dtype)
    pooled_projections = torch.zeros((batch, pooled_projection_dim), dtype=dtype)
    timestep = torch.zeros((batch,), dtype=dtype)
    img_ids = torch.zeros((image_seq_len, 3), dtype=dtype)
    txt_ids = torch.zeros((text_seq_len, 3), dtype=dtype)
    guidance = torch.zeros((batch,), dtype=dtype) if guidance_embeds else None

    started = time.time()
    with torch.inference_mode():
        kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": False,
        }
        if guidance is not None:
            kwargs["guidance"] = guidance
        output = model(**kwargs)
    seconds = round(time.time() - started, 3)
    out_tensor = first_tensor(output)
    if out_tensor is None:
        raise RuntimeError(f"could not find tensor output in {type(output).__name__}")

    report = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "forward_status": "passed",
        "seconds": seconds,
        "module_class": type(model).__name__,
        "config": {
            "in_channels": in_channels,
            "joint_attention_dim": joint_attention_dim,
            "pooled_projection_dim": pooled_projection_dim,
            "guidance_embeds": guidance_embeds,
        },
        "inputs": {
            "hidden_states": tensor_stats(hidden_states),
            "encoder_hidden_states": tensor_stats(encoder_hidden_states),
            "pooled_projections": tensor_stats(pooled_projections),
            "timestep": tensor_stats(timestep),
            "img_ids": tensor_stats(img_ids),
            "txt_ids": tensor_stats(txt_ids),
            "guidance": None if guidance is None else tensor_stats(guidance),
        },
        "output": tensor_stats(out_tensor),
    }
    FORWARD_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "forward_status": report["forward_status"],
        "seconds": report["seconds"],
        "output": report["output"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
