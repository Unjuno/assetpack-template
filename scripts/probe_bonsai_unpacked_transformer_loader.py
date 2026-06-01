#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors import safe_open

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
TRANSFORMER_CONFIG = "transformer/config.json"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-unpacked-transformer-loader")


def dtype_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for param in model.parameters():
        counts[str(param.dtype)] = counts.get(str(param.dtype), 0) + int(param.numel())
    return counts


def key_samples(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return {
        "checkpoint_key_count": len(keys),
        "checkpoint_keys_sample": keys[:80],
        "fused_like_keys_sample": [k for k in keys if "to_qkv" in k or "modulation" in k][:40],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=TRANSFORMER_CONFIG, repo_type="model"))
    ckpt_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))

    report: dict[str, Any] = {
        "unpacked_ref": UNPACKED_REF,
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        **key_samples(ckpt_path),
    }

    model = FluxTransformer2DModel.from_pretrained(
        UNPACKED_REF,
        subfolder="transformer",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    module_keys = list(model.state_dict().keys())
    report.update(
        {
            "load_status": "passed",
            "module_class": type(model).__name__,
            "module_key_count": len(module_keys),
            "module_keys_sample": module_keys[:80],
            "split_like_module_keys_sample": [k for k in module_keys if ".to_q." in k or ".to_k." in k or ".to_v." in k or ".proj_mlp." in k][:80],
            "param_dtype_counts": dtype_counts(model),
            "param_count": int(sum(p.numel() for p in model.parameters())),
        }
    )
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "load_status": report["load_status"],
        "checkpoint_key_count": report["checkpoint_key_count"],
        "module_key_count": report["module_key_count"],
        "param_count": report["param_count"],
        "param_dtype_counts": report["param_dtype_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
