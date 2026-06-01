#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes, recover_quantized_weight

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
TRANSFORMER_CONFIG = "transformer/config.json"
OUT_DIR = Path("reports/bonsai-recovered-transformer-checkpoint")
CHECKPOINT_DIR = OUT_DIR / "transformer"


def build_recovered_state_dict(lowbit_sd: dict[str, Any], dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    q_prefixes = set(quantized_prefixes(lowbit_sd))
    quantized_parts = {f"{prefix}.{suffix}" for prefix in q_prefixes for suffix in ["W_q", "scales", "zeros", "orig_shape", "metadata"]}
    recovered: dict[str, torch.Tensor] = {}

    for key, value in lowbit_sd.items():
        if key in quantized_parts:
            continue
        if isinstance(value, torch.Tensor):
            recovered[key] = value.detach().cpu()

    for prefix in sorted(q_prefixes):
        weight, _meta = recover_quantized_weight(lowbit_sd, prefix, output_dtype=dtype)
        recovered[f"{prefix}.weight"] = weight.detach().cpu()
    return recovered


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config_src = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=TRANSFORMER_CONFIG, repo_type="model"))
    shutil.copyfile(config_src, CHECKPOINT_DIR / "config.json")

    lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    recovered = build_recovered_state_dict(lowbit_sd, dtype=torch.float16)
    checkpoint_path = CHECKPOINT_DIR / "diffusion_pytorch_model.safetensors"
    save_file(recovered, str(checkpoint_path), metadata={"format": "pt"})

    total_numel = sum(int(t.numel()) for t in recovered.values())
    dtype_counts: dict[str, int] = {}
    for tensor in recovered.values():
        dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + int(tensor.numel())

    report = {
        "lowbit_ref": LOWBIT_REF,
        "unpacked_ref": UNPACKED_REF,
        "lowbit_path": str(lowbit_path),
        "checkpoint_root": str(OUT_DIR),
        "checkpoint_subfolder": "transformer",
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "config_path": str(CHECKPOINT_DIR / "config.json"),
        "recovered_key_count": len(recovered),
        "quantized_prefix_count": len(q_prefixes := quantized_prefixes(lowbit_sd)),
        "total_numel": total_numel,
        "dtype_numel_counts": dtype_counts,
        "keys_sample": sorted(recovered.keys())[:80],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_root": report["checkpoint_root"],
        "checkpoint_subfolder": report["checkpoint_subfolder"],
        "checkpoint_size_bytes": report["checkpoint_size_bytes"],
        "recovered_key_count": report["recovered_key_count"],
        "quantized_prefix_count": report["quantized_prefix_count"],
        "dtype_numel_counts": report["dtype_numel_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
