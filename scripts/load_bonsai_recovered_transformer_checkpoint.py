#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel

from build_bonsai_recovered_transformer_checkpoint import OUT_DIR, CHECKPOINT_DIR, main as build_checkpoint


def dtype_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for param in model.parameters():
        counts[str(param.dtype)] = counts.get(str(param.dtype), 0) + int(param.numel())
    return counts


def main() -> int:
    build_checkpoint()
    model = FluxTransformer2DModel.from_pretrained(
        str(CHECKPOINT_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    state_keys = list(model.state_dict().keys())
    report = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "load_status": "passed",
        "module_class": type(model).__name__,
        "module_key_count": len(state_keys),
        "module_keys_sample": state_keys[:80],
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "param_dtype_counts": dtype_counts(model),
    }
    (OUT_DIR / "load_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "load_status": report["load_status"],
        "module_key_count": report["module_key_count"],
        "param_count": report["param_count"],
        "param_dtype_counts": report["param_dtype_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
