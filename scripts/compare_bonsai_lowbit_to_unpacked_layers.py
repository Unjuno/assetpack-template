#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

LOWBIT_REF = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
OUT_DIR = Path("reports/bonsai-lowbit-compare-layers")
MAX_LAYERS = int(os.getenv("BONSAI_COMPARE_LAYER_LIMIT", "8"))


def unpack_cols_transposed(wq_t: torch.Tensor, nbits: int, out_cols: int) -> torch.Tensor:
    wq = wq_t.t().contiguous()
    rows, packed_cols = wq.shape
    if out_cols % packed_cols != 0:
        raise ValueError(f"out_cols={out_cols} packed_cols={packed_cols}")
    elems = out_cols // packed_cols
    shifts = torch.arange(elems, dtype=wq.dtype) * nbits
    mask = (1 << nbits) - 1
    return (((wq.unsqueeze(-1) >> shifts) & mask).to(torch.float32)).reshape(rows, out_cols)


def expand_col_groups(x: torch.Tensor, out_cols: int, group_size: int) -> torch.Tensor:
    xt = x.t().contiguous().to(torch.float32)
    if xt.shape[1] * group_size != out_cols:
        raise ValueError(f"groups={list(x.shape)} group_size={group_size} out_cols={out_cols}")
    return xt.repeat_interleave(group_size, dim=1)


def recover_weight(sd: dict, prefix: str) -> tuple[torch.Tensor, dict]:
    W_q = sd[f"{prefix}.W_q"].cpu()
    scales = sd[f"{prefix}.scales"].cpu()
    zeros = sd[f"{prefix}.zeros"].cpu()
    orig_shape = [int(v) for v in sd[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{prefix}.metadata"].tolist()]
    nbits = metadata[1]
    group_size = metadata[2]
    unpacked = unpack_cols_transposed(W_q, nbits, orig_shape[1])
    weight = unpacked * expand_col_groups(scales, unpacked.shape[1], group_size) + expand_col_groups(zeros, unpacked.shape[1], group_size)
    meta = {"orig_shape": orig_shape, "metadata": metadata, "nbits": nbits, "group_size": group_size}
    return weight, meta


def choose_prefixes(sd: dict, ref_keys: set[str]) -> list[str]:
    prefixes = []
    for key in sorted(sd.keys()):
        if not key.endswith(".W_q"):
            continue
        prefix = key[:-4]
        required = [f"{prefix}.{name}" for name in ["W_q", "scales", "zeros", "orig_shape", "metadata"]]
        if not all(k in sd for k in required):
            continue
        if f"{prefix}.weight" not in ref_keys:
            continue
        prefixes.append(prefix)
    # Deterministic spread: beginning, middle, and end of the sorted key list.
    if len(prefixes) <= MAX_LAYERS:
        return prefixes
    indices = sorted({round(i * (len(prefixes) - 1) / (MAX_LAYERS - 1)) for i in range(MAX_LAYERS)})
    return [prefixes[i] for i in indices]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    low_path = Path(hf_hub_download(repo_id=LOWBIT_REF, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename="transformer/diffusion_pytorch_model.safetensors", repo_type="model"))
    sd = torch.load(low_path, map_location="cpu", weights_only=True)

    results = []
    with safe_open(ref_path, framework="pt", device="cpu") as ref_file:
        prefixes = choose_prefixes(sd, set(ref_file.keys()))
        for prefix in prefixes:
            low_weight, meta = recover_weight(sd, prefix)
            ref_weight = ref_file.get_tensor(f"{prefix}.weight").cpu().to(torch.float32)
            transposed = False
            if list(low_weight.shape) != list(ref_weight.shape):
                if list(low_weight.t().shape) == list(ref_weight.shape):
                    low_weight = low_weight.t().contiguous()
                    transposed = True
                else:
                    raise ValueError(f"shape mismatch prefix={prefix} low={list(low_weight.shape)} ref={list(ref_weight.shape)}")
            diff = low_weight.to(torch.float32) - ref_weight
            results.append({
                "prefix": prefix,
                "shape": list(ref_weight.shape),
                "transposed_for_compare": transposed,
                "metadata": meta,
                "mean_abs_error": float(diff.abs().mean()),
                "max_abs_error": float(diff.abs().max()),
                "exact_equal": bool(torch.equal(low_weight.to(torch.float32), ref_weight)),
            })
            del low_weight, ref_weight, diff

    exact_count = sum(1 for r in results if r["exact_equal"])
    report = {
        "lowbit_ref": LOWBIT_REF,
        "unpacked_ref": UNPACKED_REF,
        "lowbit_path": str(low_path),
        "unpacked_path": str(ref_path),
        "sampled_layers": len(results),
        "exact_equal_count": exact_count,
        "all_exact_equal": exact_count == len(results),
        "max_mean_abs_error": max((r["mean_abs_error"] for r in results), default=None),
        "max_abs_error": max((r["max_abs_error"] for r in results), default=None),
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampled_layers": report["sampled_layers"], "all_exact_equal": report["all_exact_equal"], "max_abs_error": report["max_abs_error"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
