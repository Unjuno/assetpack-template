#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

LOWBIT_REF = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
LAYER = "single_transformer_blocks.0.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-compare")


def stat(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


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


def load_lowbit_layer() -> tuple[torch.Tensor, dict]:
    path = Path(hf_hub_download(repo_id=LOWBIT_REF, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    W_q = sd[f"{LAYER}.W_q"].cpu()
    scales = sd[f"{LAYER}.scales"].cpu()
    zeros = sd[f"{LAYER}.zeros"].cpu()
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]
    nbits = metadata[1]
    group_size = metadata[2]
    unpacked = unpack_cols_transposed(W_q, nbits, orig_shape[1])
    s = expand_col_groups(scales, unpacked.shape[1], group_size)
    z = expand_col_groups(zeros, unpacked.shape[1], group_size)
    w = unpacked * s + z
    meta = {"path": str(path), "orig_shape": orig_shape, "metadata": metadata, "formula": "W_q * scale + zero"}
    return w, meta


def find_unpacked_tensor() -> tuple[torch.Tensor, dict]:
    path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename="transformer/diffusion_pytorch_model.safetensors", repo_type="model"))
    candidates = [
        f"{LAYER}.weight",
        f"{LAYER}.0.weight",
        f"{LAYER}.linear.weight",
    ]
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        chosen = None
        for c in candidates:
            if c in keys:
                chosen = c
                break
        if chosen is None:
            suffix_hits = [k for k in keys if LAYER in k and k.endswith("weight")]
            if suffix_hits:
                chosen = suffix_hits[0]
        if chosen is None:
            raise KeyError(json.dumps({"message": "matching unpacked tensor not found", "layer": LAYER, "sample_keys": keys[:80], "hits": [k for k in keys if LAYER in k][:80]}, indent=2))
        tensor = f.get_tensor(chosen).cpu()
    meta = {"path": str(path), "key": chosen}
    return tensor, meta


def cosine64(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().to(torch.float64)
    bf = b.flatten().to(torch.float64)
    return float((af @ bf) / (torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    low, low_meta = load_lowbit_layer()
    ref, ref_meta = find_unpacked_tensor()
    if list(low.shape) != list(ref.shape):
        if list(low.t().shape) == list(ref.shape):
            low = low.t().contiguous()
            low_meta["transposed_for_compare"] = True
        else:
            raise ValueError(f"shape mismatch low={list(low.shape)} ref={list(ref.shape)}")
    ref_f = ref.to(torch.float32)
    low_f = low.to(torch.float32)
    diff = low_f - ref_f
    report = {
        "lowbit_ref": LOWBIT_REF,
        "unpacked_ref": UNPACKED_REF,
        "layer": LAYER,
        "lowbit_meta": low_meta,
        "unpacked_meta": ref_meta,
        "lowbit_stats": stat(low_f),
        "unpacked_stats": stat(ref_f),
        "diff_stats": stat(diff),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "cosine_similarity_float64": cosine64(low_f, ref_f),
        "exact_equal": bool(torch.equal(low_f, ref_f)),
        "interpretation": "If exact_equal is true or MAE is near zero, this one-layer dequantization formula matches the unpacked reference.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layer": LAYER, "low_shape": report["lowbit_stats"]["shape"], "ref_shape": report["unpacked_stats"]["shape"], "mae": report["mean_abs_error"], "cosine64": report["cosine_similarity_float64"], "exact_equal": report["exact_equal"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
