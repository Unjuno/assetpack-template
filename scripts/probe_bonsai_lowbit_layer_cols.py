#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

MODEL_REF = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
LAYER = "single_transformer_blocks.0.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-layer")


def stat(x: torch.Tensor) -> dict:
    y = x.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=MODEL_REF, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    W_q = sd[f"{LAYER}.W_q"].cpu()
    scales = sd[f"{LAYER}.scales"].cpu()
    zeros = sd[f"{LAYER}.zeros"].cpu()
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]
    meta_nbits = metadata[1]
    group_size = metadata[2]

    attempts = []
    for nbits in [1, 2, 4, 8]:
        try:
            uq = unpack_cols_transposed(W_q, nbits, orig_shape[1])
            s = expand_col_groups(scales, uq.shape[1], group_size)
            z = expand_col_groups(zeros, uq.shape[1], group_size)
            if uq.shape != s.shape or uq.shape != z.shape:
                raise ValueError(f"shape mismatch uq={list(uq.shape)} s={list(s.shape)} z={list(z.shape)}")
            attempts.append({"nbits": nbits, "metadata_match": nbits == meta_nbits, "status": "passed", "unpacked": stat(uq), "minus_zero": stat((uq - z) * s), "plus_zero": stat(uq * s + z)})
        except Exception as e:
            attempts.append({"nbits": nbits, "metadata_match": nbits == meta_nbits, "status": "failed", "error_type": type(e).__name__, "error": str(e)[:500]})

    report = {"model_ref": MODEL_REF, "layer": LAYER, "orig_shape": orig_shape, "metadata": metadata, "metadata_nbits": meta_nbits, "group_size": group_size, "W_q": stat(W_q), "scales": stat(scales), "zeros": stat(zeros), "attempts": attempts, "note": "Column-group scale expansion probe; numerical correctness still needs reference validation."}
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": [(a["nbits"], a["status"]) for a in attempts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
