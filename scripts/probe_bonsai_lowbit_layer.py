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
    xf = x.detach().to(torch.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "mean": float(xf.mean()),
        "std": float(xf.std()),
    }


def unpack_rows(packed: torch.Tensor, nbits: int, output_rows: int) -> torch.Tensor:
    rows, cols = packed.shape
    if output_rows % rows != 0:
        raise ValueError(f"output_rows {output_rows} is not divisible by packed rows {rows}")
    elems = output_rows // rows
    shifts = torch.arange(elems, dtype=packed.dtype) * nbits
    mask = (1 << nbits) - 1
    y = ((packed.unsqueeze(-1) >> shifts) & mask).to(torch.float32)
    return y.permute(0, 2, 1).reshape(output_rows, cols)


def repeat_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] == rows:
        return x.to(torch.float32)
    if rows % x.shape[0] != 0:
        raise ValueError(f"cannot repeat {x.shape[0]} groups to {rows} rows")
    return x.to(torch.float32).repeat_interleave(rows // x.shape[0], dim=0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=MODEL_REF, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    sd = torch.load(path, map_location="cpu", weights_only=True)

    W_q = sd[f"{LAYER}.W_q"].cpu()
    scales = sd[f"{LAYER}.scales"].cpu()
    zeros = sd[f"{LAYER}.zeros"].cpu()
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]

    attempts = []
    for nbits in [1, 2, 4, 8]:
        try:
            unpacked = unpack_rows(W_q, nbits=nbits, output_rows=orig_shape[0])
            s = repeat_rows(scales, unpacked.shape[0])
            z = repeat_rows(zeros, unpacked.shape[0])
            if s.shape != unpacked.shape or z.shape != unpacked.shape:
                raise ValueError(f"shape mismatch unpacked={list(unpacked.shape)} scales={list(s.shape)} zeros={list(z.shape)}")
            attempts.append({
                "nbits": nbits,
                "status": "passed",
                "unpacked": stat(unpacked),
                "formula_minus_zero": stat((unpacked - z) * s),
                "formula_plus_zero": stat(unpacked * s + z),
            })
        except Exception as e:
            attempts.append({"nbits": nbits, "status": "failed", "error_type": type(e).__name__, "error": str(e)[:500]})

    report = {
        "model_ref": MODEL_REF,
        "layer": LAYER,
        "state_dict_path": str(path),
        "state_dict_size_bytes": path.stat().st_size,
        "orig_shape": orig_shape,
        "metadata": metadata,
        "W_q": stat(W_q),
        "scales": stat(scales),
        "zeros": stat(zeros),
        "attempts": attempts,
        "note": "This checks CPU unpacking candidates only; numerical correctness still needs reference validation.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layer": LAYER, "orig_shape": orig_shape, "metadata": metadata, "attempts": [(a["nbits"], a["status"]) for a in attempts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
