from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

LOWBIT_REF = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
TRANSFORMER_STATE_DICT = "transformer-gemlite-int1/state_dict.pt"


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


def load_lowbit_transformer_state_dict(model_ref: str = LOWBIT_REF) -> tuple[Path, dict[str, Any]]:
    path = Path(hf_hub_download(repo_id=model_ref, filename=TRANSFORMER_STATE_DICT, repo_type="model"))
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"expected dict state_dict, got {type(state_dict).__name__}")
    return path, state_dict


def quantized_prefixes(state_dict: dict[str, Any]) -> list[str]:
    prefixes = []
    for key in sorted(state_dict.keys()):
        if not key.endswith(".W_q"):
            continue
        prefix = key[:-4]
        required = [f"{prefix}.{name}" for name in ["W_q", "scales", "zeros", "orig_shape", "metadata"]]
        if all(k in state_dict for k in required):
            prefixes.append(prefix)
    return prefixes


def recover_quantized_weight(state_dict: dict[str, Any], prefix: str, output_dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, dict[str, Any]]:
    wq = state_dict[f"{prefix}.W_q"].cpu()
    scales = state_dict[f"{prefix}.scales"].cpu()
    zeros = state_dict[f"{prefix}.zeros"].cpu()
    orig_shape = [int(v) for v in state_dict[f"{prefix}.orig_shape"].tolist()]
    metadata = [int(v) for v in state_dict[f"{prefix}.metadata"].tolist()]
    nbits = metadata[1]
    group_size = metadata[2]
    unpacked = unpack_cols_transposed(wq, nbits, orig_shape[1])
    weight = unpacked * expand_col_groups(scales, unpacked.shape[1], group_size) + expand_col_groups(zeros, unpacked.shape[1], group_size)
    return weight.to(output_dtype), {"orig_shape": orig_shape, "metadata": metadata, "nbits": nbits, "group_size": group_size}
