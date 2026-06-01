#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
import onnxruntime as ort

MODEL_REF = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
LAYER = "single_transformer_blocks.0.attn.to_out"
OUT_DIR = Path("reports/bonsai-lowbit-layer-onnx")


def unpack_cols_transposed(wq_t: torch.Tensor, nbits: int, out_cols: int) -> torch.Tensor:
    wq = wq_t.t().contiguous()
    rows, packed_cols = wq.shape
    elems = out_cols // packed_cols
    shifts = torch.arange(elems, dtype=wq.dtype) * nbits
    mask = (1 << nbits) - 1
    return (((wq.unsqueeze(-1) >> shifts) & mask).to(torch.float32)).reshape(rows, out_cols)


def expand_col_groups(x: torch.Tensor, out_cols: int, group_size: int) -> torch.Tensor:
    return x.t().contiguous().to(torch.float32).repeat_interleave(group_size, dim=1)


def load_weight() -> torch.Tensor:
    path = Path(hf_hub_download(repo_id=MODEL_REF, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    W_q = sd[f"{LAYER}.W_q"].cpu()
    scales = sd[f"{LAYER}.scales"].cpu()
    zeros = sd[f"{LAYER}.zeros"].cpu()
    orig_shape = [int(v) for v in sd[f"{LAYER}.orig_shape"].tolist()]
    metadata = [int(v) for v in sd[f"{LAYER}.metadata"].tolist()]
    nbits = metadata[1]
    group_size = metadata[2]
    unpacked = unpack_cols_transposed(W_q, nbits, orig_shape[1])
    return unpacked * expand_col_groups(scales, unpacked.shape[1], group_size) + expand_col_groups(zeros, unpacked.shape[1], group_size)


class OneLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.linear.weight.data.copy_(weight.to(torch.float32))
        self.linear.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weight = load_weight()
    model = OneLinear(weight).eval()
    x = torch.randn(1, weight.shape[1], dtype=torch.float32) / 10.0
    with torch.no_grad():
        y_pt = model(x).detach().cpu().numpy()

    onnx_path = OUT_DIR / "single_transformer_blocks_0_attn_to_out.onnx"
    torch.onnx.export(
        model,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )

    external_data_path = Path(str(onnx_path) + ".data")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"x": x.cpu().numpy().astype(np.float32)})[0]
    diff = y_ort - y_pt
    report = {
        "model_ref": MODEL_REF,
        "layer": LAYER,
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": path_size(onnx_path),
        "external_data_path": str(external_data_path) if external_data_path.exists() else None,
        "external_data_size_bytes": path_size(external_data_path),
        "total_onnx_artifact_size_bytes": path_size(onnx_path) + path_size(external_data_path),
        "input_shape": list(x.shape),
        "output_shape": list(y_pt.shape),
        "pytorch_output_mean": float(y_pt.mean()),
        "onnx_output_mean": float(y_ort.mean()),
        "mean_abs_error": float(np.abs(diff).mean()),
        "max_abs_error": float(np.abs(diff).max()),
        "allclose_rtol_1e_4_atol_1e_5": bool(np.allclose(y_ort, y_pt, rtol=1e-4, atol=1e-5)),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
