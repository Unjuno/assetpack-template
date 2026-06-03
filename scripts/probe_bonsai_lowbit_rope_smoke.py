#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_lowbit_modulated_block_cores import explicit_attention, layer_norm_modulated, lb, modulation_chunks, ref_linear, rms_norm_per_head, stat, swiglu

OUT_DIR = Path("reports/bonsai-lowbit-rope-smoke")


def rope(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    pos = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).view(1, x.shape[1], 1, 1)
    dim = torch.arange(d // 2, device=x.device, dtype=torch.float32).view(1, 1, 1, -1)
    ang = pos / (10000.0 ** (2.0 * dim / float(d))) * 0.01
    c = torch.cos(ang).to(dtype=x.dtype)
    s = torch.sin(ang).to(dtype=x.dtype)
    a = x[..., 0::2]
    b = x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


def run_block(sd, img, txt, temb, runtime: bool):
    p = "transformer_blocks.0"
    im = modulation_chunks(sd, "double_stream_modulation_img.linear.weight", temb, 6)
    tx = modulation_chunks(sd, "double_stream_modulation_txt.linear.weight", temb, 6)
    img_in = layer_norm_modulated(img, im[0], im[1])
    txt_in = layer_norm_modulated(txt, tx[0], tx[1])
    lin = lb if runtime else ref_linear
    qi = lin(sd, f"{p}.attn.to_q", img_in)
    ki = lin(sd, f"{p}.attn.to_k", img_in)
    vi = lin(sd, f"{p}.attn.to_v", img_in)
    qt = lin(sd, f"{p}.attn.add_q_proj", txt_in)
    kt = lin(sd, f"{p}.attn.add_k_proj", txt_in)
    vt = lin(sd, f"{p}.attn.add_v_proj", txt_in)
    h = img.shape[-1]
    hd = int(sd[f"{p}.attn.norm_q.weight"].numel())
    heads = h // hd
    def shp(x):
        return x.view(x.shape[0], x.shape[1], heads, hd)
    qi = rms_norm_per_head(shp(qi), sd[f"{p}.attn.norm_q.weight"])
    ki = rms_norm_per_head(shp(ki), sd[f"{p}.attn.norm_k.weight"])
    qt = rms_norm_per_head(shp(qt), sd[f"{p}.attn.norm_added_q.weight"])
    kt = rms_norm_per_head(shp(kt), sd[f"{p}.attn.norm_added_k.weight"])
    q = rope(torch.cat([qt, qi], dim=1))
    k = rope(torch.cat([kt, ki], dim=1))
    v = torch.cat([shp(vt), shp(vi)], dim=1)
    y = explicit_attention(q, k, v, h, hd)
    yt, yi = y.split([txt.shape[1], img.shape[1]], dim=1)
    img = img + im[2].unsqueeze(1) * lin(sd, f"{p}.attn.to_out.0", yi)
    txt = txt + tx[2].unsqueeze(1) * lin(sd, f"{p}.attn.to_add_out", yt)
    img = img + im[5].unsqueeze(1) * lin(sd, f"{p}.ff.linear_out", swiglu(lin(sd, f"{p}.ff.linear_in", layer_norm_modulated(img, im[3], im[4]))))
    txt = txt + tx[5].unsqueeze(1) * lin(sd, f"{p}.ff_context.linear_out", swiglu(lin(sd, f"{p}.ff_context.linear_in", layer_norm_modulated(txt, tx[3], tx[4]))))
    return img, txt


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    h = [int(v) for v in sd["transformer_blocks.0.attn.to_q.orig_shape"].tolist()][1]
    g = torch.Generator(device="cpu")
    g.manual_seed(170000)
    img = torch.randn((1, 2, h), generator=g) * 0.01
    txt = torch.randn((1, 3, h), generator=g) * 0.01
    temb = torch.randn((1, h), generator=g) * 0.01
    with torch.inference_mode():
        ir, tr = run_block(sd, img, txt, temb, True)
        ie, te = run_block(sd, img, txt, temb, False)
    di, dt = ir - ie, tr - te
    ok = torch.allclose(ir, ie, rtol=1e-4, atol=1e-5) and torch.allclose(tr, te, rtol=1e-4, atol=1e-5)
    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "single double-block modulated attention with pairwise rotary embedding",
        "not_full_diffusers_transformer": True,
        "double_block_count": 1,
        "single_block_count": 0,
        "outputs": {"img": stat(ir), "txt": stat(tr)},
        "max_abs_error": max(float(di.abs().max()), float(dt.abs().max())),
        "mean_abs_error": max(float(di.abs().mean()), float(dt.abs().mean())),
        "allclose_rtol_1e_4_atol_1e_5": bool(ok),
        "lowbit_path": str(path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"uses_lowbit_source": True, "writes_expanded_checkpoint": False, "double_block_count": 1, "allclose": bool(ok), "max_abs_error": report["max_abs_error"]}, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
