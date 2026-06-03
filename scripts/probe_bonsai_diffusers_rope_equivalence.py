#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from probe_bonsai_lowbit_rope_smoke import rope as local_pairwise_rope

OUT_DIR = Path("reports/bonsai-diffusers-rope-equivalence")


def make_cos_sin(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, scale: float = 0.01):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).view(seq_len, 1)
    dim = torch.arange(head_dim // 2, device=device, dtype=torch.float32).view(1, head_dim // 2)
    angles = pos / (10000.0 ** (2.0 * dim / float(head_dim))) * scale
    cos_half = torch.cos(angles).to(dtype=dtype)
    sin_half = torch.sin(angles).to(dtype=dtype)
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full = sin_half.repeat_interleave(2, dim=-1)
    return cos_half, sin_half, cos_full, sin_full


def compare_pair(name, a, b, rtol=1e-6, atol=1e-7):
    d = a - b
    return {
        "name": name,
        "ok": bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs_error": float(d.abs().max()),
        "mean_abs_error": float(d.abs().mean()),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
        import diffusers
        import inspect
    except Exception as exc:
        report = {"target": "Diffusers RoPE equivalence", "skipped": True, "skip_reason": f"diffusers_import_failed: {type(exc).__name__}: {str(exc)[:500]}"}
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2)); return 1

    generator = torch.Generator(device="cpu"); generator.manual_seed(240000)
    x = torch.randn((1, 5, 24, 128), generator=generator, dtype=torch.float32) * 0.01
    expected = local_pairwise_rope(x)
    cos_half, sin_half, cos_full, sin_full = make_cos_sin(x.shape[1], x.shape[-1], x.device, x.dtype)

    apply_candidates = []
    for name, emb in [("tuple_half_dim", (cos_half, sin_half)), ("tuple_full_dim_repeat_interleave", (cos_full, sin_full))]:
        try:
            got = apply_rotary_emb(x.clone(), emb, sequence_dim=1)
            c = compare_pair(name, got, expected)
            c["output_shape"] = list(got.shape); c["error"] = None
            apply_candidates.append(c)
        except Exception as exc:
            apply_candidates.append({"name": name, "ok": False, "max_abs_error": None, "mean_abs_error": None, "output_shape": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

    pos_default = torch.arange(x.shape[1], dtype=torch.float32)
    pos_scaled = pos_default * 0.01
    frequency_candidates = []
    for name, pos in [("diffusers_pos_default", pos_default), ("diffusers_pos_scaled_0_01", pos_scaled)]:
        try:
            dc, ds = get_1d_rotary_pos_embed(dim=x.shape[-1], pos=pos, theta=10000, use_real=True, repeat_interleave_real=True)
            frequency_candidates.append(compare_pair(name + "_cos", dc.to(cos_full.dtype), cos_full))
            frequency_candidates.append(compare_pair(name + "_sin", ds.to(sin_full.dtype), sin_full))
        except Exception as exc:
            frequency_candidates.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

    matched_apply = [c for c in apply_candidates if c.get("ok")]
    matched_freq = [c for c in frequency_candidates if c.get("ok")]
    report = {
        "target": "Diffusers RoPE application and frequency construction equivalence",
        "diffusers_version": getattr(diffusers, "__version__", None),
        "apply_rotary_emb_signature": str(inspect.signature(apply_rotary_emb)),
        "get_1d_rotary_pos_embed_signature": str(inspect.signature(get_1d_rotary_pos_embed)),
        "input_shape": list(x.shape),
        "sequence_dim": 1,
        "local_theta_scale": 0.01,
        "apply_matched_candidates": [c["name"] for c in matched_apply],
        "frequency_matched_candidates": [c["name"] for c in matched_freq],
        "apply_allclose_rtol_1e_6_atol_1e_7": bool(matched_apply),
        "frequency_allclose_rtol_1e_6_atol_1e_7": bool(matched_freq),
        "apply_candidates": apply_candidates,
        "frequency_candidates": frequency_candidates,
        "note": "Frequency check compares local cos/sin construction against Diffusers get_1d_rotary_pos_embed for default and scaled positions. It still does not validate full Flux text/image id concatenation.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"apply_ok": report["apply_allclose_rtol_1e_6_atol_1e_7"], "frequency_ok": report["frequency_allclose_rtol_1e_6_atol_1e_7"], "frequency_matches": report["frequency_matched_candidates"]}, indent=2))
    return 0 if report["apply_allclose_rtol_1e_6_atol_1e_7"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
