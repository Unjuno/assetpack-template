#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

from probe_bonsai_lowbit_rope_smoke import rope as local_pairwise_rope

OUT_DIR = Path("reports/bonsai-diffusers-rope-equivalence")


def make_cos_sin(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).view(seq_len, 1)
    dim = torch.arange(head_dim // 2, device=device, dtype=torch.float32).view(1, head_dim // 2)
    angles = pos / (10000.0 ** (2.0 * dim / float(head_dim))) * 0.01
    cos_half = torch.cos(angles).to(dtype=dtype)
    sin_half = torch.sin(angles).to(dtype=dtype)
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full = sin_half.repeat_interleave(2, dim=-1)
    return cos_half, sin_half, cos_full, sin_full


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.models.embeddings import apply_rotary_emb
        import diffusers
        import inspect
    except Exception as exc:
        report = {
            "target": "Diffusers apply_rotary_emb equivalence",
            "skipped": True,
            "skip_reason": f"diffusers_import_failed: {type(exc).__name__}: {str(exc)[:500]}",
        }
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(240000)
    x = torch.randn((1, 5, 24, 128), generator=generator, dtype=torch.float32) * 0.01
    expected = local_pairwise_rope(x)
    cos_half, sin_half, cos_full, sin_full = make_cos_sin(x.shape[1], x.shape[-1], x.device, x.dtype)

    candidates = []
    signature = str(inspect.signature(apply_rotary_emb))
    for name, emb in [
        ("tuple_half_dim", (cos_half, sin_half)),
        ("tuple_full_dim_repeat_interleave", (cos_full, sin_full)),
    ]:
        try:
            got = apply_rotary_emb(x.clone(), emb, sequence_dim=1)
            diff = got - expected
            candidates.append({
                "name": name,
                "ok": bool(torch.allclose(got, expected, rtol=1e-6, atol=1e-7)),
                "max_abs_error": float(diff.abs().max()),
                "mean_abs_error": float(diff.abs().mean()),
                "output_shape": list(got.shape),
                "error": None,
            })
        except Exception as exc:
            candidates.append({
                "name": name,
                "ok": False,
                "max_abs_error": None,
                "mean_abs_error": None,
                "output_shape": None,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            })

    matched = [c for c in candidates if c["ok"]]
    report = {
        "target": "Diffusers apply_rotary_emb equivalence against local pairwise RoPE",
        "diffusers_version": getattr(diffusers, "__version__", None),
        "apply_rotary_emb_signature": signature,
        "input_shape": list(x.shape),
        "sequence_dim": 1,
        "theta_scale": 0.01,
        "candidate_count": len(candidates),
        "matched_candidate_count": len(matched),
        "matched_candidates": [c["name"] for c in matched],
        "allclose_rtol_1e_6_atol_1e_7": bool(len(matched) > 0),
        "candidates": candidates,
        "note": "This validates the local pairwise RoPE application against Diffusers apply_rotary_emb for the tested tensor layout only. It does not yet validate Flux image_rotary_emb construction from image ids.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "diffusers_version": report["diffusers_version"],
        "matched_candidates": report["matched_candidates"],
        "allclose": report["allclose_rtol_1e_6_atol_1e_7"],
    }, indent=2))
    return 0 if report["allclose_rtol_1e_6_atol_1e_7"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
