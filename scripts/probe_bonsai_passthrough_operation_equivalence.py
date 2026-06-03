#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict

UNPACKED_REF = "prism-ml/bonsai-image-binary-4B-unpacked"
UNPACKED_TRANSFORMER = "transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = Path("reports/bonsai-passthrough-operation-equivalence")


def stat(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().to(torch.float32)
    finite = torch.isfinite(y)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(finite.all()),
        "min": float(y[finite].min()) if bool(finite.any()) else None,
        "max": float(y[finite].max()) if bool(finite.any()) else None,
        "mean": float(y[finite].mean()) if bool(finite.any()) else None,
        "std": float(y[finite].std()) if int(finite.sum()) > 1 else None,
    }


def linear(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight.to(dtype=x.dtype))


def compare(name: str, low: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    diff = low.to(torch.float32) - ref.to(torch.float32)
    return {
        "name": name,
        "lowbit": stat(low),
        "reference": stat(ref),
        "shape_matches": list(low.shape) == list(ref.shape),
        "allclose_rtol_0_atol_0": bool(torch.equal(low, ref)),
        "allclose_rtol_1e_6_atol_1e_7": bool(torch.allclose(low.to(torch.float32), ref.to(torch.float32), rtol=1e-6, atol=1e-7)),
        "mean_abs_error": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
    }


def weights_from_ref(ref_path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    with safe_open(ref_path, framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key) for key in keys}


def run_components(sd: dict[str, Any], generator: torch.Generator) -> dict[str, torch.Tensor]:
    x = torch.randn((1, 2, 128), generator=generator, dtype=torch.float32) / 10.0
    ctx = torch.randn((1, 3, 7680), generator=generator, dtype=torch.float32) / 10.0
    tfeat = torch.randn((1, 256), generator=generator, dtype=torch.float32) / 10.0
    temb_seed = torch.randn((1, 3072), generator=generator, dtype=torch.float32) / 10.0
    hidden = torch.randn((1, 2, 3072), generator=generator, dtype=torch.float32) / 10.0

    x_emb = linear(sd["x_embedder.weight"], x)
    ctx_emb = linear(sd["context_embedder.weight"], ctx)
    temb = linear(sd["time_guidance_embed.timestep_embedder.linear_2.weight"], F.silu(linear(sd["time_guidance_embed.timestep_embedder.linear_1.weight"], tfeat)))
    double_img_mod = linear(sd["double_stream_modulation_img.linear.weight"], temb_seed)
    double_txt_mod = linear(sd["double_stream_modulation_txt.linear.weight"], temb_seed)
    single_mod = linear(sd["single_stream_modulation.linear.weight"], temb_seed)
    norm_out_params = linear(sd["norm_out.linear.weight"], temb_seed)
    shift, scale = norm_out_params.chunk(2, dim=-1)
    normed = F.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
    norm_out = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    proj_out = linear(sd["proj_out.weight"], norm_out)

    return {
        "x_embedder": x_emb,
        "context_embedder": ctx_emb,
        "time_guidance_timestep_mlp": temb,
        "double_stream_modulation_img": double_img_mod,
        "double_stream_modulation_txt": double_txt_mod,
        "single_stream_modulation": single_mod,
        "norm_out_params": norm_out_params,
        "norm_out_applied": norm_out,
        "proj_out": proj_out,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = Path(hf_hub_download(repo_id=UNPACKED_REF, filename=UNPACKED_TRANSFORMER, repo_type="model"))
    lowbit_path, lowbit_sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    keys = [
        "x_embedder.weight",
        "context_embedder.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.linear.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ]
    ref_sd = weights_from_ref(ref_path, keys)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(140_000)
    with torch.inference_mode():
        low_outputs = run_components(lowbit_sd, generator)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(140_000)
    with torch.inference_mode():
        ref_outputs = run_components(ref_sd, generator)

    results = [compare(name, low_outputs[name], ref_outputs[name]) for name in sorted(low_outputs)]
    failures = [r for r in results if not r["shape_matches"] or not r["allclose_rtol_0_atol_0"]]
    report = {
        "source_model_ref": LOWBIT_REF,
        "reference_model_ref": UNPACKED_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "non-quantized passthrough operation equivalence",
        "component_count": len(results),
        "failure_count": len(failures),
        "all_components_exact": len(failures) == 0,
        "max_abs_error": max((r["max_abs_error"] for r in results), default=0.0),
        "results": results,
        "failures": failures,
        "lowbit_path": str(lowbit_path),
        "reference_path": str(ref_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "component_count": report["component_count"],
        "failure_count": report["failure_count"],
        "all_components_exact": report["all_components_exact"],
        "max_abs_error": report["max_abs_error"],
    }, indent=2))
    return 0 if report["all_components_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
