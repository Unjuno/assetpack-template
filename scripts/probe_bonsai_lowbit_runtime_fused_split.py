#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict, quantized_prefixes, recover_quantized_weight

OUT_DIR = Path("reports/bonsai-lowbit-runtime-fused-split")
FUSED_PATTERN = re.compile(r"^single_transformer_blocks\.\d+\.attn\.to_qkv_mlp_proj$")


class LowBitFusedLinearRuntime(torch.nn.Module):
    def __init__(self, state_dict: dict[str, Any], prefix: str):
        super().__init__()
        self.prefix = prefix
        self.wq_t = state_dict[f"{prefix}.W_q"].detach().cpu().contiguous()
        self.scales = state_dict[f"{prefix}.scales"].detach().cpu().contiguous()
        self.zeros = state_dict[f"{prefix}.zeros"].detach().cpu().contiguous()
        self.orig_shape = [int(v) for v in state_dict[f"{prefix}.orig_shape"].tolist()]
        self.metadata = [int(v) for v in state_dict[f"{prefix}.metadata"].tolist()]
        self.nbits = int(self.metadata[1])
        self.group_size = int(self.metadata[2])
        self.out_features = self.orig_shape[0]
        self.in_features = self.orig_shape[1]
        self.hidden_size = self.in_features
        self.mlp_size = self.out_features - 3 * self.hidden_size
        if self.mlp_size <= 0:
            raise ValueError(f"invalid fused qkv_mlp shape for {prefix}: {self.orig_shape}")

    def recovered_weight(self) -> torch.Tensor:
        # This correctness baseline intentionally reuses the proven recovery path.
        weight, _meta = recover_quantized_weight_for_runtime(self.wq_t, self.scales, self.zeros, self.orig_shape, self.metadata)
        return weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = self.recovered_weight().to(dtype=x.dtype, device=x.device)
        fused = torch.nn.functional.linear(x, weight)
        q, k, v, mlp = torch.split(fused, [self.hidden_size, self.hidden_size, self.hidden_size, self.mlp_size], dim=-1)
        return q, k, v, mlp


def recover_quantized_weight_for_runtime(wq_t: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, orig_shape: list[int], metadata: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    # Local arithmetic version mirrors bonsai_lowbit_recover without relying on module state.
    wq = wq_t.t().contiguous()
    nbits = int(metadata[1])
    group_size = int(metadata[2])
    out_features, in_features = int(orig_shape[0]), int(orig_shape[1])
    packed_cols = int(wq.shape[1])
    elements_per_sample = in_features // packed_cols
    shifts = torch.arange(elements_per_sample, dtype=wq.dtype) * nbits
    mask = (1 << nbits) - 1
    unpacked = ((wq.unsqueeze(-1) >> shifts) & mask).to(torch.float32).reshape(out_features, in_features)
    s = scales.t().contiguous().to(torch.float32).repeat_interleave(group_size, dim=1)
    z = zeros.t().contiguous().to(torch.float32).repeat_interleave(group_size, dim=1)
    return unpacked * s + z, {"nbits": nbits, "group_size": group_size, "orig_shape": orig_shape}


def stat(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().to(torch.float32)
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}


def check_prefix(sd: dict[str, Any], prefix: str, index: int) -> dict[str, Any]:
    module = LowBitFusedLinearRuntime(sd, prefix).eval()
    ref_weight, meta = recover_quantized_weight(sd, prefix, output_dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(30_000 + index)
    x = torch.randn((2, module.in_features), generator=generator, dtype=torch.float32) / 10.0

    with torch.inference_mode():
        q, k, v, mlp = module(x)
        fused_ref = torch.nn.functional.linear(x, ref_weight)
        q_ref, k_ref, v_ref, mlp_ref = torch.split(
            fused_ref,
            [module.hidden_size, module.hidden_size, module.hidden_size, module.mlp_size],
            dim=-1,
        )
    diffs = {
        "q": q - q_ref,
        "k": k - k_ref,
        "v": v - v_ref,
        "mlp": mlp - mlp_ref,
    }
    max_abs = max(float(d.abs().max()) for d in diffs.values())
    mean_abs = max(float(d.abs().mean()) for d in diffs.values())
    return {
        "prefix": prefix,
        "orig_shape": module.orig_shape,
        "hidden_size": module.hidden_size,
        "mlp_size": module.mlp_size,
        "metadata": meta,
        "input": stat(x),
        "outputs": {"q": stat(q), "k": stat(k), "v": stat(v), "mlp": stat(mlp)},
        "max_mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "allclose_rtol_1e_4_atol_1e_5": all(torch.allclose(out, ref, rtol=1e-4, atol=1e-5) for out, ref in [(q, q_ref), (k, k_ref), (v, v_ref), (mlp, mlp_ref)]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lowbit_path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    prefixes = [p for p in quantized_prefixes(sd) if FUSED_PATTERN.match(p)]
    results = []
    failures = []
    for index, prefix in enumerate(prefixes):
        try:
            result = check_prefix(sd, prefix, index)
            results.append(result)
            if not result["allclose_rtol_1e_4_atol_1e_5"]:
                failures.append(result)
        except Exception as exc:
            failure = {"prefix": prefix, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results.append(failure)
            failures.append(failure)

    report = {
        "source_model_ref": LOWBIT_REF,
        "uses_lowbit_source": True,
        "writes_expanded_checkpoint": False,
        "target": "single_transformer_blocks.*.attn.to_qkv_mlp_proj split into q/k/v/mlp",
        "lowbit_path": str(lowbit_path),
        "fused_projection_count": len(results),
        "failure_count": len(failures),
        "all_passed": len(failures) == 0,
        "max_abs_error": max((r.get("max_abs_error", 0.0) for r in results), default=None),
        "failures": failures[:20],
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "uses_lowbit_source": report["uses_lowbit_source"],
        "writes_expanded_checkpoint": report["writes_expanded_checkpoint"],
        "fused_projection_count": report["fused_projection_count"],
        "failure_count": report["failure_count"],
        "all_passed": report["all_passed"],
        "max_abs_error": report["max_abs_error"],
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
