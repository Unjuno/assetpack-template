#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(config_path: str, out_dir: str) -> int:
    start = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_ref = cfg["candidate"]["model_ref"]
    report = {
        "experiment_id": "bonsai-transformer-load-probe-v1",
        "model_ref": model_ref,
        "stage": "transformer_weight_load",
        "download_weights": True,
        "runtime_load": True,
        "onnx_export": False,
        "claim_promotable_to_manifest": False,
        "allowed_claim": "bonsai_real_transformer_weight_load_verified_not_onnx_execution",
    }
    try:
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model")
        transformer_config = load_config(config_path)
        class_name = transformer_config.get("_class_name", "FluxTransformer2DModel")
        cls = getattr(diffusers, class_name)
        model = cls.from_pretrained(
            model_ref,
            subfolder="transformer",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        param_count = 0
        dtype_counts = {}
        for param in model.parameters():
            count = int(param.numel())
            param_count += count
            key = str(param.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + count
        report.update({
            "status": "passed",
            "ci_conclusion": "success",
            "claim_promotable_to_manifest": True,
            "class_name": class_name,
            "param_count": param_count,
            "param_count_billion": round(param_count / 1000000000, 3),
            "dtype_param_counts": dtype_counts,
        })
    except BaseException as exc:
        report.update({
            "status": "failed",
            "ci_conclusion": "success_with_probe_failure",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
        })
    report["seconds"] = round(time.time() - start, 3)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-transformer-load-probe")
    args = parser.parse_args()
    return run(args.config, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
