#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

PROBE_PATHS = [
    "transformer/config.json",
    "transformer/model.safetensors.index.json",
    "transformer/model.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors",
]


def url(model_ref: str, path: str) -> str:
    encoded = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://huggingface.co/{model_ref}/resolve/main/{encoded}"


def head(model_ref: str, path: str) -> dict:
    req = urllib.request.Request(url(model_ref, path), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("content-length")
            return {
                "path": path,
                "present": True,
                "status_code": resp.status,
                "content_length": int(length) if length and length.isdigit() else None,
                "etag": resp.headers.get("etag"),
                "x_linked_size": resp.headers.get("x-linked-size"),
            }
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"path": path, "present": False, "status_code": 404}
        return {"path": path, "present": False, "status_code": exc.code, "error": str(exc)[:1000]}


def run(config_path: str, out_dir: str) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_ref = cfg["candidate"]["model_ref"]
    start = time.time()
    probes = [head(model_ref, path) for path in PROBE_PATHS]
    present = [item for item in probes if item.get("present")]
    has_config = any(item["path"] == "transformer/config.json" for item in present)
    has_weight_or_index = any(item["path"].endswith((".safetensors", ".index.json")) and item["path"] != "transformer/config.json" for item in present)
    status = "passed" if has_config and has_weight_or_index else "failed"
    report = {
        "experiment_id": "bonsai-transformer-weight-metadata-probe-v1",
        "model_ref": model_ref,
        "status": status,
        "ci_conclusion": "success" if status == "passed" else "failure",
        "claim_promotable_to_manifest": status == "passed",
        "allowed_claim": "bonsai_transformer_weight_metadata_boundary_verified_not_runtime_load_not_onnx_execution",
        "download_weights": False,
        "runtime_load": False,
        "onnx_export": False,
        "present_paths": [item["path"] for item in present],
        "probes": probes,
        "seconds": round(time.time() - start, 3),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if status == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-transformer-weight-metadata-probe")
    args = parser.parse_args()
    return run(args.config, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
