#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

MILESTONES = ["pack_layout_probe", "metadata_download", "cpu_conversion_plan"]


def rank(name: str) -> int:
    if name not in MILESTONES:
        raise ValueError(f"unknown milestone: {name}")
    return MILESTONES.index(name)


def mb(size: int | None) -> float | None:
    return None if size is None else round(size / 1024 / 1024, 3)


def hf_files(model_ref: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    try:
        for item in api.list_repo_tree(repo_id=model_ref, repo_type="model", recursive=True):
            path = getattr(item, "path", None) or getattr(item, "rfilename", None)
            if not path or getattr(item, "type", None) == "directory":
                continue
            out.append({"path": path, "size": getattr(item, "size", None), "lfs": bool(getattr(item, "lfs", None))})
    except Exception:
        for path in api.list_repo_files(repo_id=model_ref, repo_type="model"):
            out.append({"path": path, "size": None, "lfs": None})
    return sorted(out, key=lambda x: x["path"])


def classify(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return "safetensors"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".pt") or lower.endswith(".pth") or lower.endswith(".bin"):
        return "torch_or_binary"
    if lower.endswith(".hqq") or "hqq" in lower:
        return "hqq"
    if "gemlite" in lower:
        return "gemlite"
    if lower.endswith(".txt") or lower.endswith(".md"):
        return "text"
    return "other"


def analyze(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [f["path"] for f in files]
    by_kind: dict[str, dict[str, Any]] = {}
    for f in files:
        kind = classify(f["path"])
        bucket = by_kind.setdefault(kind, {"count": 0, "size_bytes": 0, "files": []})
        bucket["count"] += 1
        bucket["size_bytes"] += f.get("size") or 0
        if len(bucket["files"]) < 80:
            bucket["files"].append({"path": f["path"], "size_bytes": f.get("size"), "size_mb": mb(f.get("size"))})

    top_level = sorted({p.split("/", 1)[0] for p in paths})
    large_files = sorted(
        [{"path": f["path"], "size_bytes": f.get("size"), "size_mb": mb(f.get("size")), "kind": classify(f["path"])} for f in files],
        key=lambda x: x["size_bytes"] or 0,
        reverse=True,
    )[:30]

    for bucket in by_kind.values():
        bucket["size_mb"] = mb(bucket["size_bytes"])

    return {
        "file_count": len(paths),
        "top_level_entries": top_level,
        "by_kind": by_kind,
        "large_files": large_files,
        "has_model_index_json": "model_index.json" in set(paths),
        "json_files": sorted(p for p in paths if p.endswith(".json"))[:100],
        "runtime_inference": {
            "looks_like_gemlite_pack": any("gemlite" in p.lower() for p in paths) or "gemlite" in " ".join(top_level).lower(),
            "looks_like_hqq_pack": any("hqq" in p.lower() for p in paths),
            "looks_like_standard_diffusers_pipeline": "model_index.json" in set(paths),
            "direct_onnxruntime_cpu_likelihood": "low unless weights can be dequantized or represented with standard ONNX ops",
        },
    }


def add_stage(result: dict[str, Any], name: str, fn):
    start = time.time()
    try:
        payload = fn()
        result["stages"].append({"name": name, "status": "passed", "seconds": round(time.time() - start, 3), **payload})
        return payload
    except Exception as exc:
        result["stages"].append({"name": name, "status": "failed", "seconds": round(time.time() - start, 3), "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        raise


def conversion_plan(analysis: dict[str, Any]) -> dict[str, Any]:
    kinds = analysis.get("by_kind", {})
    return {
        "target": "onnxruntime-cpu",
        "source_pack_summary": {
            "has_safetensors": "safetensors" in kinds,
            "has_hqq_markers": analysis["runtime_inference"]["looks_like_hqq_pack"],
            "has_gemlite_markers": analysis["runtime_inference"]["looks_like_gemlite_pack"],
            "has_model_index_json": analysis["has_model_index_json"],
        },
        "next_actions": [
            "download small json/config files only and inspect quantization metadata",
            "identify exact packed tensor file format and loader class",
            "check whether dequantized torch modules can be materialized without CUDA",
            "if CPU materialization works, export dequantized component to ONNX as a correctness baseline",
            "if CPU materialization does not work, implement or reuse a pack reader before ONNX export",
        ],
        "hard_constraints": [
            "ONNX Runtime CPU does not execute Gemlite CUDA kernels directly",
            "a successful low-bit ONNX path needs either dequantization to standard tensors or custom ONNX-compatible low-bit operators",
            "do not claim full text-to-image support until transformer, text encoder, scheduler, and VAE are composed",
        ],
    }


def run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cand = cfg["candidate"]
    probe = cfg.get("probe", {})
    required = probe.get("required_milestone", "pack_layout_probe")
    rank(required)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment_id": cfg.get("id"),
        "model_id": cand.get("id"),
        "model_ref": cand.get("model_ref"),
        "base_model_ref": cand.get("base_model_ref"),
        "backend": cand.get("backend"),
        "status": "failed",
        "required_milestone": required,
        "milestone_reached": None,
        "stages": [],
    }
    start = time.time()
    try:
        payload = add_stage(result, "pack_layout_probe", lambda: {"analysis": analyze(hf_files(cand["model_ref"]))})
        result["milestone_reached"] = "pack_layout_probe"
        if rank(required) <= rank("pack_layout_probe"):
            result["status"] = "passed"
            return 0

        plan = add_stage(result, "cpu_conversion_plan", lambda: {"plan": conversion_plan(payload["analysis"])})
        result["milestone_reached"] = "cpu_conversion_plan"
        if rank(required) <= rank("cpu_conversion_plan"):
            result["status"] = "passed"
            return 0

        result["status"] = "passed"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        return 1
    finally:
        result["seconds"] = round(time.time() - start, 3)
        (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_test() -> int:
    analysis = analyze([
        {"path": "README.md", "size": 10},
        {"path": "transformer/gemlite_qlinear.pt", "size": 1024},
        {"path": "text_encoder/hqq_config.json", "size": 20},
    ])
    assert analysis["runtime_inference"]["looks_like_gemlite_pack"] is True
    assert analysis["runtime_inference"]["looks_like_hqq_pack"] is True
    assert rank("pack_layout_probe") == 0
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-lowbit-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-lowbit-smoke")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
