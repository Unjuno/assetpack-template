#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

MILESTONES = ["pack_layout_probe", "metadata_download", "state_dict_inspect", "cpu_conversion_plan"]


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


def summarize_json(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, dict):
            return {"type": "dict", "keys": sorted(map(str, value.keys()))[:40], "len": len(value)}
        if isinstance(value, list):
            return {"type": "list", "len": len(value), "sample": value[:3]}
        return value
    if isinstance(value, dict):
        return {str(k): summarize_json(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, list):
        return [summarize_json(v, depth + 1) for v in value[:20]]
    return value


def download_metadata(model_ref: str, json_files: list[str], out_dir: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        "manifest.json",
        "transformer-gemlite-int1/config.json",
        "transformer-gemlite-int1/quantization_config.json",
        "transformer-gemlite-int1/gemlite_autotune.json",
        "text_encoder-hqq-4bit/config.json",
        "vae/config.json",
    ]
    selected = [p for p in wanted if p in set(json_files)]
    summaries: dict[str, Any] = {}
    copied: list[str] = []
    for path in selected:
        local = Path(hf_hub_download(repo_id=model_ref, filename=path, repo_type="model"))
        data = json.loads(local.read_text(encoding="utf-8"))
        target = meta_dir / path.replace("/", "__")
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied.append(str(target))
        summaries[path] = {
            "top_level_keys": sorted(map(str, data.keys())) if isinstance(data, dict) else None,
            "summary": summarize_json(data),
        }
    return {"metadata_files": selected, "copied_files": copied, "summaries": summaries}


def tensor_summary(value: Any) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "numel": int(value.numel())}
    if isinstance(value, dict):
        return {"kind": "dict", "len": len(value), "keys": sorted(map(str, value.keys()))[:50]}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "len": len(value), "sample_types": [type(x).__name__ for x in list(value)[:10]]}
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def inspect_state_dict(model_ref: str, out_dir: Path) -> dict[str, Any]:
    import torch
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(repo_id=model_ref, filename="transformer-gemlite-int1/state_dict.pt", repo_type="model"))
    try:
        obj = torch.load(local, map_location="cpu", weights_only=True)
        load_mode = "weights_only_true"
    except TypeError:
        obj = torch.load(local, map_location="cpu")
        load_mode = "legacy_torch_load"
    except Exception as exc:
        obj = torch.load(local, map_location="cpu", weights_only=False)
        load_mode = f"weights_only_false_after_{type(exc).__name__}"

    if isinstance(obj, dict):
        keys = sorted(map(str, obj.keys()))
        entries = []
        dtype_counts: dict[str, int] = {}
        tensor_count = 0
        tensor_numel = 0
        for key in keys[:300]:
            value = obj[key]
            summary = tensor_summary(value)
            summary["key"] = key
            entries.append(summary)
            if summary.get("kind") == "tensor":
                tensor_count += 1
                dtype = summary["dtype"]
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                tensor_numel += int(summary["numel"])
        top_kind = "dict"
    else:
        keys = []
        entries = [tensor_summary(obj)]
        dtype_counts = {}
        tensor_count = 0
        tensor_numel = 0
        top_kind = type(obj).__name__

    result = {
        "local_path": str(local),
        "file_size_bytes": local.stat().st_size,
        "file_size_mb": mb(local.stat().st_size),
        "torch_load_mode": load_mode,
        "top_kind": top_kind,
        "top_key_count": len(keys),
        "top_keys_sample": keys[:100],
        "entries_sample": entries,
        "tensor_count_sampled": tensor_count,
        "tensor_numel_sampled": tensor_numel,
        "dtype_counts_sampled": dtype_counts,
    }
    target = out_dir / "state_dict_inspect.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["copied_summary"] = str(target)
    return result


def add_stage(result: dict[str, Any], name: str, fn):
    start = time.time()
    try:
        payload = fn()
        result["stages"].append({"name": name, "status": "passed", "seconds": round(time.time() - start, 3), **payload})
        return payload
    except Exception as exc:
        result["stages"].append({"name": name, "status": "failed", "seconds": round(time.time() - start, 3), "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        raise


def conversion_plan(analysis: dict[str, Any], metadata: dict[str, Any] | None = None, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    kinds = analysis.get("by_kind", {})
    summaries = (metadata or {}).get("summaries", {})
    qcfg = summaries.get("transformer-gemlite-int1/quantization_config.json", {})
    manifest = summaries.get("manifest.json", {})
    return {
        "target": "onnxruntime-cpu",
        "source_pack_summary": {
            "has_safetensors": "safetensors" in kinds,
            "has_hqq_markers": analysis["runtime_inference"]["looks_like_hqq_pack"],
            "has_gemlite_markers": analysis["runtime_inference"]["looks_like_gemlite_pack"],
            "has_model_index_json": analysis["has_model_index_json"],
            "largest_files": analysis.get("large_files", [])[:5],
        },
        "metadata_signals": {
            "manifest_keys": manifest.get("top_level_keys"),
            "transformer_quantization_config_keys": qcfg.get("top_level_keys"),
        },
        "state_dict_signals": {
            "available": state_dict is not None,
            "top_kind": None if state_dict is None else state_dict.get("top_kind"),
            "top_key_count": None if state_dict is None else state_dict.get("top_key_count"),
            "dtype_counts_sampled": None if state_dict is None else state_dict.get("dtype_counts_sampled"),
        },
        "decision": "inspect packed tensors first; do not claim direct ONNX Runtime CPU execution of Gemlite kernels",
        "next_actions": [
            "map state_dict key schema to quantized_fqns from quantization_config.json",
            "identify packed weight tensors, scale tensors, and skipped fp16 tensors",
            "write a CPU dequantization probe for one small Gemlite INT1 linear layer",
            "compare dequantized tensor shape against matching unpacked transformer module",
            "only after dequantization is validated, attempt ONNX export of a dequantized component",
        ],
        "hard_constraints": [
            "ONNX Runtime CPU does not execute Gemlite CUDA kernels directly",
            "HQQ text encoder and Gemlite transformer are separate packed formats",
            "full text-to-image support requires transformer, text encoder, scheduler, and VAE composition",
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
        analysis = payload["analysis"]
        result["milestone_reached"] = "pack_layout_probe"
        if rank(required) <= rank("pack_layout_probe"):
            result["status"] = "passed"
            return 0

        metadata = add_stage(result, "metadata_download", lambda: download_metadata(cand["model_ref"], analysis.get("json_files", []), out_dir))
        result["milestone_reached"] = "metadata_download"
        if rank(required) <= rank("metadata_download"):
            result["status"] = "passed"
            return 0

        state_dict = add_stage(result, "state_dict_inspect", lambda: inspect_state_dict(cand["model_ref"], out_dir))
        result["milestone_reached"] = "state_dict_inspect"
        if rank(required) <= rank("state_dict_inspect"):
            result["status"] = "passed"
            return 0

        add_stage(result, "cpu_conversion_plan", lambda: {"plan": conversion_plan(analysis, metadata, state_dict)})
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
        {"path": "manifest.json", "size": 20},
        {"path": "transformer-gemlite-int1/quantization_config.json", "size": 30},
        {"path": "transformer-gemlite-int1/state_dict.pt", "size": 1024},
        {"path": "text_encoder-hqq-4bit/qmodel.pt", "size": 2048},
    ])
    assert analysis["runtime_inference"]["looks_like_gemlite_pack"] is True
    assert analysis["runtime_inference"]["looks_like_hqq_pack"] is True
    assert rank("state_dict_inspect") == 2
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
