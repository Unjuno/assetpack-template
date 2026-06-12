#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

MILESTONES = ["layout_probe", "component_load", "component_export", "ort_cpu_forward", "cpu_image_generation"]
BOUNDARY_PROBE_PATHS = [
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/tokenizer_config.json",
    "scheduler/scheduler_config.json",
]


def rank(name: str) -> int:
    if name not in MILESTONES:
        raise ValueError(f"unknown milestone: {name}")
    return MILESTONES.index(name)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def is_external_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def mb(size: int | None) -> float | None:
    return None if size is None else round(size / 1024 / 1024, 3)


def analyze(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [f["path"] for f in files]
    path_set = set(paths)
    safes = [
        {"path": f["path"], "size_bytes": f.get("size"), "size_mb": mb(f.get("size"))}
        for f in files
        if f["path"].endswith(".safetensors")
    ]
    components = {}
    for name in ["transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"]:
        selected = sorted(p for p in paths if p == name or p.startswith(name + "/"))
        components[name] = {
            "present": bool(selected),
            "file_count": len(selected),
            "has_config": f"{name}/config.json" in path_set or f"{name}/tokenizer_config.json" in path_set or f"{name}/scheduler_config.json" in path_set,
            "sample_files": selected[:30],
        }
    return {
        "file_count": len(paths),
        "top_level_entries": sorted({p.split("/", 1)[0] for p in paths}),
        "has_model_index_json": "model_index.json" in path_set,
        "configs": sorted(p for p in paths if p.endswith("config.json") or p.endswith("model_index.json"))[:100],
        "safetensors_count": len(safes),
        "safetensors_total_mb": round(sum(s["size_bytes"] or 0 for s in safes) / 1024 / 1024, 3),
        "safetensors": safes[:100],
        "component_candidates": components,
    }


def hf_files(model_ref: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    try:
        items = api.list_repo_tree(repo_id=model_ref, repo_type="model", recursive=True)
        for item in items:
            path = getattr(item, "path", None) or getattr(item, "rfilename", None)
            if not path or getattr(item, "type", None) == "directory":
                continue
            out.append({"path": path, "size": getattr(item, "size", None)})
    except Exception:
        for path in api.list_repo_files(repo_id=model_ref, repo_type="model"):
            out.append({"path": path, "size": None})
    return sorted(out, key=lambda x: x["path"])


def resolve_url(model_ref: str, path: str) -> str:
    encoded_model = urllib.parse.quote(model_ref, safe="/")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{encoded_model}/resolve/main/{encoded_path}"


def probe_resolve_path(model_ref: str, path: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(
        resolve_url(model_ref, path),
        method="HEAD",
        headers={"User-Agent": "assetpack-template-bonsai-layout-boundary/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size_header = response.headers.get("Content-Length")
            return {
                "path": path,
                "present": True,
                "status_code": int(response.status),
                "size": int(size_header) if size_header and size_header.isdigit() else None,
            }
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"path": path, "present": False, "status_code": 404, "size": None}
        raise RuntimeError(f"HTTP {exc.code} probing {path}: {exc.reason}") from exc


def boundary_files(model_ref: str) -> list[dict[str, Any]]:
    probed = [probe_resolve_path(model_ref, path) for path in BOUNDARY_PROBE_PATHS]
    files = [{"path": item["path"], "size": item.get("size")} for item in probed if item.get("present")]
    if not files:
        raise RuntimeError("no boundary files were found via resolve probing")
    return sorted(files, key=lambda x: x["path"])


def layout_files(model_ref: str, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "tree":
        return hf_files(model_ref), {"probe_mode": "tree"}
    if mode == "boundary_files":
        return boundary_files(model_ref), {"probe_mode": "boundary_files", "probed_paths": BOUNDARY_PROBE_PATHS}
    raise ValueError(f"unknown layout probe mode: {mode}")


def add_stage(result: dict[str, Any], name: str, fn):
    start = time.time()
    try:
        payload = fn()
        result["stages"].append({"name": name, "status": "passed", "seconds": round(time.time() - start, 3), **payload})
        return payload
    except Exception as exc:
        result["stages"].append({
            "name": name,
            "status": "failed",
            "seconds": round(time.time() - start, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "external_rate_limited": is_external_rate_limit(exc),
        })
        raise


def load_component(model_ref: str, component: str) -> dict[str, Any]:
    if component == "vae_decoder":
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_ref, filename="vae/config.json", repo_type="model")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return {
            "component": component,
            "load_kind": "config_only",
            "class_name": config.get("_class_name", "AutoencoderKL"),
            "config_keys": sorted(config.keys()),
        }

    if component == "transformer":
        import torch
        import diffusers
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_ref, filename="transformer/config.json", repo_type="model")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        class_name = config.get("_class_name", "FluxTransformer2DModel")
        cls = getattr(diffusers, class_name)
        model = cls.from_pretrained(
            model_ref,
            subfolder="transformer",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        dtype_counts: dict[str, int] = {}
        param_count = 0
        for param in model.parameters():
            dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + param.numel()
            param_count += param.numel()
        payload = {
            "component": component,
            "load_kind": "weights",
            "class_name": class_name,
            "param_count": int(param_count),
            "param_count_billion": round(param_count / 1_000_000_000, 3),
            "dtype_param_counts": dtype_counts,
            "config_keys": sorted(config.keys()),
            "selected_config": {
                key: config.get(key)
                for key in [
                    "in_channels",
                    "out_channels",
                    "num_layers",
                    "num_single_layers",
                    "attention_head_dim",
                    "num_attention_heads",
                    "joint_attention_dim",
                    "pooled_projection_dim",
                    "guidance_embeds",
                    "axes_dims_rope",
                ]
                if key in config
            },
        }
        del model
        gc.collect()
        return payload

    raise RuntimeError(f"unsupported component: {component}")


def export_vae_decoder(model_ref: str, out_dir: Path, width: int, height: int) -> dict[str, Any]:
    import torch
    from diffusers import AutoencoderKL

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_path = onnx_dir / "bonsai_vae_decoder.onnx"

    vae = AutoencoderKL.from_pretrained(model_ref, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    latent_channels = int(getattr(vae.config, "latent_channels", 4))
    dummy = torch.zeros((1, latent_channels, max(1, height // 8), max(1, width // 8)), dtype=torch.float32)

    class Decoder(torch.nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae_model = vae_model

        def forward(self, latents):
            return self.vae_model.decode(latents, return_dict=False)[0]

    with torch.no_grad():
        torch.onnx.export(
            Decoder(vae),
            (dummy,),
            str(out_path),
            input_names=["latents"],
            output_names=["sample"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "sample": {0: "batch", 2: "height", 3: "width"},
            },
        )

    return {
        "component": "vae_decoder",
        "onnx_path": str(out_path),
        "onnx_size_bytes": out_path.stat().st_size,
        "onnx_size_mb": mb(out_path.stat().st_size),
        "input_shape": list(dummy.shape),
    }


def export_component(model_ref: str, component: str, out_dir: Path, width: int, height: int) -> dict[str, Any]:
    if component == "vae_decoder":
        return export_vae_decoder(model_ref, out_dir, width, height)
    if component == "transformer":
        raise RuntimeError("transformer ONNX export is intentionally gated behind a successful transformer component_load run")
    raise RuntimeError(f"unsupported component: {component}")


def ort_forward(onnx_path: Path, input_shape: list[int]) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.zeros(tuple(input_shape), dtype=np.float32)})
    summary = []
    for i, value in enumerate(outputs):
        array = np.asarray(value)
        summary.append({
            "index": i,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
        })
    return {"provider": "CPUExecutionProvider", "input_name": input_name, "outputs": summary}


def run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cand = cfg["candidate"]
    cpu = cfg.get("cpu_smoke", {})
    export_cfg = cfg.get("export", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = os.getenv("BONSAI_REQUIRED_MILESTONE") or export_cfg.get("required_milestone", "layout_probe")
    download = env_bool("BONSAI_DOWNLOAD_WEIGHTS", bool(export_cfg.get("download_weights", False)))
    component = os.getenv("BONSAI_EXPORT_COMPONENT") or export_cfg.get("component", "vae_decoder")
    allow_external_rate_limit = env_bool("BONSAI_ALLOW_EXTERNAL_RATE_LIMIT", False)
    layout_probe_mode = os.getenv("BONSAI_LAYOUT_PROBE_MODE") or "tree"
    rank(required)

    result = {
        "experiment_id": cfg.get("id"),
        "model_id": cand.get("id"),
        "model_ref": cand.get("model_ref"),
        "backend": cand.get("backend"),
        "status": "failed",
        "ci_conclusion": "failure",
        "required_milestone": required,
        "milestone_reached": None,
        "download_weights": download,
        "export_component": component,
        "layout_probe_mode": layout_probe_mode,
        "claim_promotable_to_manifest": False,
        "stages": [],
    }
    start = time.time()
    try:
        def layout_payload() -> dict[str, Any]:
            files, metadata = layout_files(cand["model_ref"], layout_probe_mode)
            return {"analysis": analyze(files), **metadata}

        add_stage(result, "layout_probe", layout_payload)
        result["milestone_reached"] = "layout_probe"
        if rank(required) <= rank("layout_probe"):
            result["status"] = "passed"
            result["ci_conclusion"] = "success"
            result["claim_promotable_to_manifest"] = True
            return 0

        if not download:
            raise RuntimeError("component load requires download_weights=true")

        add_stage(result, "component_load", lambda: load_component(cand["model_ref"], component))
        result["milestone_reached"] = "component_load"
        if rank(required) <= rank("component_load"):
            result["status"] = "passed"
            result["ci_conclusion"] = "success"
            result["claim_promotable_to_manifest"] = True
            return 0

        payload = add_stage(
            result,
            "component_export",
            lambda: export_component(cand["model_ref"], component, out_dir, int(cpu.get("width", 128)), int(cpu.get("height", 128))),
        )
        result["milestone_reached"] = "component_export"
        if rank(required) <= rank("component_export"):
            result["status"] = "passed"
            result["ci_conclusion"] = "success"
            result["claim_promotable_to_manifest"] = True
            return 0

        add_stage(result, "ort_cpu_forward", lambda: ort_forward(Path(payload["onnx_path"]), payload["input_shape"]))
        result["milestone_reached"] = "ort_cpu_forward"
        if rank(required) <= rank("ort_cpu_forward"):
            result["status"] = "passed"
            result["ci_conclusion"] = "success"
            result["claim_promotable_to_manifest"] = True
            return 0

        raise RuntimeError("full ONNX CPU image generation is not implemented yet")
    except Exception as exc:
        if allow_external_rate_limit and required == "layout_probe" and is_external_rate_limit(exc):
            result["status"] = "external_rate_limited"
            result["ci_conclusion"] = "success_with_external_rate_limit"
            result["claim_promotable_to_manifest"] = False
            result["external_rate_limited"] = True
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)[:2000]
            return 0
        result["status"] = "failed"
        result["ci_conclusion"] = "failure"
        result["claim_promotable_to_manifest"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        return 1
    finally:
        result["seconds"] = round(time.time() - start, 3)
        (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_test() -> int:
    assert rank("layout_probe") == 0
    assert rank("component_load") == 1
    assert rank("ort_cpu_forward") == 3
    assert is_external_rate_limit(RuntimeError("429 Too Many Requests")) is True
    assert is_external_rate_limit(RuntimeError("ordinary failure")) is False
    analysis = analyze([
        {"path": "README.md", "size": 1},
        {"path": "transformer/config.json", "size": 2},
        {"path": "transformer/diffusion_pytorch_model.safetensors", "size": 2048},
        {"path": "vae/config.json", "size": 2},
        {"path": "vae/diffusion_pytorch_model.safetensors", "size": 1024},
        {"path": "tokenizer/tokenizer_config.json", "size": 1},
        {"path": "scheduler/scheduler_config.json", "size": 1},
    ])
    assert analysis["has_model_index_json"] is False
    assert analysis["component_candidates"]["vae"]["present"] is True
    assert analysis["component_candidates"]["transformer"]["present"] is True
    assert analysis["component_candidates"]["tokenizer"]["present"] is True
    assert analysis["component_candidates"]["scheduler"]["present"] is True
    assert analysis["safetensors_count"] == 2
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bonsai-onnx-smoke.yml")
    parser.add_argument("--out-dir", default="reports/bonsai-onnx-smoke")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
