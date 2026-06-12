#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


def now_seconds() -> float:
    return time.time()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": int(usage.total), "used_bytes": int(usage.used), "free_bytes": int(usage.free)}


def max_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def comma_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def candidate_public_record(candidate: dict) -> dict:
    keys = [
        "id", "enabled", "batch", "ci_stage", "method", "model_ref", "base_model_ref", "lora_model_ref",
        "pipeline_class", "scheduler_class", "height", "width", "steps", "guidance_scale", "torch_dtype",
        "use_negative_prompt", "extra_call_kwargs", "license_hint", "expected_role", "notes",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def load_diffusers_class(name: str):
    import diffusers
    return getattr(diffusers, name)


def disable_safety_checker_when_supported(pipe: Any) -> None:
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except BaseException:
            pass


def torch_dtype_from_cfg(candidate: dict, cfg: dict):
    import torch
    dtype_name = candidate.get("torch_dtype") or cfg.get("runtime", {}).get("torch_dtype", "float32")
    return getattr(torch, dtype_name)


def load_pipe(candidate: dict, cfg: dict):
    cls = load_diffusers_class(candidate.get("pipeline_class", "DiffusionPipeline"))
    return cls.from_pretrained(candidate["model_ref"], torch_dtype=torch_dtype_from_cfg(candidate, cfg))


def save_image(candidate: dict, cfg: dict, out_dir: Path, image) -> dict:
    candidate_dir = out_dir / "images" / candidate["id"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    image_path = candidate_dir / "cat.png"
    if cfg.get("runtime", {}).get("save_images", True):
        image.save(image_path)
        return {
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_size_bytes": image_path.stat().st_size,
        }
    return {"image_path": None, "image_sha256": None, "image_size_bytes": None}


def generation_kwargs(candidate: dict, cfg: dict, generator) -> dict:
    kwargs = {
        "prompt": cfg["prompt"],
        "num_inference_steps": int(candidate.get("steps", 2)),
        "generator": generator,
    }
    use_negative = candidate.get("use_negative_prompt", True)
    if use_negative and (candidate.get("negative_prompt") or cfg.get("negative_prompt")):
        kwargs["negative_prompt"] = candidate.get("negative_prompt") or cfg.get("negative_prompt")
    if candidate.get("height") is not None:
        kwargs["height"] = int(candidate["height"])
    if candidate.get("width") is not None:
        kwargs["width"] = int(candidate["width"])
    if candidate.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = float(candidate["guidance_scale"])
    for key, value in (candidate.get("extra_call_kwargs") or {}).items():
        kwargs[key] = value
    return kwargs


def run_diffusers_text_to_image(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import diffusers

    started = now_seconds()
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))

    load_started = now_seconds()
    pipe = load_pipe(candidate, cfg)
    disable_safety_checker_when_supported(pipe)
    if hasattr(pipe, "to"):
        pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)

    generate_started = now_seconds()
    result = pipe(**generation_kwargs(candidate, cfg, generator))
    image = result.images[0]
    generate_seconds = round(now_seconds() - generate_started, 3)
    image_record = save_image(candidate, cfg, out_dir, image)

    del pipe, result
    gc.collect()
    return {
        "status": "passed",
        "method": candidate.get("method"),
        "pipeline_class": candidate.get("pipeline_class", "DiffusionPipeline"),
        "model_ref": candidate.get("model_ref"),
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": round(now_seconds() - started, 3),
        "height": int(candidate.get("height", image.height)),
        "width": int(candidate.get("width", image.width)),
        "steps": int(candidate.get("steps", 2)),
        "guidance_scale": candidate.get("guidance_scale"),
        "call_kwargs": {key: value for key, value in generation_kwargs(candidate, cfg, generator).items() if key != "generator"},
        **image_record,
        "execution_attempted": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None),
    }


def run_diffusers_load_only(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import diffusers

    started = now_seconds()
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    load_started = now_seconds()
    pipe = load_pipe(candidate, cfg)
    disable_safety_checker_when_supported(pipe)
    if hasattr(pipe, "to"):
        pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)
    components = sorted(getattr(pipe, "components", {}).keys()) if hasattr(pipe, "components") else []
    del pipe
    gc.collect()
    return {
        "status": "passed",
        "method": candidate.get("method"),
        "pipeline_class": candidate.get("pipeline_class", "DiffusionPipeline"),
        "model_ref": candidate.get("model_ref"),
        "load_seconds": load_seconds,
        "generate_seconds": None,
        "total_seconds": round(now_seconds() - started, 3),
        "components": components,
        "execution_attempted": False,
        "load_only": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None),
    }


def run_lora_text_to_image(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import diffusers

    started = now_seconds()
    pipeline_class = candidate.get("pipeline_class", "StableDiffusionPipeline")
    scheduler_class = candidate.get("scheduler_class", "LCMScheduler")
    cls = load_diffusers_class(pipeline_class)
    scheduler_cls = load_diffusers_class(scheduler_class)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))

    load_started = now_seconds()
    pipe = cls.from_pretrained(candidate["base_model_ref"], torch_dtype=torch_dtype_from_cfg(candidate, cfg))
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(candidate["lora_model_ref"])
    disable_safety_checker_when_supported(pipe)
    pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)

    generate_started = now_seconds()
    result = pipe(**generation_kwargs(candidate, cfg, generator))
    image = result.images[0]
    generate_seconds = round(now_seconds() - generate_started, 3)
    image_record = save_image(candidate, cfg, out_dir, image)

    del pipe, result
    gc.collect()
    return {
        "status": "passed",
        "method": candidate.get("method"),
        "pipeline_class": pipeline_class,
        "scheduler_class": scheduler_class,
        "base_model_ref": candidate.get("base_model_ref"),
        "lora_model_ref": candidate.get("lora_model_ref"),
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": round(now_seconds() - started, 3),
        "height": int(candidate.get("height", 256)),
        "width": int(candidate.get("width", 256)),
        "steps": int(candidate.get("steps", 4)),
        "guidance_scale": candidate.get("guidance_scale"),
        "call_kwargs": {key: value for key, value in generation_kwargs(candidate, cfg, generator).items() if key != "generator"},
        **image_record,
        "execution_attempted": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None),
    }


def run_lora_load_only(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import diffusers

    started = now_seconds()
    pipeline_class = candidate.get("pipeline_class", "StableDiffusionPipeline")
    scheduler_class = candidate.get("scheduler_class", "LCMScheduler")
    cls = load_diffusers_class(pipeline_class)
    scheduler_cls = load_diffusers_class(scheduler_class)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    load_started = now_seconds()
    pipe = cls.from_pretrained(candidate["base_model_ref"], torch_dtype=torch_dtype_from_cfg(candidate, cfg))
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(candidate["lora_model_ref"])
    disable_safety_checker_when_supported(pipe)
    pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)
    components = sorted(getattr(pipe, "components", {}).keys()) if hasattr(pipe, "components") else []
    del pipe
    gc.collect()
    return {
        "status": "passed",
        "method": candidate.get("method"),
        "pipeline_class": pipeline_class,
        "scheduler_class": scheduler_class,
        "base_model_ref": candidate.get("base_model_ref"),
        "lora_model_ref": candidate.get("lora_model_ref"),
        "load_seconds": load_seconds,
        "generate_seconds": None,
        "total_seconds": round(now_seconds() - started, 3),
        "components": components,
        "execution_attempted": False,
        "load_only": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None),
    }


def run_placeholder(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    return {
        "status": "skipped",
        "reason": "Placeholder or unsupported runtime; add an implementation before enabling execution.",
        "execution_attempted": False,
        "load_only": candidate.get("ci_stage") == "load_only",
    }


def run_candidate(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    started = now_seconds()
    record = {
        "candidate": candidate_public_record(candidate),
        "status": "started",
        "started_at_monotonic_seconds": round(started, 3),
        "disk_before": disk_snapshot(Path.cwd()),
        "max_rss_mb_before": max_rss_mb(),
    }
    try:
        method = candidate.get("method")
        if method == "diffusers_text_to_image":
            record.update(run_diffusers_text_to_image(candidate, cfg, out_dir))
        elif method == "diffusers_load_only":
            record.update(run_diffusers_load_only(candidate, cfg, out_dir))
        elif method == "diffusers_lora_text_to_image":
            record.update(run_lora_text_to_image(candidate, cfg, out_dir))
        elif method == "diffusers_lora_load_only":
            record.update(run_lora_load_only(candidate, cfg, out_dir))
        else:
            record.update(run_placeholder(candidate, cfg, out_dir))
    except BaseException as exc:
        record.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "execution_attempted": False,
            "total_seconds": round(now_seconds() - started, 3),
            "max_rss_mb_after_candidate": max_rss_mb(),
        })
    record["disk_after"] = disk_snapshot(Path.cwd())
    return record


def select_candidates(cfg: dict, candidate_ids: set[str], batches: set[str], include_disabled: bool) -> list[dict]:
    selected = []
    for candidate in cfg.get("candidates", []):
        if candidate_ids and candidate.get("id") not in candidate_ids:
            continue
        if batches and candidate.get("batch") not in batches:
            continue
        if not include_disabled and not candidate.get("enabled", False):
            continue
        selected.append(candidate)
    return selected


def run(config_path: str, out_dir: str, candidate_ids: str = "", batches: str = "", include_disabled: bool = False) -> int:
    started = now_seconds()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(out_dir or cfg.get("output_root", "reports/image-model-ci-benchmark"))
    out.mkdir(parents=True, exist_ok=True)
    requested_ids = comma_set(candidate_ids or os.getenv("IMAGE_MODEL_CANDIDATE_IDS", ""))
    requested_batches = comma_set(batches or os.getenv("IMAGE_MODEL_CANDIDATE_BATCHES", ""))
    include_disabled = include_disabled or os.getenv("IMAGE_MODEL_INCLUDE_DISABLED", "").lower() in {"1", "true", "yes", "on"}
    enabled_candidates = select_candidates(cfg, requested_ids, requested_batches, include_disabled)
    report = {
        "experiment_id": cfg.get("experiment_id", "image-model-ci-benchmark"),
        "prompt": cfg.get("prompt"),
        "negative_prompt": cfg.get("negative_prompt"),
        "seed": cfg.get("seed"),
        "runtime": cfg.get("runtime", {}),
        "selection": {
            "candidate_ids": sorted(requested_ids),
            "batches": sorted(requested_batches),
            "include_disabled": include_disabled,
            "selected_count": len(enabled_candidates),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(Path.cwd()),
            "disk_start": disk_snapshot(Path.cwd()),
        },
        "claim_promotable_to_manifest": False,
        "allowed_claim": "ci_image_model_benchmark_measurement_not_model_quality_claim",
        "status": "started",
        "candidates": [],
    }
    write_report(out, report)

    for candidate in enabled_candidates:
        result = run_candidate(candidate, cfg, out)
        report["candidates"].append(result)
        passed = sum(1 for item in report["candidates"] if item.get("status") == "passed")
        failed = sum(1 for item in report["candidates"] if item.get("status") == "failed")
        skipped = sum(1 for item in report["candidates"] if item.get("status") == "skipped")
        report["summary_so_far"] = {"passed": passed, "failed": failed, "skipped": skipped, "total": len(report["candidates"])}
        report["seconds_so_far"] = round(now_seconds() - started, 3)
        write_report(out, report)

    passed = sum(1 for item in report["candidates"] if item.get("status") == "passed")
    failed = sum(1 for item in report["candidates"] if item.get("status") == "failed")
    skipped = sum(1 for item in report["candidates"] if item.get("status") == "skipped")
    report["summary"] = {"passed": passed, "failed": failed, "skipped": skipped, "total": len(report["candidates"])}
    report["status"] = "passed" if passed > 0 else "failed"
    report["ci_conclusion"] = "success" if failed == 0 and passed > 0 else "success_with_candidate_failures"
    report["seconds"] = round(now_seconds() - started, 3)
    report["system"]["disk_end"] = disk_snapshot(Path.cwd())
    report["system"]["max_rss_mb_end"] = max_rss_mb()
    write_report(out, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/image-model-ci-benchmark.yml")
    parser.add_argument("--out-dir", default="reports/image-model-ci-benchmark")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--batches", default="")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()
    return run(args.config, args.out_dir, args.candidate_ids, args.batches, args.include_disabled)


if __name__ == "__main__":
    raise SystemExit(main())
