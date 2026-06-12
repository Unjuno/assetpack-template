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
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def max_rss_mb() -> float:
    # Linux returns ru_maxrss in KiB. This workflow runs on ubuntu-latest.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_public_record(candidate: dict) -> dict:
    keys = [
        "id",
        "enabled",
        "method",
        "model_ref",
        "base_model_ref",
        "lora_model_ref",
        "pipeline_class",
        "scheduler_class",
        "height",
        "width",
        "steps",
        "guidance_scale",
        "expected_role",
        "notes",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def load_diffusers_class(name: str):
    import diffusers

    return getattr(diffusers, name)


def disable_safety_checker_when_supported(pipe: Any) -> None:
    # Several pipelines expose safety_checker as a mutable attribute. Some do not.
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except BaseException:
            pass


def run_diffusers_text_to_image(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch

    started = now_seconds()
    pipeline_class = candidate.get("pipeline_class", "DiffusionPipeline")
    cls = load_diffusers_class(pipeline_class)
    dtype_name = cfg.get("runtime", {}).get("torch_dtype", "float32")
    torch_dtype = getattr(torch, dtype_name)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))

    load_started = now_seconds()
    pipe = cls.from_pretrained(candidate["model_ref"], torch_dtype=torch_dtype)
    disable_safety_checker_when_supported(pipe)
    if hasattr(pipe, "to"):
        pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)

    generate_started = now_seconds()
    kwargs = {
        "prompt": cfg["prompt"],
        "num_inference_steps": int(candidate.get("steps", 2)),
        "generator": generator,
    }
    if candidate.get("negative_prompt") or cfg.get("negative_prompt"):
        kwargs["negative_prompt"] = candidate.get("negative_prompt") or cfg.get("negative_prompt")
    if candidate.get("height") is not None:
        kwargs["height"] = int(candidate["height"])
    if candidate.get("width") is not None:
        kwargs["width"] = int(candidate["width"])
    if candidate.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = float(candidate["guidance_scale"])

    result = pipe(**kwargs)
    image = result.images[0]
    generate_seconds = round(now_seconds() - generate_started, 3)

    candidate_dir = out_dir / "images" / candidate["id"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    image_path = candidate_dir / "cat.png"
    if cfg.get("runtime", {}).get("save_images", True):
        image.save(image_path)
        image_sha256 = sha256_file(image_path)
        image_size_bytes = image_path.stat().st_size
    else:
        image_sha256 = None
        image_size_bytes = None

    # Release memory before next candidate.
    del pipe
    del result
    gc.collect()

    return {
        "status": "passed",
        "method": candidate.get("method"),
        "pipeline_class": pipeline_class,
        "model_ref": candidate.get("model_ref"),
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": round(now_seconds() - started, 3),
        "height": int(candidate.get("height", image.height)),
        "width": int(candidate.get("width", image.width)),
        "steps": int(candidate.get("steps", 2)),
        "guidance_scale": candidate.get("guidance_scale"),
        "image_path": str(image_path) if cfg.get("runtime", {}).get("save_images", True) else None,
        "image_sha256": image_sha256,
        "image_size_bytes": image_size_bytes,
        "execution_attempted": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
    }


def run_lora_text_to_image(candidate: dict, cfg: dict, out_dir: Path) -> dict:
    import torch
    import diffusers

    started = now_seconds()
    pipeline_class = candidate.get("pipeline_class", "StableDiffusionPipeline")
    scheduler_class = candidate.get("scheduler_class", "LCMScheduler")
    cls = load_diffusers_class(pipeline_class)
    scheduler_cls = load_diffusers_class(scheduler_class)
    dtype_name = cfg.get("runtime", {}).get("torch_dtype", "float32")
    torch_dtype = getattr(torch, dtype_name)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))

    load_started = now_seconds()
    pipe = cls.from_pretrained(candidate["base_model_ref"], torch_dtype=torch_dtype)
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(candidate["lora_model_ref"])
    disable_safety_checker_when_supported(pipe)
    pipe = pipe.to("cpu")
    load_seconds = round(now_seconds() - load_started, 3)

    generate_started = now_seconds()
    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg.get("negative_prompt"),
        height=int(candidate.get("height", 256)),
        width=int(candidate.get("width", 256)),
        num_inference_steps=int(candidate.get("steps", 4)),
        guidance_scale=float(candidate.get("guidance_scale", 1.0)),
        generator=generator,
    )
    image = result.images[0]
    generate_seconds = round(now_seconds() - generate_started, 3)

    candidate_dir = out_dir / "images" / candidate["id"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    image_path = candidate_dir / "cat.png"
    image.save(image_path)
    image_sha256 = sha256_file(image_path)

    del pipe
    del result
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
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "image_size_bytes": image_path.stat().st_size,
        "execution_attempted": True,
        "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None),
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
        elif method == "diffusers_lora_text_to_image":
            record.update(run_lora_text_to_image(candidate, cfg, out_dir))
        else:
            record.update({
                "status": "skipped",
                "reason": f"Unsupported or placeholder method: {method}",
                "execution_attempted": False,
            })
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


def run(config_path: str, out_dir: str) -> int:
    started = now_seconds()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(out_dir or cfg.get("output_root", "reports/image-model-ci-benchmark"))
    out.mkdir(parents=True, exist_ok=True)
    enabled_candidates = [candidate for candidate in cfg.get("candidates", []) if candidate.get("enabled", False)]
    report = {
        "experiment_id": cfg.get("experiment_id", "image-model-ci-benchmark"),
        "prompt": cfg.get("prompt"),
        "negative_prompt": cfg.get("negative_prompt"),
        "seed": cfg.get("seed"),
        "runtime": cfg.get("runtime", {}),
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
    args = parser.parse_args()
    return run(args.config, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
