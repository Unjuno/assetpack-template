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
import signal
import time
from pathlib import Path
from typing import Any

import yaml


class CandidateTimeout(TimeoutError):
    pass


def now() -> float:
    return time.time()


def max_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": int(usage.total), "used_bytes": int(usage.used), "free_bytes": int(usage.free)}


def write_report(out: Path, report: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def comma_set(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id", "enabled", "batch", "ci_stage", "method", "model_ref", "base_model_ref", "lora_model_ref",
        "pipeline_class", "scheduler_class", "height", "width", "steps", "guidance_scale", "torch_dtype",
        "use_negative_prompt", "extra_call_kwargs", "license_hint", "expected_role", "notes",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def torch_dtype(candidate: dict[str, Any], cfg: dict[str, Any]):
    import torch
    name = candidate.get("torch_dtype") or cfg.get("runtime", {}).get("torch_dtype", "float32")
    return getattr(torch, name)


def diffusers_class(name: str):
    import diffusers
    return getattr(diffusers, name)


def disable_safety_checker(pipe: Any) -> None:
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass


def call_kwargs(candidate: dict[str, Any], cfg: dict[str, Any], generator: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": cfg["prompt"],
        "num_inference_steps": int(candidate.get("steps", 2)),
        "generator": generator,
    }
    if candidate.get("use_negative_prompt", True) and (candidate.get("negative_prompt") or cfg.get("negative_prompt")):
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


def save_image(candidate: dict[str, Any], cfg: dict[str, Any], out: Path, image: Any) -> dict[str, Any]:
    image_dir = out / "images" / candidate["id"]
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "image.png"
    if cfg.get("runtime", {}).get("save_images", True):
        image.save(image_path)
        return {"image_path": str(image_path), "image_sha256": sha256_file(image_path), "image_size_bytes": image_path.stat().st_size}
    return {"image_path": None, "image_sha256": None, "image_size_bytes": None}


def run_text_to_image(candidate: dict[str, Any], cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    import torch
    import diffusers
    started = now()
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))
    cls = diffusers_class(candidate.get("pipeline_class", "DiffusionPipeline"))
    load_started = now()
    pipe = cls.from_pretrained(candidate["model_ref"], torch_dtype=torch_dtype(candidate, cfg))
    disable_safety_checker(pipe)
    if hasattr(pipe, "to"):
        pipe = pipe.to("cpu")
    load_seconds = round(now() - load_started, 3)
    generate_started = now()
    kwargs = call_kwargs(candidate, cfg, generator)
    result = pipe(**kwargs)
    image = result.images[0]
    image_record = save_image(candidate, cfg, out, image)
    generate_seconds = round(now() - generate_started, 3)
    height = int(candidate.get("height", getattr(image, "height", 0)))
    width = int(candidate.get("width", getattr(image, "width", 0)))
    del pipe, result
    gc.collect()
    return {
        "status": "passed", "method": candidate.get("method"), "pipeline_class": candidate.get("pipeline_class"),
        "model_ref": candidate.get("model_ref"), "load_seconds": load_seconds, "generate_seconds": generate_seconds,
        "total_seconds": round(now() - started, 3), "height": height, "width": width,
        "steps": int(candidate.get("steps", 2)), "guidance_scale": candidate.get("guidance_scale"),
        "call_kwargs": {key: value for key, value in kwargs.items() if key != "generator"},
        "execution_attempted": True, "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None), **image_record,
    }


def run_lora_text_to_image(candidate: dict[str, Any], cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    import torch
    import diffusers
    started = now()
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 1)))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))
    pipe_cls = diffusers_class(candidate.get("pipeline_class", "StableDiffusionPipeline"))
    scheduler_cls = diffusers_class(candidate.get("scheduler_class", "LCMScheduler"))
    load_started = now()
    pipe = pipe_cls.from_pretrained(candidate["base_model_ref"], torch_dtype=torch_dtype(candidate, cfg))
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(candidate["lora_model_ref"])
    disable_safety_checker(pipe)
    pipe = pipe.to("cpu")
    load_seconds = round(now() - load_started, 3)
    generate_started = now()
    kwargs = call_kwargs(candidate, cfg, generator)
    result = pipe(**kwargs)
    image = result.images[0]
    image_record = save_image(candidate, cfg, out, image)
    generate_seconds = round(now() - generate_started, 3)
    del pipe, result
    gc.collect()
    return {
        "status": "passed", "method": candidate.get("method"), "pipeline_class": candidate.get("pipeline_class"),
        "scheduler_class": candidate.get("scheduler_class"), "base_model_ref": candidate.get("base_model_ref"),
        "lora_model_ref": candidate.get("lora_model_ref"), "load_seconds": load_seconds,
        "generate_seconds": generate_seconds, "total_seconds": round(now() - started, 3),
        "height": int(candidate.get("height", 256)), "width": int(candidate.get("width", 256)),
        "steps": int(candidate.get("steps", 4)), "guidance_scale": candidate.get("guidance_scale"),
        "call_kwargs": {key: value for key, value in kwargs.items() if key != "generator"},
        "execution_attempted": True, "max_rss_mb_after_candidate": max_rss_mb(),
        "diffusers_version": getattr(diffusers, "__version__", None), **image_record,
    }


def timeout_handler(signum: int, frame: Any) -> None:
    raise CandidateTimeout("candidate exceeded configured timeout seconds")


def run_candidate(candidate: dict[str, Any], cfg: dict[str, Any], out: Path, timeout_seconds: int) -> dict[str, Any]:
    started = now()
    record: dict[str, Any] = {
        "candidate": public_candidate(candidate), "status": "started",
        "started_at_monotonic_seconds": round(started, 3), "disk_before": disk_snapshot(Path.cwd()),
        "max_rss_mb_before": max_rss_mb(),
    }
    old_handler = None
    if timeout_seconds > 0:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        record["candidate_timeout_seconds"] = timeout_seconds
    try:
        method = candidate.get("method")
        if method == "diffusers_text_to_image":
            record.update(run_text_to_image(candidate, cfg, out))
        elif method == "diffusers_lora_text_to_image":
            record.update(run_lora_text_to_image(candidate, cfg, out))
        else:
            record.update({"status": "skipped", "reason": f"unsupported method: {method}", "execution_attempted": False})
    except BaseException as exc:
        record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:4000], "total_seconds": round(now() - started, 3), "max_rss_mb_after_candidate": max_rss_mb()})
    finally:
        if timeout_seconds > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    record["disk_after"] = disk_snapshot(Path.cwd())
    return record


def select_candidates(cfg: dict[str, Any], ids: set[str], batches: set[str], include_disabled: bool) -> list[dict[str, Any]]:
    selected = []
    for candidate in cfg.get("candidates", []):
        if ids and candidate.get("id") not in ids:
            continue
        if batches and candidate.get("batch") not in batches:
            continue
        if not include_disabled and not candidate.get("enabled", False):
            continue
        selected.append(candidate)
    return selected


def run(config_path: str, out_dir: str, candidate_ids: str = "", batches: str = "", include_disabled: bool = False, candidate_timeout_seconds: int = 0) -> int:
    started = now()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(out_dir or cfg.get("output_root", "reports/issue-generation"))
    out.mkdir(parents=True, exist_ok=True)
    ids = comma_set(candidate_ids or os.getenv("ASSETPACK_IMAGE_CANDIDATE_IDS", ""))
    batch_set = comma_set(batches or os.getenv("ASSETPACK_IMAGE_CANDIDATE_BATCHES", ""))
    include_disabled = include_disabled or os.getenv("ASSETPACK_IMAGE_INCLUDE_DISABLED", "").lower() in {"1", "true", "yes", "on"}
    timeout_from_env = int(os.getenv("ASSETPACK_IMAGE_CANDIDATE_TIMEOUT_SECONDS", "0") or "0")
    timeout_seconds = int(candidate_timeout_seconds or timeout_from_env or 0)
    candidates = select_candidates(cfg, ids, batch_set, include_disabled)
    report: dict[str, Any] = {
        "experiment_id": cfg.get("experiment_id", "assetpack-issue-generation"),
        "prompt": cfg.get("prompt"), "negative_prompt": cfg.get("negative_prompt"), "seed": cfg.get("seed"),
        "runtime": cfg.get("runtime", {}), "selection": {"candidate_ids": sorted(ids), "batches": sorted(batch_set), "include_disabled": include_disabled, "selected_count": len(candidates)},
        "candidate_timeout_seconds": timeout_seconds, "system": {"platform": platform.platform(), "python": platform.python_version(), "cwd": str(Path.cwd()), "disk_start": disk_snapshot(Path.cwd())},
        "claim_promotable_to_manifest": False, "allowed_claim": "assetpack_issue_generation_record_not_model_quality_claim", "status": "started", "candidates": [],
    }
    write_report(out, report)
    for candidate in candidates:
        report["candidates"].append(run_candidate(candidate, cfg, out, timeout_seconds))
        write_report(out, report)
    passed = sum(1 for item in report["candidates"] if item.get("status") == "passed")
    failed = sum(1 for item in report["candidates"] if item.get("status") == "failed")
    skipped = sum(1 for item in report["candidates"] if item.get("status") == "skipped")
    report["summary"] = {"passed": passed, "failed": failed, "skipped": skipped, "total": len(report["candidates"])}
    report["status"] = "passed" if passed > 0 else "failed"
    report["seconds"] = round(now() - started, 3)
    report["system"]["disk_end"] = disk_snapshot(Path.cwd())
    report["system"]["max_rss_mb_end"] = max_rss_mb()
    write_report(out, report)
    return 0 if passed > 0 and failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reports/issue-generation/issue-generation-config.yml")
    parser.add_argument("--out-dir", default="reports/issue-generation")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--batches", default="")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=0)
    args = parser.parse_args()
    return run(args.config, args.out_dir, args.candidate_ids, args.batches, args.include_disabled, args.candidate_timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
