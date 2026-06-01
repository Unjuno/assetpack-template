#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import yaml


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def pick_models(cfg, requested):
    items = cfg.get("candidates", [])
    if requested != "all":
        items = [x for x in items if x.get("id") == requested]
    return [x for x in items if x.get("enabled") and x.get("backend") == "diffusers"]


def run_one(model, prompt, width, height, steps, out_dir):
    import torch
    from diffusers import DiffusionPipeline

    model_id = model["id"]
    model_ref = model["model_ref"]
    start = time.time()
    record = {
        "model_id": model_id,
        "model_ref": model_ref,
        "backend": "diffusers",
        "width": width,
        "height": height,
        "steps": steps,
        "status": "running",
    }

    try:
        pipe = DiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float32)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        image = pipe(prompt=prompt, num_inference_steps=steps, width=width, height=height).images[0]
        path = out_dir / f"{model_id}.png"
        image.save(path)
        record.update({
            "status": "passed",
            "seconds": round(time.time() - start, 3),
            "output": str(path),
        })
    except Exception as exc:
        record.update({
            "status": "failed",
            "seconds": round(time.time() - start, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/model-candidates.yml")
    parser.add_argument("--model", default="tiny-sd")
    parser.add_argument("--out", default="reports/diffusers-smoke")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    limits = cfg.get("limits", {})
    width = int(limits.get("width", 256))
    height = int(limits.get("height", 256))
    prompt = cfg.get("prompts", [{}])[0].get("prompt", "simple object, no text")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = pick_models(cfg, args.model)
    if not selected:
        report = {"status": "failed", "reason": "no enabled diffusers model selected", "requested": args.model}
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    rows = []
    for model in selected:
        steps = 1 if model.get("id") == "sd-turbo" else 4
        rows.append(run_one(model, prompt, width, height, steps, out_dir))

    report = {
        "experiment_id": cfg.get("id"),
        "runner_hint": cfg.get("runner"),
        "prompt_id": cfg.get("prompts", [{}])[0].get("id"),
        "results": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if any(x.get("status") == "passed" for x in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
