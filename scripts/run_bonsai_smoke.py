#!/usr/bin/env python3
import json
import time
from pathlib import Path

import yaml


def main() -> int:
    cfg = yaml.safe_load(Path("experiments/bonsai-smoke.yml").read_text(encoding="utf-8"))
    limits = cfg.get("limits", {})
    cand = cfg.get("candidate", {})
    out_dir = Path("reports/bonsai-smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ref = cand.get("model_ref")
    prompt = cfg.get("prompt", {}).get("text", "simple object, no text")
    width = int(limits.get("width", 256))
    height = int(limits.get("height", 256))

    result = {
        "experiment_id": cfg.get("id"),
        "model_id": cand.get("id"),
        "model_ref": model_ref,
        "backend": cand.get("backend"),
        "width": width,
        "height": height,
        "prompt": prompt,
        "status": "failed"
    }

    start = time.time()
    try:
        import torch
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float32)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        image = pipe(prompt=prompt, num_inference_steps=4, width=width, height=height).images[0]
        image.save(out_dir / "bonsai-image-4b.png")
        result["status"] = "passed"
        result["seconds"] = round(time.time() - start, 3)
        result["output"] = str(out_dir / "bonsai-image-4b.png")
    except Exception as exc:
        result["status"] = "failed"
        result["seconds"] = round(time.time() - start, 3)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]

    (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
