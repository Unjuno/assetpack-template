#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def run(cmd, cwd=None, timeout=None):
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "output": proc.stdout[-4000:],
    }


def main() -> int:
    cfg = yaml.safe_load(Path("experiments/bonsai-smoke.yml").read_text(encoding="utf-8"))
    limits = cfg.get("limits", {})
    cand = cfg.get("candidate", {})
    out_dir = Path("reports/bonsai-smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = cfg.get("prompt", {}).get("text", "simple object, no text")
    width = int(limits.get("width", 256))
    height = int(limits.get("height", 256))
    timeout_minutes = int(limits.get("timeout_minutes", 90))
    demo_dir = Path(".cache/bonsai-image-demo")
    output_png = out_dir / "bonsai-image-4b.png"

    result = {
        "experiment_id": cfg.get("id"),
        "model_id": cand.get("id"),
        "backend": cand.get("backend"),
        "demo_repo": cand.get("demo_repo"),
        "model": cand.get("model"),
        "runner": cfg.get("runner"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "width": width,
        "height": height,
        "prompt": prompt,
        "status": "failed",
        "stages": [],
    }

    total_start = time.time()
    try:
        if demo_dir.exists():
            shutil.rmtree(demo_dir)

        clone = run(["git", "clone", "--depth", "1", cand["demo_repo"], str(demo_dir)], timeout=600)
        result["stages"].append({"name": "clone_demo", **clone})
        if clone["returncode"] != 0:
            raise RuntimeError("demo clone failed")

        setup = run(["bash", "setup.sh"], cwd=demo_dir, timeout=timeout_minutes * 60)
        result["stages"].append({"name": "setup", **setup})
        if setup["returncode"] != 0:
            raise RuntimeError("demo setup failed")

        download = run(["bash", "scripts/download_model.sh", "--model", cand.get("model", "binary-gemlite")], cwd=demo_dir, timeout=timeout_minutes * 60)
        result["stages"].append({"name": "download_model", **download})
        if download["returncode"] != 0:
            raise RuntimeError("model download failed")

        gen = run([
            "bash", "scripts/generate.sh",
            "--model", cand.get("model", "binary-gemlite"),
            "--prompt", prompt,
            "--size", f"{width}x{height}",
            "--steps", "4",
            "--output", str(Path.cwd() / output_png),
            "--force-gpu-run",
        ], cwd=demo_dir, timeout=timeout_minutes * 60)
        result["stages"].append({"name": "generate", **gen})
        if gen["returncode"] != 0:
            raise RuntimeError("generation failed")

        result["status"] = "passed"
        result["output"] = str(output_png)
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
    finally:
        result["seconds"] = round(time.time() - total_start, 3)
        (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
