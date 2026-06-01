#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

import yaml


def run(cmd, timeout=None):
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "output": proc.stdout[-4000:],
    }


def main() -> int:
    cfg = yaml.safe_load(Path("experiments/bonsai-onnx-smoke.yml").read_text(encoding="utf-8"))
    cand = cfg.get("candidate", {})
    limits = cfg.get("limits", {})
    out_dir = Path("reports/bonsai-onnx-smoke")
    onnx_dir = out_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(limits.get("timeout_minutes", 90)) * 60
    result = {
        "experiment_id": cfg.get("id"),
        "model_id": cand.get("id"),
        "model_ref": cand.get("model_ref"),
        "backend": cand.get("backend"),
        "library_name": cand.get("library_name", "diffusers"),
        "status": "failed",
        "stages": [],
    }

    started = time.time()
    try:
        export = run([
            "optimum-cli", "export", "onnx",
            "--library", cand.get("library_name", "diffusers"),
            "--model", cand.get("model_ref"),
            "--task", cand.get("task", "text-to-image"),
            str(onnx_dir),
        ], timeout=timeout)
        result["stages"].append({"name": "onnx_export", **export})
        if export["returncode"] != 0:
            raise RuntimeError("onnx export failed")

        files = [str(p.relative_to(out_dir)) for p in onnx_dir.rglob("*") if p.is_file()]
        result["status"] = "passed"
        result["exported_files"] = files[:200]
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    finally:
        result["seconds"] = round(time.time() - started, 3)
        (out_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
