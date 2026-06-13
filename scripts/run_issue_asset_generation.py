#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_image_model import load_config, resolve_image_model  # noqa: E402
from run_image_model_ci_benchmark import run as run_benchmark  # noqa: E402


def candidate_for_generation(candidate: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(candidate)
    generation = cfg.get("generation", {})
    item["width"] = int(generation.get("width", item.get("width", 256)))
    item["height"] = int(generation.get("height", item.get("height", 256)))
    item["batch"] = "issue_generation"
    item["ci_stage"] = "generation"
    item["expected_role"] = "assetpack_issue_generation_selected_model"
    return item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="assetpack.yml")
    ap.add_argument("--request-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    if not request.get("valid"):
        raise SystemExit("request validation was not successful")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = request["selected_model_id"]
    resolved = resolve_image_model(cfg, override=model_id)
    candidate = candidate_for_generation(resolved["candidate"], cfg)
    timeout_seconds = int(cfg.get("generation", {}).get("timeout_minutes_per_model", 20)) * 60

    experiment = {
        "experiment_id": f"assetpack-issue-generation-{request.get('recipe_id')}",
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt"),
        "seed": 12345,
        "output_root": str(out_dir),
        "runtime": {
            "device": "cpu",
            "torch_dtype": "float32",
            "num_threads": 1,
            "save_images": True,
        },
        "issue_request": {
            "issue_number": request.get("issue_number"),
            "recipe_id": request.get("recipe_id"),
            "fields": request.get("fields", {}),
            "selected_model_id": model_id,
        },
        "candidates": [candidate],
    }

    config_path = out_dir / "issue-generation-config.yml"
    config_path.write_text(yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code = run_benchmark(str(config_path), str(out_dir), candidate_ids=model_id, candidate_timeout_seconds=timeout_seconds)
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    passed = report.get("summary", {}).get("passed", 0) > 0
    return 0 if code == 0 and passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
