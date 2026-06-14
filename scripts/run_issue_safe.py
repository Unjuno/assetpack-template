#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="assetpack.yml")
    p.add_argument("--request-json", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    req_path = Path(a.request_json)
    req = json.loads(req_path.read_text(encoding="utf-8"))
    need = [str(x).strip() for x in cfg.get("prompt_policy", {}).get("required_terms", []) if str(x).strip()]
    miss = [x for x in need if x not in str(req.get("prompt", ""))]
    req["required_terms"] = need
    req["missing_terms"] = miss
    req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if miss:
        report = {"status": "not_generated", "summary": {"passed": 0, "failed": 1}, "reason": "missing required terms"}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1

    result = subprocess.run([
        sys.executable,
        "scripts/run_issue_asset_generation.py",
        "--config", a.config,
        "--request-json", a.request_json,
        "--out-dir", a.out_dir,
    ])
    if result.returncode != 0 and not (out / "report.json").exists():
        report = {"status": "failed", "summary": {"passed": 0, "failed": 1}, "reason": "generation subprocess failed"}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
