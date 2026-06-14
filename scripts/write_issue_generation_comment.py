#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--outcome", default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    request_path = out / "request.json"
    report_path = out / "report.json"
    request = json.loads(request_path.read_text(encoding="utf-8")) if request_path.exists() else {}
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    passed = report.get("summary", {}).get("passed", 0) > 0

    recipe_id = request.get("recipe_id", "unknown")
    model_id = request.get("selected_model_id", "unknown")

    if args.outcome == "success" and passed:
        body = (
            "## Asset image generated and committed\n\n"
            "CI generated an image for this structured request. The prompt/image record is committed to the repository when the commit step succeeds.\n\n"
            f"- recipe: `{recipe_id}`\n"
            f"- model: `{model_id}`\n\n"
            "The workflow appends the committed asset, image, and prompt paths below after staging succeeds.\n"
        )
    else:
        reason = report.get("reason", "generation did not complete")
        body = (
            "## Asset image generation incomplete\n\n"
            "The request passed validation, but the generation step did not produce a committed image record.\n\n"
            f"- recipe: `{recipe_id}`\n"
            f"- model: `{model_id}`\n"
            f"- workflow outcome: `{args.outcome}`\n"
            f"- reason: `{reason}`\n\n"
            "Check the workflow artifact for `request.json` and `report.json`.\n"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "generation-comment.md").write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
