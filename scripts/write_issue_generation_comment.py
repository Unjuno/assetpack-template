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

    if args.outcome == "success" and passed:
        body = (
            "## Asset image generated\n\n"
            "CI generated an image for this structured request.\n\n"
            f"- recipe: `{request.get('recipe_id', 'unknown')}`\n"
            f"- model: `{request.get('selected_model_id', 'unknown')}`\n"
            "- image: available in the workflow artifact\n\n"
            "Generated images are not committed to Git by default. Review the artifact before publishing.\n"
        )
    else:
        body = (
            "## Asset image generation incomplete\n\n"
            "The request passed validation, but the generation step did not produce a successful image.\n\n"
            f"- recipe: `{request.get('recipe_id', 'unknown')}`\n"
            f"- model: `{request.get('selected_model_id', 'unknown')}`\n"
            f"- workflow outcome: `{args.outcome}`\n\n"
            "Check the workflow artifact for `report.json`.\n"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "generation-comment.md").write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
