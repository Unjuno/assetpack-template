#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def committed_asset_path(request: dict) -> str:
    recipe_id = str(request.get("recipe_id", "unknown"))
    issue_number = str(request.get("issue_number", "unknown"))
    issue_dir = f"issue-{int(issue_number):06d}" if issue_number.isdigit() else f"issue-{issue_number}"
    return f"assets/generated/{issue_dir}/{recipe_id}"


def committed_asset_url(path: str) -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        return ""
    return f"{server}/{repo}/tree/main/{path}"


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
    asset_path = committed_asset_path(request)
    asset_url = committed_asset_url(asset_path)

    if args.outcome == "success" and passed:
        url_line = f"- GitHub URL: {asset_url}\n" if asset_url else ""
        body = (
            "## Asset image generated and committed\n\n"
            "CI generated an image for this structured request. The prompt/image record is committed to the repository when the commit step succeeds.\n\n"
            f"- recipe: `{recipe_id}`\n"
            f"- model: `{model_id}`\n"
            f"- committed asset: `{asset_path}`\n"
            f"{url_line}\n"
            "The workflow appends the committed image and prompt paths below after staging succeeds.\n"
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
