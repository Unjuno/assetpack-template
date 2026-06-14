#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def find_pngs(root: Path) -> list[Path]:
    return sorted(root.glob("images/**/*.png"))


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except Exception:
        return None, None
    try:
        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="assetpack.yml")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--source-out-dir", required=True)
    parser.add_argument("--repo-output-root", default="")
    parser.add_argument("--github-run-id", default="")
    parser.add_argument("--github-sha", default="")
    parser.add_argument("--github-output", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    req = load_json(Path(args.request_json))
    report = load_json(Path(args.report_json))

    if not req.get("valid"):
        raise SystemExit("request is not valid")

    required_terms = [str(x).strip() for x in cfg.get("prompt_policy", {}).get("required_terms", []) if str(x).strip()]
    prompt = str(req.get("prompt", ""))
    missing_terms = [term for term in required_terms if term not in prompt]
    if missing_terms:
        raise SystemExit(f"missing required terms: {missing_terms}")

    pngs = find_pngs(Path(args.source_out_dir))
    if len(pngs) != 1:
        raise SystemExit(f"expected exactly one PNG, got {len(pngs)}")

    source_png = pngs[0]
    image_file_size_bytes = source_png.stat().st_size
    image_width, image_height = image_dimensions(source_png)

    recipe_id = str(req.get("recipe_id", "")).strip()
    issue_number = str(req.get("issue_number", "unknown")).strip() or "unknown"
    output_root = Path(args.repo_output_root or cfg.get("issue_generation", {}).get("committed_output_root", "assets/generated"))
    issue_dir = f"issue-{int(issue_number):06d}" if issue_number.isdigit() else f"issue-{issue_number}"
    dest = output_root / issue_dir / recipe_id

    if dest.exists():
        raise SystemExit(f"duplicate recipe_id already exists: {dest}")

    dest.mkdir(parents=True, exist_ok=False)
    dest_png = dest / "image.png"
    shutil.copy2(source_png, dest_png)
    (dest / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (dest / "negative_prompt.txt").write_text(str(req.get("negative_prompt", "")) + "\n", encoding="utf-8")
    (dest / "request.json").write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata: dict[str, Any] = {
        "issue_number": req.get("issue_number"),
        "recipe_id": recipe_id,
        "selected_model_id": req.get("selected_model_id"),
        "github_run_id": args.github_run_id,
        "github_sha": args.github_sha,
        "source_png": str(source_png),
        "committed_path": str(dest),
        "image_file_size_bytes": image_file_size_bytes,
        "image_width": image_width,
        "image_height": image_height,
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "README.md").write_text(
        "# Generated asset\n\n"
        f"- recipe_id: `{recipe_id}`\n"
        f"- model: `{req.get('selected_model_id', 'unknown')}`\n"
        "- image: `image.png`\n"
        "- prompt: `prompt.txt`\n"
        f"- image_file_size_bytes: `{image_file_size_bytes}`\n"
        f"- image_width: `{image_width}`\n"
        f"- image_height: `{image_height}`\n",
        encoding="utf-8",
    )

    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as f:
            f.write(f"committed_asset_dir={dest}\n")
            f.write(f"committed_image_path={dest / 'image.png'}\n")
            f.write(f"committed_prompt_path={dest / 'prompt.txt'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
