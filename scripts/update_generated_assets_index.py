#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    records: list[Path] = []
    for issue_dir in sorted(root.glob("issue-*")):
        if not issue_dir.is_dir():
            continue
        for recipe_dir in sorted(issue_dir.iterdir()):
            if not recipe_dir.is_dir():
                continue
            if (recipe_dir / "image.png").exists() and (recipe_dir / "prompt.txt").exists():
                records.append(recipe_dir)
    return records


def row_for(root: Path, path: Path) -> dict[str, str]:
    metadata = load_json(path / "metadata.json")
    request = load_json(path / "request.json")
    rel = path.relative_to(root).as_posix()
    issue_number = metadata.get("issue_number") or request.get("issue_number") or ""
    recipe_id = metadata.get("recipe_id") or request.get("recipe_id") or path.name
    model_id = metadata.get("selected_model_id") or request.get("selected_model_id") or ""
    return {
        "issue": str(issue_number),
        "recipe": str(recipe_id),
        "model": str(model_id),
        "asset": rel,
        "image": f"{rel}/image.png",
        "prompt": f"{rel}/prompt.txt",
    }


def render(root: Path) -> str:
    rows = [row_for(root, path) for path in record_dirs(root)]
    lines = [
        "# Generated assets",
        "",
        "This index is generated from committed asset records under `assets/generated/`.",
        "",
    ]
    if not rows:
        lines.extend([
            "No generated assets are committed yet.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "| Issue | Recipe | Model | Asset | Image | Prompt |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        issue = row["issue"]
        issue_cell = f"#{issue}" if issue else ""
        lines.append(
            f"| {issue_cell} | `{row['recipe']}` | `{row['model']}` | "
            f"[{row['asset']}]({row['asset']}/) | "
            f"[image]({row['image']}) | [prompt]({row['prompt']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="assets/generated")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(render(root), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
