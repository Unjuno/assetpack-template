#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EMPTY_INDEX = """# Generated assets

This index is generated from committed asset records under `assets/generated/`.

| Issue | Recipe | Model | Asset | Image | Prompt |
| --- | --- | --- | --- | --- | --- |
"""


def reset_generated_assets(root: Path, apply: bool) -> int:
    root.mkdir(parents=True, exist_ok=True)
    candidates = sorted(root.glob("issue-*"))

    for path in candidates:
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"{'removed' if apply else 'would remove'} {path}")

    index = root / "README.md"
    if apply:
        index.write_text(EMPTY_INDEX, encoding="utf-8")
        print(f"reset {index}")
    else:
        print(f"would reset {index}")

    return len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset template sample generated assets in a derived repository."
    )
    parser.add_argument(
        "--root",
        default="assets/generated",
        help="Generated asset root to reset. Defaults to assets/generated.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the reset. Without this flag the command only prints what it would do.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    count = reset_generated_assets(root, args.yes)
    if not args.yes:
        print("dry run only; re-run with --yes to apply")
    print(f"matched {count} generated issue record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
