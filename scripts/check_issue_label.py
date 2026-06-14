#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="assetpack.yml")
    parser.add_argument("--event-json", required=True)
    parser.add_argument("--github-output", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    required_label = str(cfg.get("issue_generation", {}).get("required_label", "asset-request"))
    event = json.loads(Path(args.event_json).read_text(encoding="utf-8"))
    labels = [str(item.get("name", "")) for item in event.get("issue", {}).get("labels", [])]
    should_run = required_label in labels

    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as f:
            f.write(f"should_run={'true' if should_run else 'false'}\n")
            f.write(f"required_label={required_label}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
