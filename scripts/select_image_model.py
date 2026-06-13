#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def candidate_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = cfg.get("models", {}).get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("models.candidates must be a list")
    index: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        cid = candidate.get("id")
        if isinstance(cid, str) and cid:
            index[cid] = candidate
    return index


def resolve_image_model(cfg: dict[str, Any], override: str | None = None) -> dict[str, Any]:
    image_generation = cfg.get("models", {}).get("image_generation")
    if not isinstance(image_generation, dict):
        raise ValueError("models.image_generation is required")

    default_model_id = image_generation.get("default_model_id")
    if not isinstance(default_model_id, str) or not default_model_id:
        raise ValueError("models.image_generation.default_model_id is required")

    allowed_model_ids = image_generation.get("allowed_model_ids", [])
    if not isinstance(allowed_model_ids, list) or not all(isinstance(x, str) for x in allowed_model_ids):
        raise ValueError("models.image_generation.allowed_model_ids must be a list of strings")

    runtime_override = image_generation.get("runtime_override", {})
    if not isinstance(runtime_override, dict):
        runtime_override = {}
    env_key = runtime_override.get("environment_variable", "ASSETPACK_IMAGE_MODEL_ID")
    env_value = os.getenv(env_key, "") if isinstance(env_key, str) else ""

    requested_model_id = override or env_value or default_model_id
    if requested_model_id not in allowed_model_ids:
        raise ValueError(
            f"Image model '{requested_model_id}' is not allowed. "
            f"Allowed values: {', '.join(allowed_model_ids)}"
        )

    candidates = candidate_index(cfg)
    candidate = candidates.get(requested_model_id)
    if not candidate:
        raise ValueError(f"Allowed image model '{requested_model_id}' has no matching models.candidates entry")
    if not candidate.get("enabled", False):
        raise ValueError(f"Selected image model '{requested_model_id}' is not enabled")

    return {
        "selected_model_id": requested_model_id,
        "default_model_id": default_model_id,
        "allowed_model_ids": allowed_model_ids,
        "override_environment_variable": env_key,
        "used_override": bool(override or env_value),
        "candidate": candidate,
        "selection_basis": image_generation.get("selection_basis", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve the configured assetpack image-generation model.")
    parser.add_argument("--config", default="assetpack.yml", help="Path to assetpack.yml")
    parser.add_argument("--model-id", default="", help="Optional explicit model ID override")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(Path(args.config))
        resolved = resolve_image_model(cfg, override=args.model_id or None)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(resolved, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
