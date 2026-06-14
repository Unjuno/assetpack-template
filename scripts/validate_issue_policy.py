#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ASCII_RE = re.compile(r"^[\x09\x0A\x0D\x20-\x7E]*$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_outputs(path: str, req: dict[str, Any]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(f"valid={'true' if req.get('valid') else 'false'}\n")
        f.write(f"selected_model_id={req.get('selected_model_id', '')}\n")
        f.write(f"recipe_id={req.get('recipe_id', '')}\n")
        f.write(f"policy_status={req.get('policy_status', '')}\n")


def duplicate_recipe_path(root: Path, recipe_id: str) -> str | None:
    if not root.exists() or not recipe_id:
        return None
    for path in root.glob(f"**/{recipe_id}"):
        if path.is_dir():
            return str(path)
    return None


def write_rejection_comment(path: Path, req: dict[str, Any]) -> None:
    if req.get("policy_status") == "duplicate":
        body = (
            "## Asset request already exists\n\n"
            "This request resolves to a recipe that is already present in the repository.\n\n"
            f"- recipe: `{req.get('recipe_id', '')}`\n"
            f"- existing asset: `{req.get('duplicate_path', '')}`\n"
        )
        path.write_text(body, encoding="utf-8")
        return

    errors = req.get("errors", [])
    lines = "\n".join(f"- {error}" for error in errors)
    body = (
        "## Asset request not generated\n\n"
        "The structured request did not pass repository policy.\n\n"
        f"{lines}\n\n"
        "No image was generated and no files were committed.\n"
    )
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="assetpack.yml")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--comment-file", required=True)
    parser.add_argument("--github-output", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    req_path = Path(args.request_json)
    req = load_json(req_path)
    errors = list(req.get("errors", []))

    input_policy = cfg.get("input_policy", {})
    ascii_only = input_policy.get("ascii_only", True)
    ascii_fields = input_policy.get("ascii_fields", ["subject", "scene", "audience", "constraints"])
    fields = req.get("fields", {})

    if ascii_only:
        for name in ascii_fields:
            value = str(fields.get(name, ""))
            if value and not ASCII_RE.fullmatch(value):
                errors.append(f"non-ASCII characters found in field: {name}")

    required_terms = [str(x).strip() for x in cfg.get("prompt_policy", {}).get("required_terms", []) if str(x).strip()]
    prompt = str(req.get("prompt", ""))
    missing_terms = [term for term in required_terms if term not in prompt]
    for term in missing_terms:
        errors.append(f"missing required term: {term}")

    recipe_id = str(req.get("recipe_id", ""))
    output_root = Path(cfg.get("issue_generation", {}).get("committed_output_root", "assets/generated"))
    duplicate_path = duplicate_recipe_path(output_root, recipe_id)
    duplicate_without_other_errors = bool(duplicate_path) and not errors
    if duplicate_path and errors:
        errors.append(f"duplicate recipe_id already exists: {recipe_id} at {duplicate_path}")

    req["required_terms"] = required_terms
    req["missing_terms"] = missing_terms
    req["duplicate_path"] = duplicate_path
    req["errors"] = errors
    req["policy_status"] = "duplicate" if duplicate_without_other_errors else ("invalid" if errors else "accepted")
    req["valid"] = bool(req.get("valid")) and not errors and not duplicate_without_other_errors

    req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not req["valid"]:
        write_rejection_comment(Path(args.comment_file), req)
    write_outputs(args.github_output, req)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
