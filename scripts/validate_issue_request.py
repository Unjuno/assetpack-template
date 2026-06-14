#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

LABELS = {
    "subject": "subject",
    "scene": "scene",
    "audience": "audience",
    "constraints": "constraints",
    "model": "model",
    "license": "license",
}


def key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")


def parse_body(text: str) -> dict[str, str]:
    current = None
    data: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            current = key(m.group(1))
            data.setdefault(current, [])
            continue
        if current:
            data[current].append(line)
    out = {}
    for k, lines in data.items():
        v = "\n".join(lines).strip()
        out[k] = "" if v == "_No response_" else v
    return out


def terms(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [str(x) for x in data.get("terms", [])]


def selected_model(cfg: dict[str, Any], value: str) -> str:
    model_cfg = cfg["models"]["image_generation"]
    value = value.strip()
    if not value or value.lower() == "default":
        return model_cfg["default_model_id"]
    return value


def make_prompt(cfg: dict[str, Any], fields: dict[str, str], model_id: str) -> dict[str, str]:
    recipe_cfg = cfg.get("prompt_recipe", {})
    policy = cfg.get("prompt_policy", {})
    required_terms = ", ".join(policy.get("required_terms", []))
    constraints = fields.get("constraints", "").strip() or recipe_cfg.get("empty_constraints", "simple composition")
    template = recipe_cfg.get("template", "{subject}, {scene}, for {audience}, {constraints}, {required_terms}")
    prompt = template.format(
        subject=fields.get("subject", "").strip(),
        scene=fields.get("scene", "").strip(),
        audience=fields.get("audience", "").strip(),
        constraints=constraints,
        required_terms=required_terms,
        theme_description=cfg.get("theme", {}).get("description", ""),
    )
    source = json.dumps({"fields": fields, "model": model_id, "prompt": prompt}, ensure_ascii=False, sort_keys=True)
    recipe_id = f"{recipe_cfg.get('recipe_id_prefix', 'assetpack')}-{hashlib.sha256(source.encode()).hexdigest()[:16]}"
    return {
        "recipe_id": recipe_id,
        "prompt": prompt,
        "negative_prompt": recipe_cfg.get("negative_prompt", "low quality, text, watermark, logo"),
    }


def validate(cfg: dict[str, Any], raw: dict[str, str]) -> dict[str, Any]:
    fields = {LABELS.get(k, k): v for k, v in raw.items()}
    issue_cfg = cfg.get("issue_generation", {})
    errors: list[str] = []
    warnings: list[str] = []

    if cfg.get("theme", {}).get("locked") is not True:
        errors.append("theme.locked must be true")
    if cfg.get("prompt_policy", {}).get("mechanical_only") is not True:
        errors.append("prompt_policy.mechanical_only must be true")
    if cfg.get("prompt_policy", {}).get("allow_free_prompt") is not False:
        errors.append("prompt_policy.allow_free_prompt must be false")

    required = issue_cfg.get("form_fields", {}).get("required", ["subject", "scene", "audience", "license"])
    optional = issue_cfg.get("form_fields", {}).get("optional", ["constraints", "model"])
    allowed = set(required) | set(optional)
    for name in required:
        if not fields.get(name, "").strip():
            errors.append(f"missing field: {name}")
    if issue_cfg.get("reject_unknown_nonempty_sections", True):
        for name, value in fields.items():
            if name not in allowed and value.strip():
                errors.append(f"unknown field: {name}")

    for name, limit in issue_cfg.get("field_limits", {}).items():
        if len(fields.get(name, "")) > int(limit):
            errors.append(f"field too long: {name}")

    if issue_cfg.get("reject_urls", True):
        for name, value in fields.items():
            if URL_RE.search(value):
                errors.append(f"URL found in field: {name}")

    text = "\n".join(fields.values()).lower()
    for term in terms(cfg.get("prompt_policy", {}).get("banned_terms_file")):
        if term.lower() in text:
            errors.append(f"configured term found: {term}")

    if fields.get("license") and fields["license"] not in cfg.get("license", {}).get("allowed", []):
        errors.append(f"license not configured: {fields['license']}")

    model_id = selected_model(cfg, fields.get("model", ""))
    allowed_models = cfg["models"]["image_generation"].get("allowed_model_ids", [])
    if model_id not in allowed_models:
        errors.append(f"model not configured: {model_id}")

    if not fields.get("constraints", "").strip():
        warnings.append("constraints was empty; default constraints were used")

    prompt_data = make_prompt(cfg, fields, model_id)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "fields": fields,
        "selected_model_id": model_id,
        "allowed_model_ids": allowed_models,
        **prompt_data,
    }


def comment(result: dict[str, Any]) -> str:
    if result["valid"]:
        return (
            "## Asset request accepted\n\n"
            "The structured request passed validation. CI will attempt image generation.\n\n"
            f"- recipe: `{result['recipe_id']}`\n"
            f"- model: `{result['selected_model_id']}`\n\n"
            "On success, the generated image, prompt, request, metadata, and report are committed under `assets/generated/`.\n"
        )
    lines = "\n".join(f"- {e}" for e in result.get("errors", []))
    return (
        "## Asset request not generated\n\n"
        "The structured request did not pass repository policy.\n\n"
        f"{lines}\n\n"
        "Edit the Issue form fields and keep the request aligned with this repository's fixed theme.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="assetpack.yml")
    ap.add_argument("--issue-body-file", required=True)
    ap.add_argument("--issue-number", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--github-output", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw = parse_body(Path(args.issue_body_file).read_text(encoding="utf-8"))
    result = validate(cfg, raw)
    result["issue_number"] = args.issue_number

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "validation-comment.md").write_text(comment(result), encoding="utf-8")

    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as f:
            f.write(f"valid={'true' if result['valid'] else 'false'}\n")
            f.write(f"selected_model_id={result['selected_model_id']}\n")
            f.write(f"recipe_id={result['recipe_id']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
