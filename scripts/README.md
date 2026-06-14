# Scripts

This directory contains Python utilities for the Issue-driven asset-generation template.

The stable configuration is `assetpack.yml`.

## Main scripts

| Script | Purpose |
|---|---|
| `validate_issue_request.py` | Parses and validates a structured Issue request. |
| `validate_issue_policy.py` | Enforces repository policy before generation. |
| `check_issue_label.py` | Checks whether an Issue has the configured request label. |
| `run_issue_safe.py` | Runs guarded generation and returns a failing exit code on generation failure. |
| `run_issue_asset_generation.py` | Resolves the selected model and prepares the generation config. |
| `run_issue_image_generation.py` | Issue-generation runner entry point. |
| `prepare_committed_asset.py` | Stages generated image, prompt, request, report, and metadata for Git commit. |
| `update_generated_assets_index.py` | Builds `assets/generated/README.md`. |
| `write_issue_generation_comment.py` | Writes a result comment body for the Issue workflow. |
| `select_image_model.py` | Resolves the configured model from `assetpack.yml`. |

## Model selection helper

Default resolution:

```bash
python scripts/select_image_model.py --pretty
```

Explicit model selection:

```bash
python scripts/select_image_model.py --model-id ssd-1b-lcm-lora-quality --pretty
```

Environment override:

```bash
ASSETPACK_IMAGE_MODEL_ID=ssd-1b-lcm-lora-quality python scripts/select_image_model.py --pretty
```

Unknown model IDs and disabled candidates are rejected.

## Issue validation

`validate_issue_request.py` reads the Issue body produced by `.github/ISSUE_TEMPLATE/generate.yml`.

It writes:

```text
request.json
validation-comment.md
```

`validate_issue_policy.py` enforces ASCII-only fields, required terms, and duplicate recipe handling.

## Issue generation

`run_issue_safe.py` calls `run_issue_asset_generation.py` only after required terms are present in the final prompt. It returns a non-zero exit code when generation fails.

`run_issue_asset_generation.py` reads `request.json`, resolves the selected model, and writes a temporary generation config under the workflow output directory.

`run_issue_image_generation.py` runs the selected image generation candidate and writes `report.json` plus generated PNG files under the workflow output directory.

Candidate timeout can be set with:

```text
ASSETPACK_IMAGE_CANDIDATE_TIMEOUT_SECONDS
```

or with the runner argument:

```text
--candidate-timeout-seconds
```

## Committed asset records

`prepare_committed_asset.py` turns a successful workflow artifact into a committed record under:

```text
assets/generated/issue-<number>/<recipe_id>/
```

Each record contains the generated PNG, final prompt, request, report, metadata, and a README.
