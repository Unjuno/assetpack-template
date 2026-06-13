# Scripts

This directory contains Python utilities for the CI-driven asset-generation template.

The stable configuration is `assetpack.yml`.

## Main scripts

| Script | Purpose |
|---|---|
| `validate_issue_request.py` | Parses and validates a structured Issue request. |
| `run_issue_asset_generation.py` | Runs generation for a validated Issue request. |
| `write_issue_generation_comment.py` | Writes a result comment body for the Issue workflow. |
| `select_image_model.py` | Resolves the configured model from `assetpack.yml`. |
| `run_image_model_ci_benchmark.py` | Shared Diffusers runner used by Issue generation. |
| `run_diffusers_smoke.py` | Lightweight Diffusers smoke-test helper. |

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

## Issue generation

`run_issue_asset_generation.py` reads `request.json`, resolves the selected model, writes a temporary generation config, and calls `run_image_model_ci_benchmark.py`.

It writes a compact `report.json` and any generated PNG under the workflow artifact directory.

## Shared runner

`run_image_model_ci_benchmark.py` reads experiment-style YAML and writes a JSON report plus generated image files.

Supported methods:

```text
diffusers_text_to_image
diffusers_load_only
diffusers_lora_text_to_image
diffusers_lora_load_only
```

Candidate timeout can be set with:

```text
IMAGE_MODEL_CANDIDATE_TIMEOUT_SECONDS
```

or:

```text
--candidate-timeout-seconds
```

Generated images should remain workflow artifacts unless a derived repository adds a review and publish step.
