# Technical design

This repository is a template for Issue-driven prompt/image asset generation.

The intended flow is:

```text
structured Issue input -> validation -> recipe -> mechanical prompt -> dedupe -> image generation -> committed asset record -> Issue feedback
```

Generated images and their prompt metadata are committed records under `assets/generated/`. The source of truth for each generated asset is the complete record directory: image, prompt, request, report, metadata, and README.

## Main technologies

| Area | Technology |
|---|---|
| Automation | GitHub Actions |
| Configuration | YAML |
| Runner language | Python 3.11 |
| Image generation | PyTorch CPU and Diffusers |
| LoRA model option | Diffusers LoRA loading |
| Reports | JSON |
| Generated records | Git commits under `assets/generated/` |

## Stable template contract

The stable template contract is `assetpack.yml`.

It defines:

- repository identity;
- locked theme;
- generation policy;
- Issue generation policy;
- prompt policy;
- deduplication policy;
- committed asset storage policy;
- license policy;
- selected image model configuration.

## Prompt pipeline concept

Prompt creation is mechanical by default.

```text
Issue structured input
  -> parse fields
  -> validate theme and policy
  -> apply required terms
  -> check configured term policy
  -> build prompt from a template
  -> normalize recipe
  -> deduplicate
  -> run generation when enabled
```

The template deliberately avoids arbitrary prompt execution as its default mode.

## Issue-driven path

The current template path is:

```text
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
scripts/validate_issue_request.py
scripts/validate_issue_policy.py
scripts/run_issue_safe.py
scripts/run_issue_asset_generation.py
scripts/run_issue_image_generation.py
scripts/prepare_committed_asset.py
scripts/write_issue_generation_comment.py
```

The workflow validates Issue fields, comments validation feedback on the Issue, runs generation only for valid requests, commits generated records, and comments success or failure back to the Issue.

## Image generation runner

The Issue-facing runner entry point is:

```text
scripts/run_issue_image_generation.py
```

It supports:

- Diffusers text-to-image pipelines;
- Diffusers LoRA text-to-image pipelines;
- load-only checks when used by lower-level runner code;
- per-candidate timeouts;
- report writing to `report.json`;
- image artifact writing under the workflow output directory.

Generation reports are operational records. They should not be converted into broad model-quality claims without separate evidence.

## Current selected image models

The selected model pair is:

```text
primary:   sdxl-turbo-quality
alternate: ssd-1b-lcm-lora-quality
```

The primary is selected as the default generation model. The alternate remains configured as an allowed fallback option.

Runtime selection reads these fields from `assetpack.yml`:

```text
models.image_generation.default_model_id
models.image_generation.allowed_model_ids
models.image_generation.runtime_override.environment_variable
```

## Storage layout

Committed generated records live under:

```text
assets/generated/issue-<number>/<recipe_id>/
```

Each record contains:

```text
image.png
prompt.txt
negative_prompt.txt
request.json
metadata.json
report.json
README.md
```

The generated asset index is:

```text
assets/generated/README.md
```
