# Issue-driven image generation

This repository is a template for CI-driven image generation.

The user opens a GitHub Issue with structured fields. GitHub Actions validates those fields, builds a prompt recipe from repository configuration, and runs image generation only when the request matches the repository policy.

## Flow

```text
GitHub Issue
  -> structured form fields
  -> policy validation
  -> prompt recipe
  -> selected model
  -> CI image generation
  -> artifact upload
  -> Issue feedback
```

## Fixed-template design

This is not a free prompt box.

A derived repository should set one fixed concept in `assetpack.yml`. The Issue supplies only slots such as:

```text
subject
scene
audience
constraints
license
model
```

The final prompt is built mechanically from:

```text
assetpack.yml
prompt_recipe.template
prompt_policy.required_terms
```

## Current files

| File | Role |
|---|---|
| `.github/ISSUE_TEMPLATE/generate.yml` | Structured Issue form. |
| `.github/workflows/assetpack-issue-generate.yml` | Issue-driven CI workflow. |
| `scripts/validate_issue_request.py` | Parses and validates Issue fields. |
| `scripts/run_issue_asset_generation.py` | Builds a temporary generation config and runs the selected model. |
| `scripts/write_issue_generation_comment.py` | Writes a result comment body for workflow use. |
| `assetpack.yml` | Main policy and model configuration. |

## Validation

The validation step checks the repository policy before image generation runs.

It verifies:

- locked theme;
- mechanical prompt policy;
- required Issue fields;
- field length limits;
- configured license values;
- configured image model values;
- configured term policy;
- URL policy.

When validation does not pass, the workflow comments on the Issue with the reason and does not run image generation.

## Generation

When validation passes, CI uses the configured model.

Current default:

```text
sdxl-turbo-quality
```

Current alternate:

```text
ssd-1b-lcm-lora-quality
```

The model may be selected through the Issue form or by the runtime key:

```text
ASSETPACK_IMAGE_MODEL_ID
```

## Artifacts

Generated images are uploaded as GitHub Actions artifacts.

They are not committed to Git by default.

Artifact naming pattern:

```text
assetpack-issue-<issue-number>-<recipe-id>
```

## Customizing a derived repository

A derived repository should customize:

- `theme` in `assetpack.yml`;
- `prompt_recipe.template`;
- `prompt_policy.required_terms`;
- constraints and corpus files;
- Issue Form fields if the slot model changes;
- model policy only when new evidence supports the change.

The repository controls the final prompt. The Issue supplies structured slots only.
