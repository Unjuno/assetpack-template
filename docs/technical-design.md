# Technical design

This repository is a template for fixed-theme asset generation.

The intended flow is:

```text
structured input -> validation -> recipe -> mechanical prompt -> dedupe -> CI generation check -> evidence record -> optional review
```

Generated images are not the source of truth. Configuration, recipes, reports, and evidence records are the source of truth.

## Main technologies

| Area | Technology |
|---|---|
| Automation | GitHub Actions |
| Configuration | YAML |
| Runner language | Python 3.11 |
| Image generation | PyTorch CPU and Diffusers |
| LoRA model option | Diffusers LoRA loading |
| Reports | JSON |
| Generated images | GitHub Actions artifacts |

## Stable template contract

The stable template contract is `assetpack.yml`.

It defines:

- repository identity;
- locked theme;
- generation policy;
- Issue generation policy;
- prompt policy;
- deduplication policy;
- record retention policy;
- license policy;
- selected image model configuration.

Experiment files under `experiments/` are historical model-selection records. They are not the main template API.

## Prompt pipeline concept

Prompt creation should be mechanical by default.

```text
Issue or local structured input
  -> parse fields
  -> validate theme and policy
  -> apply required terms
  -> check configured term policy
  -> build prompt from a template
  -> normalize recipe
  -> deduplicate
  -> run generation check when enabled
```

The template deliberately avoids arbitrary prompt execution as its default mode.

## Issue-driven CI path

The current template path is:

```text
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
scripts/validate_issue_request.py
scripts/run_issue_asset_generation.py
scripts/write_issue_generation_comment.py
```

The workflow validates Issue fields, comments validation feedback on the Issue, runs generation only for valid requests, uploads artifacts, and comments successful generation feedback.

## Image generation runner

The shared runner is:

```text
scripts/run_image_model_ci_benchmark.py
```

It supports:

- Diffusers text-to-image pipelines;
- Diffusers LoRA text-to-image pipelines;
- load-only checks;
- per-candidate timeouts;
- report writing to `report.json`;
- image artifact writing under the report directory.

CI results are measurements. They should not be converted into broad model-quality claims without evidence.

## Current selected image models

The selected model pair is:

```text
primary:   sdxl-turbo-quality
alternate: ssd-1b-lcm-lora-quality
```

The primary is selected for target identity and topology preservation. The alternate is kept for visual asset quality and style-sensitive use.

Evidence lives in:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

Runtime selection should read these fields from `assetpack.yml`:

```text
models.image_generation.default_model_id
models.image_generation.allowed_model_ids
models.image_generation.runtime_override.environment_variable
```

## Workflow classes

There are two workflow classes.

### Template workflows

These validate structured input, build recipes, run the selected image model, upload artifacts, and provide Issue feedback.

These workflows should use `assetpack.yml` as their source of truth.

### Historical experiment workflows

The `image-model-ci-*` workflows are model-selection and stress-test workflows.

They can be removed from derived repositories after the final model-selection evidence has been copied or retained.

## Artifact policy

Generated images are temporary artifacts by default.

Commit:

- configuration;
- prompt recipes;
- validation logic;
- compact reports;
- evidence summaries;
- model-selection decisions.

Do not commit by default:

- generated image batches;
- model weights;
- cache directories;
- raw logs that are already summarized.

## Evidence discipline

Model selection should distinguish:

- CI success: the workflow ran and produced a report or image;
- visual acceptance: the output is useful for the target concept;
- production selection: the model is configured as allowed/default for the template.

The current model selection reached production-selection status after subject-transfer tests and a final 20-subject runoff.
