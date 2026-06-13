# assetpack-template

Template repository for fixed-theme, CI-driven AI asset recipe generation.

This repository is not a general prompt playground. It is a template for building derived assetpack repositories where the theme is fixed, prompts are generated mechanically, generated images are kept as CI artifacts by default, and model choices are evidence-gated.

## What this template is for

Use this template when you want a repository that can:

1. lock one asset concept;
2. accept structured generation input;
3. build prompt recipes from templates and constraints;
4. deduplicate generated recipes;
5. run lightweight image-generation checks in GitHub Actions;
6. keep compact evidence records;
7. select practical image models based on recorded results.

Examples of derived repositories:

```text
coloring_lineart_animals
game_item_icons_fantasy
blog_ogp_science
ui_icons_minimal
```

## What this template is not for

This repository does not aim to:

- host a general image-generation service;
- accept arbitrary free-form prompts by default;
- commit generated image batches to Git;
- store model weights or caches;
- claim general model superiority from one CI run;
- publish generated images without review.

## Core concept

```text
structured input
  -> validation
  -> mechanical prompt recipe
  -> deduplication
  -> optional CI image-generation check
  -> artifact upload
  -> compact evidence record
  -> optional human review / publish step
```

The stable template contract is:

```text
assetpack.yml
```

Experiment YAML files under `experiments/` are reproducibility records. They are not the main template API.

## Current selected image models

The current image-generation model decision is recorded in `assetpack.yml` and `docs/ci/`.

Default model:

```text
sdxl-turbo-quality
```

Configurable alternate:

```text
ssd-1b-lcm-lora-quality
```

Runtime override:

```bash
ASSETPACK_IMAGE_MODEL_ID=ssd-1b-lcm-lora-quality
```

Allowed production model IDs are listed under:

```text
models.image_generation.allowed_model_ids
```

Evidence:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

The final selection came from `image-model-ci-final-runoff-hard-subjects` run `27446158099`, which tested 3 finalists across 20 difficult subjects for 60 expected images. This is a repository-specific asset-generation decision, not a universal benchmark claim.

## Repository map

| Path | Purpose |
|---|---|
| `assetpack.yml` | Main template configuration and selected model policy. |
| `templates/` | Prompt templates for mechanical prompt construction. |
| `constraints/` | Required, disallowed, and validation-related constraints. |
| `corpus/` | Structured subject, scene, and modifier inputs. |
| `recipes/` | Generated and approved prompt recipes. |
| `experiments/` | Model-selection and stress-test configuration history. |
| `scripts/` | Python utilities for validation, generation checks, and evidence probes. |
| `.github/workflows/` | GitHub Actions workflows. Includes both template-facing and experiment workflows. |
| `docs/` | Template guides, technical design notes, and CI evidence records. |

## Documentation

Start here:

```text
docs/getting-started.md
docs/configuration.md
docs/technical-design.md
docs/ci/README.md
experiments/README.md
.github/workflows/README.md
scripts/README.md
```

Model-selection evidence:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

## Minimum derived repository setup

After creating a derived repository from this template, edit `assetpack.yml`:

```yaml
assetpack:
  id: my-assetpack
  title: "My Assetpack"
  version: 0.1.0

theme:
  locked: true
  category: game_item_icons
  description: "Small fantasy item icons for a fixed visual style"

generation:
  image_generation: true
  mode: smoke_test
  allow_commit_generated_images: false
  width: 256
  height: 256
  images_per_prompt: 1
  timeout_minutes_per_model: 20

prompt_policy:
  mechanical_only: true
  allow_free_prompt: false
  required_terms:
    - "black outline"
    - "white background"
  banned_terms_file: constraints/banned_terms.json

models:
  image_generation:
    mode: fixed_default_with_configurable_alternate
    default_model_id: sdxl-turbo-quality
    alternate_model_id: ssd-1b-lcm-lora-quality
    allowed_model_ids:
      - sdxl-turbo-quality
      - ssd-1b-lcm-lora-quality
```

## Technical stack

| Area | Technology |
|---|---|
| CI | GitHub Actions |
| Runtime scripts | Python 3.11 |
| Image generation experiments | PyTorch CPU + Diffusers |
| LoRA candidates | Diffusers LoRA loading |
| Configuration | YAML |
| Reports | JSON |
| Generated images | GitHub Actions artifacts |
| ONNX probes | ONNX Runtime CPU, evidence-gated |

## Artifact policy

Generated images are experiment or smoke-test artifacts. They should not be committed to Git by default.

Commit:

- configuration;
- prompt recipes;
- validation logic;
- compact reports;
- evidence summaries;
- model-selection decisions.

Keep generated images as artifacts unless a derived repository adds an explicit review and publish workflow.

## Current status

The repository has completed an evidence-gated image-model selection for the current template concept.

Next implementation steps:

1. add or finalize the Issue Form for structured generation requests;
2. add validation workflow around `assetpack.yml`;
3. implement recipe builder and dedupe flow;
4. wire production generation to `models.image_generation.default_model_id`;
5. enforce `ASSETPACK_IMAGE_MODEL_ID` against `allowed_model_ids`;
6. keep generated images as temporary artifacts unless reviewed for publication.
