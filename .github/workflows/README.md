# Workflow guide

This directory contains two workflow groups.

## Template-facing workflows

Production-facing workflows should use `assetpack.yml` as the source of truth.

A derived repository should eventually use workflows that:

1. parse structured input;
2. validate `assetpack.yml`;
3. build a prompt recipe;
4. deduplicate recipes;
5. select a configured image model;
6. run a small generation check when enabled;
7. upload generated images as artifacts;
8. write compact reports.

## Experiment workflows

The `image-model-ci-*` workflows are experiment and evidence workflows.

They were used to test candidate models, collect artifacts, run subject-transfer checks, and select the current model pair.

They are retained for reproducibility. They are not the main user interface for a derived assetpack repository.

Related records:

```text
experiments/README.md
docs/ci/README.md
docs/ci/image-model-final-selection.md
```

## Current model selection

New production workflows should read model selection from `assetpack.yml`:

```text
models.image_generation.default_model_id
models.image_generation.alternate_model_id
models.image_generation.allowed_model_ids
models.image_generation.runtime_override.environment_variable
```

Current default:

```text
sdxl-turbo-quality
```

Current alternate:

```text
ssd-1b-lcm-lora-quality
```

## Artifact policy

Generated PNGs should normally stay as GitHub Actions artifacts.

If a derived repository needs published assets, add a separate review and publish workflow.
