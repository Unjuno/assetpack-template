# Workflow guide

This directory contains GitHub Actions workflows for the assetpack template.

## Main workflow

The main template workflow is:

```text
assetpack-issue-generate.yml
```

It implements:

```text
Issue Form -> validation -> prompt recipe -> selected model -> artifact -> Issue feedback
```

## Behavior

The workflow:

1. runs for Issues with the `asset-request` label;
2. parses the structured Issue Form body;
3. validates the request against `assetpack.yml`;
4. comments validation feedback on the Issue;
5. runs image generation only when validation passes;
6. uploads request, report, config, and generated PNGs as artifacts;
7. comments successful generation feedback on the Issue.

## Model selection

Workflows should read model selection from `assetpack.yml`:

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

## Historical workflows

Some historical experiment workflows may still exist while cleanup continues. Derived repositories do not need them for normal use.

## Artifact policy

Generated PNGs should normally stay as GitHub Actions artifacts.

If a derived repository needs published assets, add a separate review and publish workflow.
