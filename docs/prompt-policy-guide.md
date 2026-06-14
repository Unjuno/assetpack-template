# Prompt policy guide

This repository builds prompts from structured Issue fields. It does not use free-form prompt text directly.

## Goals

- Keep requests structured.
- Keep validated fields ASCII-only by default.
- Append required terms mechanically.
- Stop generation when a final prompt is missing a required term.
- Detect duplicate recipes before generation.
- Commit the final prompt with the generated image.

## Required terms

Required terms are configured in `assetpack.yml` under `prompt_policy.required_terms`.

The default terms are:

```text
black outline
white background
closed regions
no text
```

The policy check runs against the final prompt. If any required term is missing, no image is generated.

## Issue fields

The standard request sections are:

```text
### Subject
### Scene
### Audience
### Constraints
### Model
### License
```

The workflow parses those sections and writes the normalized request to `request.json`.

## ASCII-only inputs

`input_policy.ascii_only` is enabled by default. The default checked fields are `subject`, `scene`, `audience`, and `constraints`.

Keep those fields in plain ASCII.

## Model selection

The selected model must be listed in `models.image_generation.allowed_model_ids`.

Use `default` in the Model field to use `models.image_generation.default_model_id`.

## Duplicate recipes

The `recipe_id` is derived from structured fields, selected model, and final prompt.

If a matching record already exists under `assets/generated/`, the workflow comments with the existing asset path instead of generating another image.
