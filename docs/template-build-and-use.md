# Template build and use guide

This guide explains how to build a derived repository from this template and how to use the Issue-driven asset generation flow.

## What this template provides

This template turns GitHub Issues into committed prompt/image records.

```text
Issue request
  -> label gate
  -> policy validation
  -> mechanical prompt
  -> selected image model
  -> committed asset record
  -> Issue comment
```

Generated records are committed under:

```text
assets/generated/
```

Each generated asset record contains:

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

## Repository layout

Important paths:

```text
assetpack.yml
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
.github/workflows/assetpack-tests.yml
scripts/
tests/
docs/
assets/generated/
```

## Build a derived repository

1. Create a new repository from this template.
2. Reset template-generated sample records:

   ```bash
   python scripts/reset_template_generated_assets.py --yes
   git add assets/generated
   git commit -m "Reset generated assets for this repository"
   ```

   Use the dry run first if you want to inspect the affected paths:

   ```bash
   python scripts/reset_template_generated_assets.py
   ```

3. Enable GitHub Actions.
4. Create the request label configured in `assetpack.yml`.
5. Review the generator and prompt policy settings.
6. Run the lightweight test workflow.
7. Submit one smoke Issue and confirm the configured request label is present.
8. Confirm that `assets/generated/` receives a committed record.

## Configure `assetpack.yml`

Review these sections first:

```yaml
theme:
  description: ...

prompt_recipe:
  template: ...

prompt_policy:
  required_terms: ...

input_policy:
  ascii_only: true
  ascii_fields: ...

issue_generation:
  required_label: asset-request
  committed_output_root: assets/generated

models:
  image_generation:
    default_model_id: sdxl-turbo-quality
    allowed_model_ids:
      - sdxl-turbo-quality
      - ssd-1b-lcm-lora-quality
```

Keep `prompt_policy.allow_free_prompt` disabled if the repository should remain structured and reproducible.

## Configure the request label

The default request label is:

```text
asset-request
```

The generation workflow ignores Issues without this label.

## Submit a generation request

Create a GitHub Issue using these sections:

```md
### Subject
sleeping fox

### Scene
sitting on a white rug

### Audience
children

### Constraints
simple composition, friendly pose

### Model
default

### License
CC0-1.0
```

Confirm that the configured request label is present. If the Issue template did not add it automatically, apply it after checking the fields.

Expected result:

1. The workflow validates the Issue.
2. The workflow builds a final prompt from the structured fields.
3. The selected model generates one image.
4. The workflow commits the image and metadata under `assets/generated/`.
5. The Issue receives a comment with the committed path.

## Read a generated asset

Open the generated directory:

```text
assets/generated/issue-000001/<recipe_id>/
```

Use:

- `image.png` for the generated image.
- `prompt.txt` for the final prompt.
- `request.json` for normalized Issue input.
- `metadata.json` for committed path, model id, image size, seed, steps, image hash, and timing fields when available.
- `report.json` for generation details.

## Duplicate requests

The same structured request resolves to the same `recipe_id`.

If that `recipe_id` already exists under `assets/generated/`, the workflow does not generate a duplicate image. It comments with the existing asset path.

## Failed requests

The workflow comments back to the Issue when validation or generation fails.

Common validation failures:

- Missing required field.
- Invalid license.
- Invalid model id.
- URL in a structured field.
- Unknown non-empty section.
- Non-ASCII text in a checked field.
- Missing required term in the final prompt.

Common generation failures:

- Model download error.
- Timeout.
- Dependency failure.
- Runner resource limit.

Use the workflow artifact and `report.json` to inspect generation failures.

## Run tests

The lightweight test workflow runs without image generation:

```text
.github/workflows/assetpack-tests.yml
```

It validates policy, comment generation, asset staging, and generated asset index behavior.

For local development, run:

```bash
python -m pip install pytest pyyaml
pytest -q
```

## Operational checklist

Before accepting normal requests in a derived repository:

- Confirm generated sample records from the template have been reset.
- Confirm `assetpack.yml` matches the target theme.
- Confirm the request label exists.
- Confirm Actions are enabled.
- Confirm the workflow token can write contents and Issue comments.
- Submit one valid smoke Issue.
- Confirm `assets/generated/README.md` lists the generated record.
- Confirm duplicate re-submission comments with the existing asset path.
