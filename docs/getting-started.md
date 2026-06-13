# Getting started

This repository is a template for fixed-theme, CI-driven AI asset generation.

It is not a general prompt playground. A derived repository should lock one asset concept, accept structured input, build prompts mechanically, and keep generation results evidence-gated.

## 1. Create a derived repository

Use this repository as a template, then edit `assetpack.yml` first.

The minimum derived configuration should define:

```yaml
assetpack:
  id: my-assetpack
  title: "My Assetpack"
  version: 0.1.0

theme:
  locked: true
  category: game_item_icons
  description: "Small fantasy item icons for a fixed visual style"
```

`theme.locked: true` is intentional. A derived repository should be about one repeatable asset concept, not arbitrary image generation.

## 2. Configure generation policy

Generation settings live in `assetpack.yml`:

```yaml
generation:
  image_generation: true
  mode: smoke_test
  allow_commit_generated_images: false
  width: 256
  height: 256
  images_per_prompt: 1
  timeout_minutes_per_model: 20
```

Generated images should normally remain GitHub Actions artifacts. Do not commit large generated image batches to Git unless a separate human review and publishing process is added.

## 3. Configure prompt policy

Prompt generation should be mechanical by default:

```yaml
prompt_policy:
  mechanical_only: true
  allow_free_prompt: false
  required_terms:
    - "black outline"
    - "white background"
  banned_terms_file: constraints/banned_terms.json
```

Users provide structured slots, such as `subject`, `scene`, `audience`, and `constraints`. The repository builds the final prompt from templates, constraints, and corpus files.

## 4. Use the selected image model

The current selected image-generation configuration is stored in `assetpack.yml`.

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

Only configured values should be accepted. Unknown model IDs should be rejected by future production generation workflows.

## 5. Keep evidence, not image dumps

The template separates:

- configuration: `assetpack.yml`;
- prompt templates: `templates/`;
- constraints: `constraints/`;
- experiments: `experiments/`;
- scripts: `scripts/`;
- evidence records: `docs/ci/`;
- temporary generated images: GitHub Actions artifacts.

The final image-model decision is recorded in:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

## 6. What to implement next in a derived repository

A derived repository should add or customize:

1. issue form for structured generation requests;
2. prompt templates for the fixed theme;
3. corpus files for allowed subjects and modifiers;
4. validation for required and banned terms;
5. recipe generation;
6. deduplication;
7. a production smoke-test workflow wired to `assetpack.yml` model selection;
8. a human review/publish step if generated images are to be kept.
