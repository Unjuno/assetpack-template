# Configuration reference

The primary configuration file is:

```text
assetpack.yml
```

This file is the intended stable contract for derived repositories. Experiment-specific YAML files under `experiments/` are evidence and test inputs, not the main template API.

## Top-level sections

```yaml
assetpack:
  id: assetpack-template
  title: "Assetpack Template"
  version: 0.1.0

theme:
  locked: true
  category: coloring_lineart
  description: "Fixed-theme prompt recipe generation and CI image-generation experiments"

generation:
  image_generation: true
  mode: smoke_test
  allow_commit_generated_images: false
  width: 256
  height: 256
  images_per_prompt: 1
  timeout_minutes_per_model: 20
```

## `assetpack`

Repository identity.

| Key | Meaning |
|---|---|
| `id` | Stable machine-readable assetpack ID. |
| `title` | Human-readable title. |
| `version` | Template or derived assetpack version. |

## `theme`

The fixed concept for the derived repository.

| Key | Meaning |
|---|---|
| `locked` | Must be `true` for template-aligned generation. |
| `category` | Broad asset category, such as `coloring_lineart`, `game_item_icons`, or `blog_ogp`. |
| `description` | Human-readable description of the locked concept. |

A derived repository should not accept arbitrary prompt categories. If the concept changes, create a new derived repository or explicitly version the assetpack.

## `generation`

Controls whether image generation is enabled and how generated images should be treated.

| Key | Meaning |
|---|---|
| `image_generation` | Enables CI generation workflows when true. |
| `mode` | Current default mode. `smoke_test` means lightweight validation, not publication. |
| `allow_commit_generated_images` | Should normally remain `false`. Generated images should stay as artifacts unless reviewed. |
| `width`, `height` | Default image size for production-oriented generation. Experiments may use their own sizes. |
| `images_per_prompt` | Default number of images per recipe. |
| `timeout_minutes_per_model` | Budget guard for slow runners. |

## `prompt_policy`

```yaml
prompt_policy:
  mechanical_only: true
  allow_free_prompt: false
  required_terms:
    - "black outline"
    - "white background"
    - "closed regions"
    - "no text"
  banned_terms_file: constraints/banned_terms.json
```

| Key | Meaning |
|---|---|
| `mechanical_only` | Final prompts are assembled from templates and structured fields. |
| `allow_free_prompt` | Should default to `false`; free-form prompt execution is not the template concept. |
| `required_terms` | Terms that must appear in the generated prompt. |
| `banned_terms_file` | JSON file containing terms or patterns that must be rejected. |

## `dedupe`

Deduplication policy for generated prompt recipes.

```yaml
dedupe:
  exact: true
  canonical_slots: true
  normalized_prompt_hash: true
  near_duplicate:
    enabled: true
    method: jaccard_shingles
    shingle_size: 3
    threshold: 0.85
```

The dedupe layer is intended to prevent repeated or near-identical recipes from producing redundant asset batches.

## `records`

```yaml
records:
  keep_summary: true
  keep_failure_reason: true
  keep_artifacts: false
  artifact_retention_days: 7
```

Keep records and failure reasons. Do not keep large generated assets in Git by default.

## `license`

Defines allowed output licensing values for structured requests.

```yaml
license:
  allowed:
    - CC0-1.0
    - CC-BY-4.0
    - OWNED
  default: CC0-1.0
```

## `models.image_generation`

Current evidence-based model selection.

```yaml
models:
  image_generation:
    mode: fixed_default_with_configurable_alternate
    default_model_id: sdxl-turbo-quality
    alternate_model_id: ssd-1b-lcm-lora-quality
    allowed_model_ids:
      - sdxl-turbo-quality
      - ssd-1b-lcm-lora-quality
    runtime_override:
      enabled: true
      environment_variable: ASSETPACK_IMAGE_MODEL_ID
```

| Key | Meaning |
|---|---|
| `default_model_id` | The default model production workflows should use. |
| `alternate_model_id` | A manually selectable alternate. |
| `allowed_model_ids` | The complete allowlist for production model selection. |
| `runtime_override.environment_variable` | Environment variable that may select the alternate at runtime. |
| `selection_basis` | Evidence record linking this choice to final-runoff CI results. |

Unknown model IDs should be rejected by production generation code.

## `models.candidates`

`models.candidates` describes the selected and historical candidates.

Current selected pair:

| Candidate | Status | Role |
|---|---|---|
| `sdxl-turbo-quality` | selected | default image generation |
| `ssd-1b-lcm-lora-quality` | selected | configurable alternate |

`segmind-vega-quality` is preserved as a finalist that was not selected after final runoff. Legacy smoke candidates are disabled but retained for template history.
