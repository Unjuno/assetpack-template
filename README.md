# assetpack-template

Template repository for fixed-theme, CI-driven AI asset recipe generation, image-generation smoke tests, and reproducible prompt workflows.

## Purpose

`assetpack-template` is a template repository for creating fixed-theme AI asset generation repositories.

The intended workflow is:

1. A repository creator uses this template.
2. The creator fixes a theme, prompt template, required terms, banned terms, and generation policy.
3. Users contribute by either:
   - creating an Issue with structured prompt inputs, or
   - cloning the repository and running local commands.
4. GitHub Actions validates the input, builds a mechanical prompt recipe, removes duplicates, and runs image-generation experiments when enabled.
5. Experimental results are recorded so the best model can later be selected and fixed.

This repository is not a general free-form prompt site. It is a fixed-theme prompt and image-generation experiment framework.

## Core Requirements

### 1. Fixed Theme

Each derived repository must define one fixed theme.

Examples:

- `coloring_lineart_animals`
- `game_item_icons_fantasy`
- `blog_ogp_science`
- `ui_icons_minimal`

The theme must be configured in `assetpack.yml`.

```yaml
theme:
  locked: true
  category: coloring_lineart
  description: "Flood-fill friendly animal line art"
```

If `theme.locked` is not `true`, CI should reject generation jobs.

### 2. Mechanical Prompt Generation

Prompts must be generated mechanically from:

- fixed templates,
- structured Issue inputs,
- corpus files,
- required terms,
- banned terms,
- generation constraints.

Free-form prompts are not the default mode.

```yaml
prompt_policy:
  mechanical_only: true
  allow_free_prompt: false
```

The user does not directly control the final prompt string. The user supplies slots such as `subject`, `scene`, `audience`, or `constraints`. The repository template builds the final prompt.

### 3. Required Terms

A derived repository may require specific terms or constraints to appear in the generated prompt.

Example:

```yaml
prompt_policy:
  required_terms:
    - "black outline"
    - "white background"
    - "closed regions"
    - "no text"
```

CI must verify that the generated prompt contains the required terms before any image generation is attempted.

### 4. Issue-Based Generation

Users can create an Issue to request generation.

The Issue should not be treated as executable code. Issue content is untrusted input and must be parsed as structured data.

Example Issue fields:

```yaml
category: coloring_lineart
subject: sleeping cat
scene: simple sitting pose
audience: children
license: CC0-1.0
```

The workflow is:

```text
Issue input
  -> parse structured fields
  -> validate theme and constraints
  -> build prompt from template
  -> deduplicate recipe
  -> run image-generation smoke test if enabled
  -> record result
  -> comment back on the Issue
```

### 5. Clone-Based Usage

Users can also clone a derived repository and run commands locally.

Expected future commands:

```bash
git clone https://github.com/<owner>/<assetpack-repo>.git
cd <assetpack-repo>

assetpack validate-config
assetpack generate-recipes
assetpack dedupe
assetpack smoke-bench
assetpack report
```

The exact CLI is not fixed yet. The repository structure should be designed so these commands can be added later without changing the concept.

### 6. CI Image Generation

This template is intended to support CI-based image-generation experiments.

CI should be able to:

- validate the repository configuration,
- validate Issue input,
- build a prompt recipe,
- check required terms,
- check banned terms,
- remove duplicate recipes,
- run lightweight image-generation smoke tests,
- record model results,
- upload temporary artifacts,
- comment results back to the Issue.

Image generation should be controlled by configuration.

```yaml
generation:
  image_generation: true
  mode: smoke_test
  width: 256
  height: 256
  images_per_prompt: 1
  timeout_minutes_per_model: 20
```

Generated images are experiment artifacts. They should not be committed to Git by default.

### 7. Model Candidates

Model selection is evidence-gated. The project should test candidate models first, then fix the highest-quality practical model for each use case.

Initial candidate classes:

- Bonsai Image 4B
- Tiny-SD
- SD-Turbo
- BK-SDM Tiny / Small
- OnnxStream or stable-diffusion.cpp based runners
- other lightweight text-to-image models discovered later

The repository should store model test results so the final model decision is based on evidence, not preference.

#### Current image-generation model selection

The current evidence-based image-generation choice is configured in `assetpack.yml`.

Default model:

```text
sdxl-turbo-quality
```

Configurable alternate:

```text
ssd-1b-lcm-lora-quality
```

Runtime override key:

```text
ASSETPACK_IMAGE_MODEL_ID
```

Allowed override values:

```text
sdxl-turbo-quality
ssd-1b-lcm-lora-quality
```

Evidence records:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

The final runoff used `image-model-ci-final-runoff-hard-subjects` run `27446158099`: 3 candidates across 20 hard subjects, for 60 expected images. The selection prioritizes target identity, topology, small parts, and contact with the bonsai branch or pot over merely attractive images.

This is a repository-specific asset-generation decision, not a general benchmark claim.

#### Bonsai ONNX Runtime CPU experiment

`bonsai-onnx-smoke` is a CPU experiment. It does not use the official Bonsai GPU runtime.

The previous probe tried to export `prism-ml/bonsai-image-binary-4B-unpacked` as one complete Diffusers pipeline with `optimum-cli export onnx --library diffusers`. That path fails before runtime because the model repository does not provide `model_index.json` for whole-pipeline loading.

The current experiment is staged:

1. `layout_probe` inspects the Hugging Face repository file layout.
2. `component_export` exports an explicit component to ONNX. The first implemented target is `vae_decoder`.
3. `ort_cpu_forward` loads the exported ONNX model with ONNX Runtime CPU and runs one tensor forward pass.
4. `cpu_image_generation` is reserved for a later composed pipeline with text encoder, scheduler, transformer, and VAE.

The default config currently requires `ort_cpu_forward`. That means CI must at least produce an ONNX component and execute it on CPU. It is not yet a full text-to-image pass. Full Bonsai CPU image generation remains a separate milestone because it must compose multiple heavy components and may exceed GitHub-hosted runner RAM.

#### Bonsai low-bit verification evidence

Bonsai low-bit verification is evidence-gated. The canonical machine-readable source is:

```text
docs/bonsai-lowbit-verification-manifest.json
```

Supporting human-readable records are:

```text
docs/bonsai-lowbit-claim-matrix.md
docs/bonsai-lowbit-local-artifact-audit.md
docs/bonsai-lowbit-completion-plan.md
docs/bonsai-lowbit-scope-cap.md
```

Current verified low-bit scope, based only on manifest-recorded artifact reports:

- attention `to_out` path;
- single block modulated attention `to_out` residual;
- two single blocks modulated residual stack;
- four single blocks via two-by-two segmented ONNX;
- eight single blocks via four-by-two segmented ONNX;
- sixteen single blocks via eight-by-two segmented ONNX;
- all ten pair segments for blocks 0-19;
- twenty single blocks via ten-by-two chained segmented export probe;
- twenty single blocks via split persistent ten-by-two ONNX segment artifacts, validated as a reusable segmented ONNX Runtime CPU chain;
- ten-by-two chain-state handoff report;
- ten-by-two input boundary report.

Current pending low-bit scope:

- none for the current low-bit critical-path scope through ten-by-two input-boundary documentation.

Scope cap decision:

- the current low-bit critical-path scope is closed in `docs/bonsai-lowbit-scope-cap.md`;
- the strongest allowed claim is a reusable segmented ONNX Runtime CPU critical-path chain with persisted two-block ONNX artifacts, hidden-state handoff evidence, and hidden/temb-only external input-boundary evidence;
- any future expansion must start a new manifest key and evidence boundary.

The twenty-block ten-by-two evidence has four distinct verified boundaries:

1. `bonsai-lowbit-ten-by-two-chain-report-json` verifies a reproducible export probe. It exports temporary segmented ONNX files, validates them, and uploads JSON reports only.
2. `bonsai-lowbit-ten-by-two-split-persistent-onnx-validation-report-json` verifies persisted ONNX artifacts. It validates ten downloaded two-block ONNX segment artifacts plus a reference artifact without reloading the low-bit source.
3. `bonsai-lowbit-ten-by-two-chain-handoff-report-json` verifies the segment-to-segment tensor handoff evidence for the reusable segmented chain.
4. `bonsai-lowbit-ten-by-two-input-boundary-report-json` verifies that the current ONNX chain accepts only `hidden` and `temb` as external ONNX inputs and does not include prompt tokens, a text encoder, scheduler, VAE, image latents, or full Bonsai pipeline inputs.

Any future verification claim must be represented in `docs/bonsai-lowbit-verification-manifest.json` before the README or other docs may treat it as verified.

Do not collapse segmented critical-path evidence into a broader claim. In particular, the current low-bit evidence does not verify:

- full Bonsai ONNX pipeline execution;
- real transformer block ONNX execution;
- prompt-to-image generation;
- single monolithic multi-block ONNX, where the verified path is segmented.

### 8. Experiment Records

Experiment records must be kept even if generated images are later deleted.

Keep:

- experiment ID,
- date,
- runner type,
- model ID,
- backend,
- prompt recipe ID,
- width and height,
- step count,
- timeout,
- success or failure,
- failure reason,
- runtime seconds,
- memory if available,
- output artifact reference if available.

Do not permanently keep by default:

- large generated image batches,
- model weights,
- caches,
- raw logs that are already summarized.

## Suggested Repository Structure

```text
assetpack-template/
  README.md
  assetpack.yml

  .github/
    ISSUE_TEMPLATE/
      generate.yml
    workflows/
      validate-template.yml
      issue-to-recipe.yml
      smoke-bench.yml
      report.yml

  templates/
    default.prompt.yml

  constraints/
    required_terms.json
    banned_terms.json
    dedupe.json
    schema.recipe.json

  corpus/
    subjects.jsonl
    scenes.jsonl
    modifiers.jsonl

  recipes/
    approved/
      .gitkeep
    generated/
      .gitkeep

  experiments/
    smoke-bench.yml

  reports/
    index.jsonl
    latest.json
    failures.jsonl

  scripts/
    validate_config.py
    parse_issue.py
    build_recipe.py
    dedupe.py
    run_smoke.py
    make_report.py
```

## Minimum Derived Repository Setup

After creating a new repository from this template, edit `assetpack.yml`:

```yaml
assetpack:
  id: coloring-animals
  title: "Coloring Animals Assetpack"
  version: 0.1.0

theme:
  locked: true
  category: coloring_lineart
  description: "Flood-fill friendly animal line art"

generation:
  image_generation: true
  mode: smoke_test
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
    - "closed regions"
    - "no text"

models:
  image_generation:
    mode: fixed_default_with_configurable_alternate
    default_model_id: sdxl-turbo-quality
    alternate_model_id: ssd-1b-lcm-lora-quality
    allowed_model_ids:
      - sdxl-turbo-quality
      - ssd-1b-lcm-lora-quality
```

## Non-Goals

This template does not aim to:

- host a general image-generation service,
- accept arbitrary free-form prompts by default,
- store model weights in Git,
- store large generated image collections in Git,
- guarantee that every model runs on every GitHub runner,
- automatically publish generated images without review.

## Current Status

Early template design with an evidence-recorded image-model selection for the current fixed-theme asset-generation concept.

Next steps:

1. Add Issue Form for structured generation requests.
2. Add validation workflow.
3. Add recipe builder.
4. Add deduplication logic.
5. Add smoke benchmark workflow.
6. Wire generation workflows to `models.image_generation.default_model_id` and `ASSETPACK_IMAGE_MODEL_ID`.
7. Keep generated images as temporary artifacts unless an explicit review/publish step is added.
