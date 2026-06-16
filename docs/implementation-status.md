# Implementation status

This document summarizes the current implementation state of `assetpack-template`.

## Status summary

The repository is past cleanup and is now in public-readiness validation.

Implemented:

- Issue-driven image generation workflow.
- Structured Issue request parsing and validation.
- Label-gated generation using the configured request label.
- Policy validation before generation.
- Default and alternate image model selection.
- Committed generated asset records under `assets/generated/`.
- Issue success and failure comments.
- Duplicate recipe detection against existing generated records.
- Lightweight tests for validation, duplicate handling, committed asset preparation, comment generation, and generated asset index behavior.
- External user manual.
- Maintainer release checklist.
- Legacy experiment, benchmark, ONNX, Bonsai, and lowbit surfaces tombstoned or disabled.
- Historical experiment pull requests have been closed or otherwise removed from the active release path.

Not yet fully verified on latest main:

- Lightweight test workflow after the latest documentation commits.
- Fresh smoke Issue after documentation stabilization.
- Duplicate replay after the fresh smoke Issue.
- Intentionally invalid Issue that fails before generation.

## Active workflow

The active generation workflow is:

```text
.github/workflows/assetpack-issue-generate.yml
```

The expected flow is:

```text
GitHub Issue + asset-request label
  -> issue label gate
  -> request validation
  -> policy validation
  -> image generation
  -> committed asset under assets/generated/
  -> Issue result comment
```

## Active configuration

The active repository configuration is:

```text
assetpack.yml
```

Key active settings:

| Area | Value |
| --- | --- |
| Request label | `asset-request` |
| Committed output root | `assets/generated` |
| Generation mode | `issue_persistent_generation` |
| Text input policy | ASCII-only structured fields |
| Free prompt policy | Disabled |
| Required prompt terms | `black outline`, `white background`, `closed regions`, `no text` |

## Active models

| Role | Model id | Backend | Notes |
| --- | --- | --- | --- |
| Default | `sdxl-turbo-quality` | `diffusers` | Uses `stabilityai/sdxl-turbo`. |
| Alternate | `ssd-1b-lcm-lora-quality` | `diffusers_lora` | Uses `segmind/SSD-1B` with `latent-consistency/lcm-lora-ssd-1b`. |

Normal users should select `default` unless a maintainer asks them to compare or override model behavior.

## Generated asset record

A successful generation should commit:

```text
assets/generated/issue-<issue-number>/<recipe_id>/
```

Expected files:

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

## Cleanup state

The old experimental surface has been reduced.

Confirmed removed or no longer active in current main:

- `image-model-ci-benchmark-v3`
- `image-model-onnx-feasibility-v1`
- `run_image_model_ci_benchmark`
- `IMAGE_MODEL_CANDIDATE_TIMEOUT_SECONDS`
- `lowbit`
- `onnx`

Bonsai-related files may still appear in stale code-search results. Current main has representative Bonsai scripts tombstoned and representative Bonsai workflows replaced with inert `workflow_dispatch` workflows.

## Branch and PR state

For external users, `main` is the only supported release candidate. Branches and pull requests are not supported unless their changes are present on `main`.

Historical experiment pull requests for Bonsai, lowbit, ONNX, adapter, and projection work are not part of the active release path.

## Public-readiness gate

Do not treat the repository as fully public-ready until these checks pass on latest main:

1. Run the lightweight test workflow.
2. Submit one fresh valid smoke Issue.
3. Re-submit or re-trigger the same request and verify duplicate handling.
4. Submit one intentionally invalid request and verify generation is not attempted.

## Current judgment

Implementation completeness: high.

Public-readiness completeness: not final.

The remaining work is validation and release hardening, not core feature implementation.
