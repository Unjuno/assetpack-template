# Experiment configurations

This directory contains historical model-selection experiment configs.

The main template contract is:

```text
assetpack.yml
```

## Current result

The experiment history selected this pair:

```text
default:   sdxl-turbo-quality
alternate: ssd-1b-lcm-lora-quality
```

The durable decision is recorded in:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
assetpack.yml
```

## Derived repository guidance

A derived repository does not need to rerun these historical experiments.

Use the Issue-driven path unless you are changing model selection:

```text
.github/ISSUE_TEMPLATE/generate.yml
.github/workflows/assetpack-issue-generate.yml
scripts/validate_issue_request.py
scripts/run_issue_asset_generation.py
```

Generated images should remain workflow artifacts unless a derived repository adds a review and publish step.
