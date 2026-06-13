# CI evidence records

This directory stores compact, durable CI evidence for the assetpack template.

It is not a generated-image archive. Generated PNGs should normally remain GitHub Actions artifacts.

## Current image-model selection

Current selected default:

```text
sdxl-turbo-quality
```

Configurable alternate:

```text
ssd-1b-lcm-lora-quality
```

Selection records:

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

Final runoff evidence:

```text
workflow: image-model-ci-final-runoff-hard-subjects
run_id: 27446158099
expected_images: 60
extracted_pngs: 57
```

This is a repository-specific asset-generation decision, not a universal image-model benchmark.

## Evidence levels

| Level | Meaning |
|---|---|
| CI success | A workflow ran and produced a report or image. |
| Visual acceptance | A human reviewed the output and found it useful for the concept. |
| Model selection | A model is configured as selected or allowed in `assetpack.yml`. |

Do not treat CI success alone as model-quality evidence.

## Repo-persisted report policy

Keep:

- compact JSON reports;
- selected summaries;
- model-selection records;
- human-readable evidence notes.

Do not keep by default:

- generated image batches;
- model weights;
- dependency caches;
- large raw logs that are already summarized.

## Issue-driven generation artifacts

Issue-driven generation writes workflow artifacts named like:

```text
assetpack-issue-<issue-number>-<recipe-id>
```

Those artifacts may contain:

```text
request.json
validation-comment.md
generation-comment.md
issue-generation-config.yml
report.json
images/**/*.png
```

Generated images are artifacts for review. They are not committed to Git by default.

## Current durable records

```text
docs/ci/image-model-final-selection.md
docs/ci/image-model-final-selection.json
```

Derived repositories may add their own compact CI records here, but should avoid storing large generated outputs.
