# Repo-persisted CI reports

This directory is reserved for CI reports that are committed back into the repository so downstream verification does not depend on manually sharing GitHub Actions artifact URLs.

## Current reports

```text
docs/ci/bonsai-layout-boundary-latest.json
```

This file is produced by the `bonsai-onnx-smoke` workflow when `target=layout-boundary` runs. It is copied from:

```text
reports/bonsai-layout-boundary/report.json
```

The workflow path filters intentionally do not include `docs/ci/**`, so committing report snapshots does not retrigger the workflow.

## Promotion rule

A repo-persisted report can be promoted to `docs/bonsai-lowbit-verification-manifest.json` only when the report itself says:

```json
{
  "status": "passed",
  "ci_conclusion": "success",
  "claim_promotable_to_manifest": true
}
```

Reports with either of the following are evidence records, but not verified manifest entries:

```json
{
  "status": "external_rate_limited",
  "claim_promotable_to_manifest": false
}
```

```json
{
  "status": "failed",
  "claim_promotable_to_manifest": false
}
```

## Non-claims

Persisting a CI report into this directory does not by itself verify full Bonsai ONNX pipeline execution, real transformer ONNX execution, text encoder execution, scheduler execution, VAE execution, prompt-to-image generation, or single monolithic multi-block ONNX execution. Those require separate report keys and successful evidence-gated promotion.
