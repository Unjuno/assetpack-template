# Repo-persisted CI reports

This directory is reserved for CI reports that are committed back into the repository so downstream verification does not depend on manually sharing GitHub Actions artifact URLs.

## Current reports

```text
docs/ci/bonsai-layout-boundary-latest.json
docs/ci/bonsai-combined-component-probe-latest.json
docs/ci/bonsai-transformer-load-probe-latest.json
```

`docs/ci/bonsai-layout-boundary-latest.json` is produced by the `bonsai-onnx-smoke` workflow when `target=layout-boundary` runs. It is copied from:

```text
reports/bonsai-layout-boundary/report.json
```

`docs/ci/bonsai-combined-component-probe-latest.json` is produced by the `bonsai-combined-component-probe` workflow. It is copied from:

```text
reports/bonsai-combined-component-probe/report.json
```

The current canonical transformer weight-load evidence is also in:

```text
docs/ci/bonsai-combined-component-probe-latest.json
```

with:

```text
run_transformer_load: true
transformer_weight_load: passed
```

`docs/ci/bonsai-transformer-load-probe-latest.json` is reserved for the dedicated transformer load workflow when that workflow is used. It is copied from:

```text
reports/bonsai-transformer-load-probe/report.json
```

The current canonical transformer minimal ONNX export evidence is in:

```text
docs/ci/bonsai-transformer-load-probe-latest.json
```

with:

```text
onnx_export_attempted: true
onnx_export.status: passed
onnx_export.allowed_claim: bonsai_transformer_minimal_onnx_export_verified_not_full_pipeline
onnx_export.external_data_enabled: true
```

This verifies only the minimal standalone transformer ONNX export boundary. It does not verify ONNX Runtime execution, full pipeline composition, prompt-to-image generation, or a single monolithic multi-block ONNX graph.

The workflow path filters intentionally do not include `docs/ci/**`, so committing report snapshots does not retrigger the workflows.

## Promotion rule

A repo-persisted report can be promoted to `docs/bonsai-combined-component-verification-manifest.json` only when the specific stage or report claim itself says:

```json
{
  "status": "passed",
  "ci_conclusion": "success",
  "claim_promotable_to_manifest": true
}
```

For combined reports, individual stages may be promotable even when later full-pipeline stages are blocked or not implemented. Promote only the explicit stage-level `allowed_claim` values with `claim_promotable_to_manifest=true`.

Reports or stages with `external_rate_limited`, `failed`, or `blocked` are evidence records, but not verified manifest entries.

## Non-claims

Persisting a CI report into this directory does not by itself verify full Bonsai ONNX pipeline execution, real transformer ONNX execution, prompt-to-image generation, or single monolithic multi-block ONNX execution. Those require explicit successful report keys and successful evidence-gated promotion.
