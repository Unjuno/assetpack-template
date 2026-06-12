# Repo-persisted CI reports

This directory is reserved for CI reports that are committed back into the repository so downstream verification does not depend on manually sharing GitHub Actions artifact URLs.

## Current reports

```text
docs/ci/bonsai-layout-boundary-latest.json
docs/ci/bonsai-combined-component-probe-latest.json
docs/ci/bonsai-transformer-load-probe-latest.json
docs/ci/ort-cpu-kernel-probe-latest.json
docs/ci/image-model-ci-benchmark-smoke-latest.json
docs/ci/image-model-ci-benchmark-lcm-latest.json
docs/ci/image-model-ci-benchmark-turbo-latest.json
docs/ci/image-model-ci-benchmark-load-only-latest.json
docs/ci/image-model-ci-benchmark-runtime-latest.json
docs/ci/image-model-ci-candidates/<candidate-id>-latest.json
docs/ci/image-model-onnx-feasibility/<candidate-id>-latest.json
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

`docs/ci/ort-cpu-kernel-probe-latest.json` is produced by the `ort-cpu-kernel-probe` workflow. It is copied from:

```text
reports/ort-cpu-kernel-probe/report.json
```

That report is synthetic ONNX Runtime CPUExecutionProvider evidence for `Cos` type handling. The current relevant result is:

```text
Cos(FLOAT16) load: passed
Cos(FLOAT32) load: passed
Cos(DOUBLE) load: failed with Cos(7) NotImplemented
Cast DOUBLE -> FLOAT -> Cos -> Cast back load: passed
Cos(INT64) load: failed as invalid graph
Cast INT64 -> FLOAT -> Cos -> Cast back load: passed
```

This supports the diagnostic interpretation that a Bonsai ONNX Runtime error such as `Cos(7)` can be caused by DOUBLE trig inputs on CPUExecutionProvider. It is not Bonsai model evidence by itself, not transformer graph evidence by itself, not execution evidence, and not a full pipeline claim.

The `image-model-ci-benchmark` workflow writes one repo-persisted report per benchmark batch:

```text
reports/image-model-ci-benchmark/smoke/report.json     -> docs/ci/image-model-ci-benchmark-smoke-latest.json
reports/image-model-ci-benchmark/lcm/report.json       -> docs/ci/image-model-ci-benchmark-lcm-latest.json
reports/image-model-ci-benchmark/turbo/report.json     -> docs/ci/image-model-ci-benchmark-turbo-latest.json
reports/image-model-ci-benchmark/load_only/report.json -> docs/ci/image-model-ci-benchmark-load-only-latest.json
reports/image-model-ci-benchmark/runtime/report.json   -> docs/ci/image-model-ci-benchmark-runtime-latest.json
```

The `image-model-ci-candidate-dynamic` workflow reads `experiments/image-model-ci-benchmark.yml` at runtime and writes one repo-persisted report per selected candidate:

```text
reports/image-model-ci-candidate/<candidate-id>/report.json -> docs/ci/image-model-ci-candidates/<candidate-id>-latest.json
```

It also uploads a per-candidate artifact named:

```text
image-model-ci-candidate-<candidate-id>
```

The artifact contains the candidate report and any generated `cat.png` image. PNGs are intentionally not committed to the repository; they are for human inspection from the Actions artifact UI. This dynamic workflow supersedes the removed static `image-model-ci-candidate-matrix` workflow.

The `image-model-onnx-feasibility-dynamic` workflow reads `experiments/image-model-ci-benchmark.yml` at runtime and writes one repo-persisted ONNX feasibility report per selected candidate:

```text
reports/image-model-onnx-feasibility/<candidate-id>/report.json -> docs/ci/image-model-onnx-feasibility/<candidate-id>-latest.json
```

It also uploads a per-candidate artifact named:

```text
image-model-onnx-feasibility-<candidate-id>
```

For Stable-Diffusion-style UNet candidates, the ONNX feasibility workflow attempts a minimal UNet ONNX export, ONNX Runtime CPU load, and ONNX Runtime dummy execution. For SDXL, Flux, Qwen, PixArt, adapter-only repositories, component-only methods, OpenVINO placeholders, and native runtime placeholders, the workflow records the skip or failure boundary instead of pretending a generic ONNX path is valid. These are feasibility measurements only; they do not verify a full prompt-to-image ONNX pipeline. This dynamic workflow supersedes the removed static `image-model-onnx-feasibility-matrix` workflow.

Those reports record CI measurements for candidate image-generation methods using a fixed cat prompt. Candidate-level records include load seconds, generation seconds, total seconds, image SHA-256, image size, disk snapshots, max RSS, method, pipeline class, and failure boundary. These reports are measurement evidence only; they are not model-quality claims, not general benchmark claims, and not prompt-to-image product claims beyond the specific CI run configuration.

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

Reports or stages with `external_rate_limited`, `failed`, `preflight_started`, `preflight_only`, or `blocked` are evidence records, but not verified manifest entries. Benchmark measurement reports are not verification manifest entries unless a separate explicit promotion gate is added.

## Non-claims

Persisting a CI report into this directory does not by itself verify full Bonsai ONNX pipeline execution, real transformer ONNX execution, general model quality, general benchmark superiority, prompt-to-image product readiness, or single monolithic multi-block ONNX execution. Those require explicit successful report keys and successful evidence-gated promotion.
