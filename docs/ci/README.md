# CI evidence records

This directory stores compact, durable CI evidence.

It is not a generated-image archive. Generated PNGs should normally remain GitHub Actions artifacts. This directory should contain summaries, JSON reports, manifests, and human-readable decisions that are small enough to review in Git.

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
| Verification manifest promotion | A specific evidence key is explicitly promoted to a manifest after passing its gate. |

Do not treat CI success alone as model-quality evidence.

## Repo-persisted report policy

Persisted reports are useful when downstream verification should not depend on manually shared GitHub Actions artifact URLs.

Keep:

- compact JSON reports;
- selected summaries;
- model-selection records;
- explicit verification manifests;
- human-readable evidence notes.

Do not keep by default:

- generated image batches;
- model weights;
- dependency caches;
- large raw logs that are already summarized.

## Current report families

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
docs/ci/image-model-final-selection.json
docs/ci/image-model-final-selection.md
```

## Bonsai and ONNX probe records

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

## ONNX Runtime CPU diagnostic records

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

## Image-model CI records

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

The artifact contains the candidate report and any generated image. PNGs are intentionally not committed to the repository; they are for human inspection from the Actions artifact UI.

The `image-model-onnx-feasibility-dynamic` workflow reads `experiments/image-model-ci-benchmark.yml` at runtime and writes one repo-persisted ONNX feasibility report per selected candidate:

```text
reports/image-model-onnx-feasibility/<candidate-id>/report.json -> docs/ci/image-model-onnx-feasibility/<candidate-id>-latest.json
```

It also uploads a per-candidate artifact named:

```text
image-model-onnx-feasibility-<candidate-id>
```

For Stable-Diffusion-style UNet candidates, the ONNX feasibility workflow attempts a minimal UNet ONNX export, ONNX Runtime CPU load, and ONNX Runtime dummy execution. For SDXL, Flux, Qwen, PixArt, adapter-only repositories, component-only methods, OpenVINO placeholders, and native runtime placeholders, the workflow records the skip or failure boundary instead of pretending a generic ONNX path is valid. These are feasibility measurements only; they do not verify a full prompt-to-image ONNX pipeline.

Those reports record CI measurements for candidate image-generation methods using fixed prompts. Candidate-level records include load seconds, generation seconds, total seconds, image SHA-256, image size, disk snapshots, max RSS, method, pipeline class, and failure boundary. These reports are measurement evidence only; they are not model-quality claims, not general benchmark claims, and not product-readiness claims beyond the specific CI run configuration.

The workflow path filters intentionally do not include `docs/ci/**`, so committing report snapshots does not retrigger the workflows.

## Promotion rule

A repo-persisted report can be promoted to a verification manifest only when the specific stage or report claim itself says:

```json
{
  "status": "passed",
  "ci_conclusion": "success",
  "claim_promotable_to_manifest": true
}
```

For combined reports, individual stages may be promotable even when later stages are blocked or not implemented. Promote only the explicit stage-level `allowed_claim` values with `claim_promotable_to_manifest=true`.

Reports or stages with `external_rate_limited`, `failed`, `preflight_started`, `preflight_only`, or `blocked` are evidence records, but not verified manifest entries. Benchmark measurement reports are not verification manifest entries unless a separate explicit promotion gate is added.

## Non-claims

Persisting a CI report into this directory does not by itself verify full Bonsai ONNX pipeline execution, real transformer ONNX execution, general model quality, general benchmark superiority, prompt-to-image product readiness, or single monolithic multi-block ONNX execution. Those require explicit successful report keys and successful evidence-gated promotion.
