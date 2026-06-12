# Bonsai low-bit artifact audit

This audit is driven by the canonical manifest:

```text
docs/bonsai-lowbit-verification-manifest.json
```

The manifest combines existing local artifact ZIP reports with verified GitHub Actions artifact reports. Claims remain evidence-gated and must not exceed manifest scope.

## Latest verified CI artifacts

| Scope | Run | Artifact | SHA-256 | Status |
|---|---:|---:|---|---|
| all ten pair segments for blocks 0-19 | 27366197393 | 7572369505 | `653af8965e7181b9d7063239e438f21e08de7284337258d1f4ee98e30a4ada8a` | verified |
| twenty blocks via ten-by-two chained export probe, JSON reports only | 27368101404 | 7573239516 | `7e5a75327d81e576512546b3e0c611ae3bbf464c5cdbb1043831fb67d63b99eb` | verified as reproducible export probe |
| twenty blocks via split persistent ONNX segment artifacts | 27370454697 | 7574146614 | `a09ce6303c70b0fd95556803cfd45b3765db1a216ad58f70fdc2e8871d758ab3` | verified as reusable segmented ONNX artifact chain |
| ten-by-two chain-state handoff report | 27372009301 | 7574781425 | `5fdc2278ced66cbaed794943d3975c18dfbeda1cd849328e29a0a7b3cfabd5fb` | verified as segment-to-segment tensor handoff evidence |
| ten-by-two input boundary report | 27399310016 | 7584863386 | `321133159edaa0136412c09efc04e59f61e53da502cc27d737282b565b8665ac` | verified as hidden/temb-only external ONNX input boundary evidence |

## Persistent ONNX artifact set

The split persistent stage uploads these artifacts:

```text
bonsai-lowbit-ten-by-two-persistent-reference
bonsai-lowbit-persistent-onnx-segment-0
bonsai-lowbit-persistent-onnx-segment-2
bonsai-lowbit-persistent-onnx-segment-4
bonsai-lowbit-persistent-onnx-segment-6
bonsai-lowbit-persistent-onnx-segment-8
bonsai-lowbit-persistent-onnx-segment-10
bonsai-lowbit-persistent-onnx-segment-12
bonsai-lowbit-persistent-onnx-segment-14
bonsai-lowbit-persistent-onnx-segment-16
bonsai-lowbit-persistent-onnx-segment-18
bonsai-lowbit-ten-by-two-split-persistent-onnx-validation-report-json
bonsai-lowbit-ten-by-two-chain-handoff-report-json
bonsai-lowbit-ten-by-two-input-boundary-report-json
```

The validation report records:

```json
{
  "ok": true,
  "artifact_kind": "split_persistent_onnx_segment_bundle_validation",
  "sequence_block_count": 20,
  "onnx_segment_count": 10,
  "persistent_onnx_artifacts": true,
  "reusable_onnx_chain_artifact": true,
  "validated_without_lowbit_source_reload": true,
  "validated_from_persisted_onnx_files": true,
  "critical_outputs_allclose_rtol_1e_4_atol_1e_5": true
}
```

The handoff report records:

```json
{
  "ok": true,
  "artifact_kind": "chain_state_handoff_report",
  "sequence_block_count": 20,
  "onnx_segment_count": 10,
  "handoff_count": 10,
  "persistent_onnx_artifacts": true,
  "reusable_onnx_chain_artifact": true,
  "validated_without_lowbit_source_reload": true,
  "validated_from_persisted_onnx_files": true
}
```

The input boundary report records:

```json
{
  "ok": true,
  "artifact_kind": "input_boundary_report",
  "external_onnx_inputs": ["hidden", "temb"],
  "prompt_tokens_present": false,
  "text_encoder_present": false,
  "scheduler_present": false,
  "vae_present": false,
  "image_latents_present": false,
  "real_bonsai_pipeline_inputs_present": false
}
```

This stage verifies reusable ONNX segment artifacts, their segment-to-segment tensor handoffs, and the current hidden/temb-only external ONNX input boundary. It remains narrower than a full Bonsai pipeline.

## Previously verified local artifact set

Latest JSON-only local artifact inspected:

```text
bonsai-lowbit-smoke-report-json-7560534190.zip
sha256: 99d88924b39e0253667c086875253f4dadcdab04c58e2578b8b068820f244c0f
report count: 32
```

That local artifact contains verification reports through the sixteen-block eight-by-two stage.

## Verified scopes

| Scope | Evidence | Status |
|---|---|---|
| attention `to_out` path | local artifact report | verified |
| single block modulated attention `to_out` residual | local artifact report | verified |
| two single blocks modulated residual stack | local artifact report | verified |
| four single blocks via two-by-two segmented ONNX | local artifact report | verified |
| eight single blocks via four-by-two segmented ONNX | local artifact report | verified |
| sixteen single blocks via eight-by-two segmented ONNX | local artifact report | verified |
| all ten pair segments for blocks 0-19 | CI artifact `bonsai-lowbit-pair-segments-aggregate-report-json` | verified |
| twenty single blocks via ten-by-two chained segmented export probe | CI artifact `bonsai-lowbit-ten-by-two-chain-report-json` | verified as report-only reproducible export probe |
| twenty single blocks via split persistent segmented ONNX artifacts | CI artifact `bonsai-lowbit-ten-by-two-split-persistent-onnx-validation-report-json` plus reference and ten segment artifacts | verified as reusable artifact chain |
| ten-by-two chain-state handoffs | CI artifact `bonsai-lowbit-ten-by-two-chain-handoff-report-json` | verified as reusable segment handoff evidence |
| ten-by-two input boundary | CI artifact `bonsai-lowbit-ten-by-two-input-boundary-report-json` | verified as hidden/temb-only external ONNX input boundary evidence |

## Ten-by-two verification notes

Artifact `bonsai-lowbit-ten-by-two-chain-report-json` contains both export and validation reports, but not persisted ONNX files:

```text
bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json
bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

That stage verifies that the export can be reproduced and validated in one job. It does not prove that reusable ONNX artifacts were stored for downstream use.

The split persistent stage supersedes that limitation by validating downloaded ONNX segment artifacts without low-bit source reload.

The chain-state handoff report documents how each segment's `block1_output` is handed off as the following segment's hidden input.

The input boundary report documents that external ONNX inputs are limited to `hidden` and `temb`, and that prompt tokens, text encoder, scheduler, VAE, image latents, and full Bonsai pipeline inputs are not present in this verified chain.

The strict all-output check remains false in the reproducible export probe because some diagnostic tensors use the relaxed diagnostic threshold. This does not change the critical-path pass condition recorded in the manifest.

## Pending claims

None for the current low-bit critical-path scope through ten-by-two input-boundary documentation.

## Execution policy

Do not rerun already verified stages unless their implementation or evidence boundary changes.

Forbidden claims remain forbidden unless a future manifest entry verifies them:

- full Bonsai ONNX pipeline;
- real transformer block ONNX verification;
- prompt-to-image generation verification;
- single monolithic multi-block ONNX when the verified path is segmented.
