# Bonsai low-bit ONNX completion plan

This plan is intentionally stepwise. Complete one step, verify the artifact, then move to the next step. Do not rerun already verified stages unless their implementation or evidence boundary changes.

Canonical verification source:

```text
docs/bonsai-lowbit-verification-manifest.json
```

## Current baseline

Verified and recorded in the manifest:

- attention `to_out` path;
- single block modulated attention `to_out` residual;
- two single blocks;
- four single blocks via two-by-two segmented ONNX;
- eight single blocks via four-by-two segmented ONNX;
- sixteen single blocks via eight-by-two segmented ONNX;
- all ten pair segments for blocks 0-19;
- twenty single blocks via ten-by-two chained segmented export probe;
- twenty single blocks via split persistent ten-by-two ONNX segment artifacts, validated as a reusable segmented ONNX chain.

Pending for the current low-bit critical-path scope: none.

## Step 1 — Execution discipline

Status: done.

Artifacts:

- `docs/bonsai-lowbit-verification-manifest.json`
- `.github/workflows/bonsai-lowbit-smoke.yml`

Completion condition:

- Workflow accepts a `target` input for manual execution.
- Verified historical stages are not part of the default active script list.
- Claims are promoted only after JSON artifact verification and manifest update.

## Step 2 — Pair-segment verification for blocks 0-19

Status: verified.

Run:

```text
27366197393
```

Artifact:

```text
bonsai-lowbit-pair-segments-aggregate-report-json / 7572369505
sha256: 653af8965e7181b9d7063239e438f21e08de7284337258d1f4ee98e30a4ada8a
```

Required report:

```text
bonsai-lowbit-pair-segments-aggregate-validation/report.json
```

Verified completion condition:

```json
{
  "ok": true,
  "sequence_block_count": 20,
  "onnx_segment_count": 10,
  "critical_outputs_allclose_rtol_1e_4_atol_1e_5": true,
  "is_single_monolithic_onnx": false,
  "is_real_transformer_block": false,
  "is_full_bonsai_pipeline": false
}
```

## Step 3 — Twenty-block chained segmented export probe

Status: verified as reproducible export probe.

Run:

```text
27368101404
```

Artifact:

```text
bonsai-lowbit-ten-by-two-chain-report-json / 7573239516
sha256: 7e5a75327d81e576512546b3e0c611ae3bbf464c5cdbb1043831fb67d63b99eb
head_sha: 8de1a0ed4504017d87024a6401c91e4d91732ad7
```

Required reports:

```text
bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json
bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

Verified completion condition:

```json
{
  "ok": true,
  "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
  "sequence_block_count": 20,
  "onnx_segment_count": 10,
  "persistent_onnx_artifacts": false,
  "reusable_onnx_chain_artifact": false,
  "critical_outputs_allclose_rtol_1e_4_atol_1e_5": true,
  "is_single_monolithic_onnx": false,
  "is_real_transformer_block": false,
  "is_full_bonsai_pipeline": false
}
```

Allowed claim after completion:

```text
ten-by-two single blocks modulated attention to_out residual stack ONNX Runtime CPU critical path verified as a reproducible export probe, not as a persistent ONNX artifact chain
```

## Step 4 — Split persistent ONNX segment artifacts

Status: verified.

Run:

```text
27370454697
```

Validation artifact:

```text
bonsai-lowbit-ten-by-two-split-persistent-onnx-validation-report-json / 7574146614
sha256: a09ce6303c70b0fd95556803cfd45b3765db1a216ad58f70fdc2e8871d758ab3
head_sha: 012ab71d0fe2dcb9737e2ad0192c05be92209814
```

Supporting artifacts:

```text
bonsai-lowbit-ten-by-two-persistent-reference / 7574094960
bonsai-lowbit-persistent-onnx-segment-0 / 7574104834
bonsai-lowbit-persistent-onnx-segment-2 / 7574104157
bonsai-lowbit-persistent-onnx-segment-4 / 7574104773
bonsai-lowbit-persistent-onnx-segment-6 / 7574103247
bonsai-lowbit-persistent-onnx-segment-8 / 7574106732
bonsai-lowbit-persistent-onnx-segment-10 / 7574104608
bonsai-lowbit-persistent-onnx-segment-12 / 7574105962
bonsai-lowbit-persistent-onnx-segment-14 / 7574106402
bonsai-lowbit-persistent-onnx-segment-16 / 7574105292
bonsai-lowbit-persistent-onnx-segment-18 / 7574105357
```

Verified completion condition:

```json
{
  "ok": true,
  "artifact_kind": "split_persistent_onnx_segment_bundle_validation",
  "graph_kind": "ten_by_two_single_blocks_modulated_attention_to_out_residual_stack",
  "sequence_block_count": 20,
  "onnx_segment_count": 10,
  "persistent_onnx_artifacts": true,
  "reusable_onnx_chain_artifact": true,
  "validated_without_lowbit_source_reload": true,
  "validated_from_persisted_onnx_files": true,
  "critical_outputs_allclose_rtol_1e_4_atol_1e_5": true,
  "is_single_monolithic_onnx": false,
  "is_real_transformer_block": false,
  "is_full_bonsai_pipeline": false
}
```

Allowed claim after completion:

```text
twenty single blocks via ten split persistent two-block ONNX segment artifacts are validated as a reusable ONNX Runtime CPU chain without low-bit source reload
```

Forbidden claims remain:

- full Bonsai ONNX pipeline;
- real transformer block ONNX verification;
- prompt-to-image generation verification;
- single monolithic 20-block ONNX.

## Step 5 — Decide next expansion boundary

Status: ready for decision.

Options:

1. add a lightweight chain-state handoff report between persistent pair segments;
2. add image/semantic input boundary probes;
3. add double-block side of the real architecture;
4. keep ONNX scope capped at single-stream block chain and document the boundary.

Do not start Step 5 unless the next boundary is explicitly defined and its claim language is added to the manifest evidence policy.
