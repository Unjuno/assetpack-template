# Bonsai low-bit ONNX completion plan

This plan is intentionally stepwise. Complete one step, verify the artifact, then move to the next step. Do not rerun already verified stages unless their implementation changes.

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
- twenty single blocks via ten-by-two chained segmented ONNX.

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

## Step 3 — Twenty-block chained segmented path

Status: verified.

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
  "critical_outputs_allclose_rtol_1e_4_atol_1e_5": true,
  "is_single_monolithic_onnx": false,
  "is_real_transformer_block": false,
  "is_full_bonsai_pipeline": false
}
```

Allowed claim after completion:

```text
ten-by-two single blocks modulated attention to_out residual stack ONNX Runtime CPU critical path verified
```

Forbidden claims remain:

- full Bonsai ONNX pipeline;
- real transformer block ONNX verification;
- prompt-to-image generation verification;
- single monolithic 20-block ONNX.

## Step 4 — Decide next expansion boundary

Status: ready for decision.

Options:

1. add a lightweight chain-state handoff report between pair segments;
2. add image/semantic input boundary probes;
3. add double-block side of the real architecture;
4. keep ONNX scope capped at single-stream block chain and document the boundary.

Do not start Step 4 unless the next boundary is explicitly defined and its claim language is added to the manifest evidence policy.
