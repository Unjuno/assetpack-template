# Bonsai low-bit ONNX completion plan

This plan is intentionally stepwise. Complete one step, verify the artifact, then move to the next step. Do not rerun already verified stages unless their implementation changes.

## Current baseline

Already verified and recorded in `docs/bonsai-lowbit-verification-ledger.md`:

- attention `to_out` path
- single block modulated attention `to_out` residual
- two single blocks
- four single blocks via two-by-two segmented ONNX
- eight single blocks via four-by-two segmented ONNX
- sixteen single blocks via eight-by-two segmented ONNX

Pending:

- all ten pair segments for single blocks 0-19
- twenty single blocks via ten-by-two chained segmented ONNX

## Step 1 — Execution discipline

Status: done.

Artifacts:

- `docs/bonsai-lowbit-verification-ledger.md`
- `.github/workflows/bonsai-lowbit-smoke.yml`

Completion condition:

- Workflow is manual only via `workflow_dispatch`.
- Workflow accepts a `target` input.
- Verified historical stages are not part of the default active script list.

## Step 2 — Pair-segment verification for blocks 0-19

Status: ready.

Run:

```text
Actions → bonsai-lowbit-smoke → Run workflow
branch: main
target: pair-segments-all
segment_start: 0
```

Expected artifact:

```text
bonsai-lowbit-pair-segments-aggregate-report-json
```

Required report:

```text
reports/bonsai-lowbit-pair-segments-aggregate-validation/report.json
```

Completion condition:

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

If a segment fails, rerun only that segment:

```text
target: pair-segment
segment_start: <failed even index>
```

## Step 3 — Twenty-block chained segmented path

Status: blocked until Step 2 passes.

Run:

```text
Actions → bonsai-lowbit-smoke → Run workflow
branch: main
target: ten-by-two-chain
segment_start: 0
```

Expected artifact:

```text
bonsai-lowbit-ten-by-two-chain-report-json
```

Required reports:

```text
reports/bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json
reports/bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

Completion condition:

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

- full Bonsai ONNX pipeline
- real transformer block ONNX verification
- prompt-to-image generation verification
- single monolithic 20-block ONNX

## Step 4 — Decide next expansion boundary

Status: blocked until Step 3 passes.

Options:

1. add a lightweight chain-state handoff report between pair segments;
2. add image/semantic input boundary probes;
3. add double-block side of the real architecture;
4. keep ONNX scope capped at single-stream block chain and document the boundary.

Do not start Step 4 until Step 3 is verified from a JSON artifact.

