# Bonsai low-bit local artifact audit

This audit was produced from artifact ZIP files already available in the local execution environment. No new GitHub Actions workflow was triggered.

## Inventory

Local ZIP files inspected: 16.

Latest JSON-only artifact inspected:

```text
bonsai-lowbit-smoke-report-json-7560534190.zip
sha256: 99d88924b39e0253667c086875253f4dadcdab04c58e2578b8b068820f244c0f
report count: 32
```

This artifact contains verification reports through the sixteen-block eight-by-two stage. It does not contain the twenty-block ten-by-two validation report or the pair-segment aggregate validation report.

## Verified locally

| Scope | Report path | Local status |
|---|---|---|
| attention `to_out` path | `bonsai-lowbit-attention-to-out-validation/report.json` | `ok: true` |
| single block modulated attention `to_out` residual | `bonsai-lowbit-single-block-modulated-validation/report.json` | `ok: true` |
| two single blocks modulated residual stack | `bonsai-lowbit-two-single-blocks-modulated-validation/report.json` | `ok: true`, `sequence_block_count: 2` |
| four single blocks via two-by-two segmented ONNX | `bonsai-lowbit-two-by-two-single-blocks-modulated-validation/report.json` | `ok: true`, `sequence_block_count: 4`, `onnx_segment_count: 2`, `critical_outputs_allclose: true`, `critical_max_abs_error: 3.814697265625e-06` |
| eight single blocks via four-by-two segmented ONNX | `bonsai-lowbit-four-by-two-single-blocks-modulated-validation/report.json` | `ok: true`, `sequence_block_count: 8`, `onnx_segment_count: 4`, `critical_outputs_allclose: true`, `critical_max_abs_error: 1.049041748046875e-05` |
| sixteen single blocks via eight-by-two segmented ONNX | `bonsai-lowbit-eight-by-two-single-blocks-modulated-validation/report.json` | `ok: true`, `sequence_block_count: 16`, `onnx_segment_count: 8`, `critical_outputs_allclose: true`, `critical_max_abs_error: 9.059906005859375e-06` |

## Still pending from local evidence

The local artifact set does not include:

```text
reports/bonsai-lowbit-pair-segments-aggregate-validation/report.json
reports/bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

Pending claims:

- all ten pair segments for blocks 0-19;
- twenty single blocks via ten-by-two chained segmented ONNX.

## Execution policy

Do not run CI only to reconfirm the verified stages above.

Use CI only to generate missing artifacts:

1. `target: pair-segments-all` to create `bonsai-lowbit-pair-segments-aggregate-report-json`;
2. `target: ten-by-two-chain` to create `bonsai-lowbit-ten-by-two-chain-report-json`, only after pair segments pass.

If a workflow-dispatch execution tool is available, use it for those missing artifacts. If no such tool is available, keep the workflow manual and inspect artifacts after they are produced.
