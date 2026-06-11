# Bonsai low-bit artifact audit

This audit is now driven by the canonical manifest:

```text
docs/bonsai-lowbit-verification-manifest.json
```

The manifest combines existing local artifact ZIP reports with verified GitHub Actions artifact reports. Claims remain evidence-gated and must not exceed manifest scope.

## Latest verified CI artifacts

| Scope | Run | Artifact | SHA-256 | Status |
|---|---:|---:|---|---|
| all ten pair segments for blocks 0-19 | 27366197393 | 7572369505 | `653af8965e7181b9d7063239e438f21e08de7284337258d1f4ee98e30a4ada8a` | verified |
| twenty blocks via ten-by-two chained segmented ONNX | 27368101404 | 7573239516 | `7e5a75327d81e576512546b3e0c611ae3bbf464c5cdbb1043831fb67d63b99eb` | verified |

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
| twenty single blocks via ten-by-two chained segmented ONNX | CI artifact `bonsai-lowbit-ten-by-two-chain-report-json` | verified |

## Ten-by-two verification notes

Artifact `bonsai-lowbit-ten-by-two-chain-report-json` contains both export and validation reports:

```text
bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json
bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

The validation report records `ok: true`, `sequence_block_count: 20`, `onnx_segment_count: 10`, and `critical_outputs_allclose_rtol_1e_4_atol_1e_5: true`.

The export report records `is_single_monolithic_onnx: false`, `is_real_transformer_block: false`, and `is_full_bonsai_pipeline: false`.

The strict all-output check remains false because some diagnostic tensors use the relaxed diagnostic threshold. This does not change the critical-path pass condition recorded in the manifest.

## Pending claims

None for the current low-bit critical-path scope through ten-by-two twenty-block segmented ONNX.

## Execution policy

Do not rerun already verified stages unless their implementation or evidence boundary changes.

Forbidden claims remain forbidden unless a future manifest entry verifies them:

- full Bonsai ONNX pipeline;
- real transformer block ONNX verification;
- prompt-to-image generation verification;
- single monolithic multi-block ONNX when the verified path is segmented.
