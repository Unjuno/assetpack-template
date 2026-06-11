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

This stage is the first stage in this repository that verifies reusable ONNX segment artifacts rather than only a reproducible export probe.

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

## Ten-by-two verification notes

Artifact `bonsai-lowbit-ten-by-two-chain-report-json` contains both export and validation reports, but not persisted ONNX files:

```text
bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json
bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json
```

That stage verifies that the export can be reproduced and validated in one job. It does not prove that reusable ONNX artifacts were stored for downstream use.

The split persistent stage supersedes that limitation by validating downloaded ONNX segment artifacts without low-bit source reload.

The strict all-output check remains false in the reproducible export probe because some diagnostic tensors use the relaxed diagnostic threshold. This does not change the critical-path pass condition recorded in the manifest.

## Pending claims

None for the current low-bit critical-path scope through split persistent ten-by-two ONNX segment artifacts.

## Execution policy

Do not rerun already verified stages unless their implementation or evidence boundary changes.

Forbidden claims remain forbidden unless a future manifest entry verifies them:

- full Bonsai ONNX pipeline;
- real transformer block ONNX verification;
- prompt-to-image generation verification;
- single monolithic multi-block ONNX when the verified path is segmented.
