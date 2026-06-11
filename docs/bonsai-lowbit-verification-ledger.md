# Bonsai low-bit ONNX verification ledger

This ledger records what has already been verified and what should not be rerun by default.

## Verified claims

| Scope | Evidence | Allowed claim | Not claimed |
|---|---|---|---|
| attention to_out path | GitHub Actions artifact `7530461105`, run `27263409257`, head `8f495ee303017d2164b5f8e44733c7334ebed22c` | `attention to_out path ONNX Runtime CPU verified` | full Bonsai pipeline, real transformer block |
| single block modulated attention to_out residual | GitHub Actions artifact `7531397648`, run `27265707905`, head `c95fab6000a211d1c78c7569b96955fd7d782f2e` | `single block modulated attention to_out residual ONNX Runtime CPU verified` | full Bonsai pipeline, real transformer block |
| two single blocks modulated attention to_out residual stack | GitHub Actions JSON artifact `7555937768`, run `27326817157`, head `ebd5c2c951506eac612e16e0b473607164db020d` | `two single blocks modulated attention to_out residual stack ONNX Runtime CPU verified` | full Bonsai pipeline, real transformer block |
| four single blocks via two-by-two segmented ONNX | GitHub Actions JSON artifact `7557717825`, run `27331249957`, head `ede59a0574a9f7cfb49b7f4a144913e57174c8ab` | `two-by-two single blocks modulated attention to_out residual stack ONNX Runtime CPU critical path verified` | single monolithic 4-block ONNX, full Bonsai pipeline, real transformer block |
| eight single blocks via four-by-two segmented ONNX | GitHub Actions JSON artifact `7558446912`, run `27332812571`, head `b468530610c361a37475f14ed632567ff3ff57ab` | `four-by-two single blocks modulated attention to_out residual stack ONNX Runtime CPU critical path verified` | single monolithic 8-block ONNX, full Bonsai pipeline, real transformer block |
| sixteen single blocks via eight-by-two segmented ONNX | GitHub Actions JSON artifact `7559753013`, run `27335861344`, head `b143c693da369b16cf3b1aa11c29360e275ed397` | `eight-by-two single blocks modulated attention to_out residual stack ONNX Runtime CPU critical path verified` | single monolithic 16-block ONNX, full Bonsai pipeline, real transformer block |

## Pending claims

| Scope | Required evidence before claim |
|---|---|
| twenty single blocks via ten-by-two segmented ONNX | `ten-by-two-chain` workflow target succeeds and uploads `bonsai-lowbit-ten-by-two-chain-report-json` containing `reports/bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json` with `ok: true` |
| all ten pair segments for blocks 0-19 | `pair-segments-all` workflow target succeeds and uploads `bonsai-lowbit-pair-segments-aggregate-report-json` with `ok: true` |

## Workflow use policy

Default rule: do not rerun verified historical stages unless the implementation they depend on changes.

Use `bonsai-lowbit-smoke` with `workflow_dispatch` inputs:

| target | When to use |
|---|---|
| `pair-segment` | Recheck one specific pair segment only. Set `segment_start` to one of `0,2,4,6,8,10,12,14,16,18`. |
| `pair-segments-all` | Recheck all pair segments 0-19 and create a single aggregate report. Use this before a full 20-block chain run or after exporter/core changes. |
| `ten-by-two-chain` | Run only the 20-block chained path verification. Use after all pair segments are verified. |

Do not add already verified stages to the default workflow script list. Historical verification scripts may remain in the repository for reproducibility, but they are not part of the active workflow by default.

## Claim discipline

Allowed language must include the exact tested scope. Do not collapse any segmented-chain result into a full Bonsai ONNX pipeline claim.

Never claim:

- full Bonsai ONNX pipeline
- real transformer block ONNX verification
- prompt-to-image generation verification
- single monolithic multi-block ONNX when the verified result is segmented

