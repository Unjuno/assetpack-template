# Bonsai low-bit claim matrix

Source: existing local artifact ZIPs only. No CI was triggered for this matrix.

## Summary

| Stage | Status | Blocks | ONNX segments | Critical allclose | Critical max abs error | Monolithic ONNX | Full Bonsai pipeline |
|---|---|---:|---:|---|---:|---|---|
| Attention to_out path | `verified` |  |  |  |  |  |  |
| Single block modulated residual | `verified` |  |  |  |  |  |  |
| Two single blocks | `verified` | 2 |  |  |  |  |  |
| Four blocks via 2x2 segments | `verified` | 4 | 2 | true | 3.814697265625e-06 | false |  |
| Eight blocks via 4x2 segments | `verified` | 8 | 4 | true | 1.049041748046875e-05 | false |  |
| Sixteen blocks via 8x2 segments | `verified` | 16 | 8 | true | 9.059906005859375e-06 | false |  |
| All ten pair segments 0-19 | `missing_artifact` |  |  |  |  |  |  |
| Twenty blocks via 10x2 chained segments | `missing_artifact` |  |  |  |  |  |  |

## Verified claim strings

### Attention to_out path

- Report: `bonsai-lowbit-attention-to-out-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `qkv_projection_attention_to_out`
- Claim: `attention_to_out_path_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline`

### Single block modulated residual

- Report: `bonsai-lowbit-single-block-modulated-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `single_block_modulated_attention_to_out_residual`
- Claim: `single_block_modulated_attention_to_out_residual_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline`

### Two single blocks

- Report: `bonsai-lowbit-two-single-blocks-modulated-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `two_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline`

### Four blocks via 2x2 segments

- Report: `bonsai-lowbit-two-by-two-single-blocks-modulated-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `two_by_two_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `two_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline`

### Eight blocks via 4x2 segments

- Report: `bonsai-lowbit-four-by-two-single-blocks-modulated-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `four_by_two_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `four_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline`

### Sixteen blocks via 8x2 segments

- Report: `bonsai-lowbit-eight-by-two-single-blocks-modulated-validation/report.json`
- ZIP: `bonsai-lowbit-smoke-report-json-7560534190.zip`
- Graph kind: `eight_by_two_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `eight_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline`

## Pending artifacts

- `pair_segments_aggregate` — all ten pair segments 0-19 — `missing_artifact`
- `ten_by_two_twenty_blocks` — twenty blocks via 10x2 chained segments — `missing_artifact`

## Next execution boundary

Only these missing artifacts require CI/manual workflow execution:

1. `pair_segments_aggregate`
2. `ten_by_two_twenty_blocks` after pair segments pass
