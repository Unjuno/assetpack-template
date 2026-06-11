# Bonsai low-bit claim matrix

Source: `docs/bonsai-lowbit-verification-manifest.json` is canonical. This matrix is a human-readable summary of manifest-gated claims.

## Summary

| Stage | Status | Blocks | ONNX segments | Critical allclose | Critical max abs error | Monolithic ONNX | Full Bonsai pipeline |
|---|---|---:|---:|---|---:|---|---|
| Attention to_out path | `verified` |  |  |  |  |  |  |
| Single block modulated residual | `verified` |  |  |  |  |  |  |
| Two single blocks | `verified` | 2 |  |  |  |  |  |
| Four blocks via 2x2 segments | `verified` | 4 | 2 | true | 3.814697265625e-06 | false |  |
| Eight blocks via 4x2 segments | `verified` | 8 | 4 | true | 1.049041748046875e-05 | false |  |
| Sixteen blocks via 8x2 segments | `verified` | 16 | 8 | true | 9.059906005859375e-06 | false |  |
| All ten pair segments 0-19 | `verified` | 20 | 10 | true | 4.76837158203125e-06 | false | false |
| Twenty blocks via 10x2 chained segments | `verified` | 20 | 10 | true | 8.58306884765625e-06 | false | false |

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

### All ten pair segments 0-19

- Run: `27366197393`
- Artifact: `bonsai-lowbit-pair-segments-aggregate-report-json` / `7572369505`
- SHA-256: `653af8965e7181b9d7063239e438f21e08de7284337258d1f4ee98e30a4ada8a`
- Report: `bonsai-lowbit-pair-segments-aggregate-validation/report.json`
- Graph kind: `pair_segments_aggregate_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `all_pair_segments_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_paths_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline`

### Twenty blocks via 10x2 chained segments

- Run: `27368101404`
- Head SHA: `8de1a0ed4504017d87024a6401c91e4d91732ad7`
- Artifact: `bonsai-lowbit-ten-by-two-chain-report-json` / `7573239516`
- SHA-256: `7e5a75327d81e576512546b3e0c611ae3bbf464c5cdbb1043831fb67d63b99eb`
- Validation report: `bonsai-lowbit-ten-by-two-single-blocks-modulated-validation/report.json`
- Export report: `bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx/report.json`
- Graph kind: `ten_by_two_single_blocks_modulated_attention_to_out_residual_stack`
- Claim: `ten_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline`

## Pending artifacts

None. The current low-bit critical-path scope through ten-by-two twenty-block segmented ONNX is verified in the manifest.

## Next execution boundary

Decide the next low-bit boundary after the verified ten-by-two twenty-block critical path. Forbidden claims still remain forbidden unless a new manifest entry verifies them.
