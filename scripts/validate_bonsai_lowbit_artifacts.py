#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('reports')

LOWBIT = ROOT / 'bonsai-lowbit-smoke' / 'report.json'
LAYER = ROOT / 'bonsai-lowbit-layer' / 'report.json'
COMPARE = ROOT / 'bonsai-lowbit-compare' / 'report.json'
COMPARE_LAYERS = ROOT / 'bonsai-lowbit-compare-layers' / 'report.json'
COMPARE_ALL = ROOT / 'bonsai-lowbit-compare-all-layers' / 'report.json'
LAYER_ONNX = ROOT / 'bonsai-lowbit-layer-onnx' / 'report.json'
RUNTIME_LINEAR_ONNX = ROOT / 'bonsai-lowbit-runtime-linear-onnx' / 'report.json'
MULTI_LINEAR_ONNX = ROOT / 'bonsai-lowbit-multi-linear-onnx' / 'report.json'
SAME_BLOCK_PROJECTION_ONNX = ROOT / 'bonsai-lowbit-same-block-projection-onnx' / 'report.json'
QKV_MLP_SPLIT_ONNX = ROOT / 'bonsai-lowbit-qkv-mlp-split-onnx' / 'report.json'
QKV_HEAD_RESHAPE_ONNX = ROOT / 'bonsai-lowbit-qkv-head-reshape-onnx' / 'report.json'
QKV_SEQ_HEAD_LAYOUT_ONNX = ROOT / 'bonsai-lowbit-qkv-seq-head-layout-onnx' / 'report.json'
ATTENTION_MATH_ONNX = ROOT / 'bonsai-lowbit-attention-math-onnx' / 'report.json'


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f'missing report: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise AssertionError(f'invalid json: {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise AssertionError(f'report is not an object: {path}')
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_onnx_file(report: dict[str, Any], label: str) -> None:
    require(int(report.get('onnx_size_bytes', 0)) > 0, f'{label} empty onnx file: {report}')
    onnx_path = Path(str(report.get('onnx_path', '')))
    require(onnx_path.is_file(), f'{label} onnx file missing from workspace: {onnx_path}')


def validate_onnx_report(report: dict[str, Any], label: str) -> None:
    require(report.get('allclose_rtol_1e_4_atol_1e_5') is True, f'{label} onnxruntime output not close to pytorch: {report}')
    require(float(report.get('max_abs_error', 1.0)) <= 1e-3, f'{label} max_abs_error too high: {report}')
    require_onnx_file(report, label)


def validate_runtime_lowbit_metadata(report: dict[str, Any], label: str) -> None:
    require(report.get('uses_lowbit_source') is True, f'{label} did not use lowbit source: {report}')
    require(report.get('writes_expanded_checkpoint') is False, f'{label} wrote expanded checkpoint: {report}')
    require(report.get('constant_folding_disabled') is True, f'{label} constant folding not disabled: {report}')
    require(report.get('unpack_lowering') == 'arithmetic_floor_div_mod_no_bitshift', f'{label} unexpected unpack lowering: {report}')
    require(int(report.get('packed_nbytes', 0)) > 0, f'{label} missing packed bytes: {report}')
    require(int(report.get('expanded_fp32_weight_nbytes', 0)) > int(report.get('packed_nbytes', 0)), f'{label} packed bytes not smaller than expanded fp32: {report}')


def validate_multi_output_report(report: dict[str, Any], label: str) -> None:
    require(report.get('all_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'{label} outputs not allclose: {report}')
    require(float(report.get('max_abs_error', 1.0)) <= 1e-3, f'{label} max_abs_error too high: {report}')
    require_onnx_file(report, label)


def validate_qkv_split_schema(report: dict[str, Any], label: str, expected_hidden_dim: int | None = None, expected_output_dim: int | None = None) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    hidden_dim = int(report.get('hidden_dim', 0))
    projection_output_dim = int(report.get('projection_output_dim', 0))
    split_schema = report.get('split_schema', {})
    sizes = split_schema.get('sizes', {}) if isinstance(split_schema, dict) else {}
    require(hidden_dim > 0, f'{label} missing hidden_dim: {report}')
    require(projection_output_dim > hidden_dim * 3, f'{label} output too small: {report}')
    if expected_hidden_dim is not None:
        require(hidden_dim == expected_hidden_dim, f'{label} hidden_dim mismatch: {report}')
    if expected_output_dim is not None:
        require(projection_output_dim == expected_output_dim, f'{label} output_dim mismatch: {report}')
    require(sizes.get('q') == hidden_dim, f'{label} q size mismatch: {report}')
    require(sizes.get('k') == hidden_dim, f'{label} k size mismatch: {report}')
    require(sizes.get('v') == hidden_dim, f'{label} v size mismatch: {report}')
    require(int(sizes.get('mlp', 0)) == projection_output_dim - hidden_dim * 3, f'{label} mlp size mismatch: {report}')
    require(int(split_schema.get('sum', 0)) == projection_output_dim, f'{label} split sum mismatch: {report}')
    return hidden_dim, projection_output_dim, split_schema, sizes


def validate_head_schema(report: dict[str, Any], label: str, expected_hidden_dim: int) -> tuple[dict[str, Any], int, int]:
    head_schema = report.get('head_schema', {})
    num_heads = int(head_schema.get('num_heads', 0)) if isinstance(head_schema, dict) else 0
    head_dim = int(head_schema.get('head_dim', 0)) if isinstance(head_schema, dict) else 0
    require(num_heads > 0 and head_dim > 0, f'{label} missing head schema: {report}')
    require(num_heads * head_dim == expected_hidden_dim, f'{label} num_heads*head_dim mismatch: {report}')
    return head_schema, num_heads, head_dim


def output_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = report.get('outputs', [])
    return {item.get('name'): item for item in outputs if isinstance(item, dict)}


def require_output_shape(report: dict[str, Any], output_name: str, label: str) -> None:
    outputs = output_by_name(report)
    require(output_name in outputs, f'{label} missing output {output_name}: {report}')
    require(outputs[output_name].get('expected_shape') == outputs[output_name].get('output_shape'), f'{label} {output_name} output shape mismatch: {report}')


def main() -> None:
    lowbit = load(LOWBIT)
    layer = load(LAYER)
    compare = load(COMPARE)
    compare_layers = load(COMPARE_LAYERS)
    compare_all = load(COMPARE_ALL)
    layer_onnx = load(LAYER_ONNX)
    runtime_linear_onnx = load(RUNTIME_LINEAR_ONNX)
    multi_linear_onnx = load(MULTI_LINEAR_ONNX)
    same_block_projection_onnx = load(SAME_BLOCK_PROJECTION_ONNX)
    qkv_mlp_split_onnx = load(QKV_MLP_SPLIT_ONNX)
    qkv_head_reshape_onnx = load(QKV_HEAD_RESHAPE_ONNX)
    qkv_seq_head_layout_onnx = load(QKV_SEQ_HEAD_LAYOUT_ONNX)
    attention_math_onnx = load(ATTENTION_MATH_ONNX)

    require(lowbit.get('status') == 'passed', f'lowbit probe did not pass: {lowbit}')
    require(lowbit.get('milestone_reached') in {'state_dict_inspect', 'cpu_conversion_plan'}, f'lowbit milestone too early: {lowbit}')

    require(layer.get('layer'), f'missing layer name: {layer}')
    require(layer.get('orig_shape') is not None, f'missing layer orig_shape: {layer}')

    require(compare.get('exact_equal') is True, f'single layer unpacked compare not exact: {compare}')
    require(compare_layers.get('all_exact_equal') is True, f'sampled layer compare not exact: {compare_layers}')
    require(compare_all.get('all_exact_equal') is True, f'all layer compare not exact: {compare_all}')
    require(int(compare_all.get('total_compared_layers', 0)) > 0, f'no layers compared: {compare_all}')
    require(int(compare_all.get('failure_count', -1)) == 0, f'all layer failures present: {compare_all}')

    validate_onnx_report(layer_onnx, 'recovered-layer')

    validate_runtime_lowbit_metadata(runtime_linear_onnx, 'runtime-lowbit-linear')
    validate_onnx_report(runtime_linear_onnx, 'runtime-lowbit-linear')

    validate_runtime_lowbit_metadata(multi_linear_onnx, 'multi-runtime-lowbit-linear')
    require(multi_linear_onnx.get('block_kind') == 'multi_runtime_lowbit_linear_bundle', f'multi-linear unexpected block kind: {multi_linear_onnx}')
    require(int(multi_linear_onnx.get('layer_count', 0)) >= 2, f'multi-linear layer_count too small: {multi_linear_onnx}')
    validate_multi_output_report(multi_linear_onnx, 'multi-runtime-lowbit-linear')

    validate_runtime_lowbit_metadata(same_block_projection_onnx, 'same-block-projection')
    require(same_block_projection_onnx.get('bundle_kind') == 'same_block_qkv_mlp_proj_and_to_out_projection_bundle', f'same-block unexpected bundle kind: {same_block_projection_onnx}')
    require(same_block_projection_onnx.get('is_real_transformer_block') is False, f'same-block bundle must not claim real transformer block: {same_block_projection_onnx}')
    require(int(same_block_projection_onnx.get('block_index', -1)) == 0, f'same-block unexpected block index: {same_block_projection_onnx}')
    require(int(same_block_projection_onnx.get('layer_count', 0)) >= 2, f'same-block layer_count too small: {same_block_projection_onnx}')
    require('single_transformer_blocks.0.attn.to_qkv_mlp_proj' in same_block_projection_onnx.get('layers', []), f'same-block missing to_qkv_mlp_proj: {same_block_projection_onnx}')
    require('single_transformer_blocks.0.attn.to_out' in same_block_projection_onnx.get('layers', []), f'same-block missing to_out: {same_block_projection_onnx}')
    validate_multi_output_report(same_block_projection_onnx, 'same-block-projection')

    validate_runtime_lowbit_metadata(qkv_mlp_split_onnx, 'qkv-mlp-split')
    require(qkv_mlp_split_onnx.get('graph_kind') == 'qkv_mlp_projection_and_shape_derived_split', f'qkv/mlp split unexpected graph kind: {qkv_mlp_split_onnx}')
    require(qkv_mlp_split_onnx.get('is_attention') is False, f'qkv/mlp split must not claim attention: {qkv_mlp_split_onnx}')
    require(qkv_mlp_split_onnx.get('is_real_transformer_block') is False, f'qkv/mlp split must not claim real transformer block: {qkv_mlp_split_onnx}')
    require(int(qkv_mlp_split_onnx.get('block_index', -1)) == 0, f'qkv/mlp split unexpected block index: {qkv_mlp_split_onnx}')
    require(qkv_mlp_split_onnx.get('layer') == 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', f'qkv/mlp split unexpected layer: {qkv_mlp_split_onnx}')
    hidden_dim, projection_output_dim, _, _ = validate_qkv_split_schema(qkv_mlp_split_onnx, 'qkv-mlp-split')
    validate_multi_output_report(qkv_mlp_split_onnx, 'qkv-mlp-split')

    validate_runtime_lowbit_metadata(qkv_head_reshape_onnx, 'qkv-head-reshape')
    require(qkv_head_reshape_onnx.get('graph_kind') == 'qkv_projection_split_and_head_reshape', f'qkv head unexpected graph kind: {qkv_head_reshape_onnx}')
    require(qkv_head_reshape_onnx.get('is_attention') is False, f'qkv head must not claim attention: {qkv_head_reshape_onnx}')
    require(qkv_head_reshape_onnx.get('is_real_transformer_block') is False, f'qkv head must not claim real transformer block: {qkv_head_reshape_onnx}')
    require(int(qkv_head_reshape_onnx.get('block_index', -1)) == 0, f'qkv head unexpected block index: {qkv_head_reshape_onnx}')
    require(qkv_head_reshape_onnx.get('layer') == 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', f'qkv head unexpected layer: {qkv_head_reshape_onnx}')
    validate_qkv_split_schema(qkv_head_reshape_onnx, 'qkv-head-reshape', hidden_dim, projection_output_dim)
    head_schema, num_heads, head_dim = validate_head_schema(qkv_head_reshape_onnx, 'qkv-head-reshape', hidden_dim)
    require_output_shape(qkv_head_reshape_onnx, 'q_heads', 'qkv-head-reshape')
    require_output_shape(qkv_head_reshape_onnx, 'k_heads', 'qkv-head-reshape')
    require_output_shape(qkv_head_reshape_onnx, 'v_heads', 'qkv-head-reshape')
    validate_multi_output_report(qkv_head_reshape_onnx, 'qkv-head-reshape')

    validate_runtime_lowbit_metadata(qkv_seq_head_layout_onnx, 'qkv-seq-head-layout')
    require(qkv_seq_head_layout_onnx.get('graph_kind') == 'qkv_projection_split_and_sequence_head_layout', f'qkv seq head unexpected graph kind: {qkv_seq_head_layout_onnx}')
    require(qkv_seq_head_layout_onnx.get('is_attention') is False, f'qkv seq head must not claim attention: {qkv_seq_head_layout_onnx}')
    require(qkv_seq_head_layout_onnx.get('is_real_transformer_block') is False, f'qkv seq head must not claim real transformer block: {qkv_seq_head_layout_onnx}')
    require(int(qkv_seq_head_layout_onnx.get('block_index', -1)) == 0, f'qkv seq head unexpected block index: {qkv_seq_head_layout_onnx}')
    require(qkv_seq_head_layout_onnx.get('layer') == 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', f'qkv seq head unexpected layer: {qkv_seq_head_layout_onnx}')
    validate_qkv_split_schema(qkv_seq_head_layout_onnx, 'qkv-seq-head-layout', hidden_dim, projection_output_dim)
    seq_head_schema, seq_num_heads, seq_head_dim = validate_head_schema(qkv_seq_head_layout_onnx, 'qkv-seq-head-layout', hidden_dim)
    require(seq_num_heads == num_heads and seq_head_dim == head_dim, f'qkv seq head schema mismatch against previous head reshape: {qkv_seq_head_layout_onnx}')
    require(seq_head_schema.get('layout') == 'batch_heads_seq_head_dim', f'qkv seq head unexpected layout: {qkv_seq_head_layout_onnx}')
    seq_len = int(qkv_seq_head_layout_onnx.get('sequence_length', 0))
    require(seq_len > 0, f'qkv seq head missing sequence length: {qkv_seq_head_layout_onnx}')
    require_output_shape(qkv_seq_head_layout_onnx, 'q_seq_heads', 'qkv-seq-head-layout')
    require_output_shape(qkv_seq_head_layout_onnx, 'k_seq_heads', 'qkv-seq-head-layout')
    require_output_shape(qkv_seq_head_layout_onnx, 'v_seq_heads', 'qkv-seq-head-layout')
    require_output_shape(qkv_seq_head_layout_onnx, 'mlp', 'qkv-seq-head-layout')
    validate_multi_output_report(qkv_seq_head_layout_onnx, 'qkv-seq-head-layout')

    validate_runtime_lowbit_metadata(attention_math_onnx, 'attention-math')
    require(attention_math_onnx.get('graph_kind') == 'qkv_projection_scaled_dot_product_attention_math', f'attention math unexpected graph kind: {attention_math_onnx}')
    require(attention_math_onnx.get('is_attention_math') is True, f'attention math must claim attention math: {attention_math_onnx}')
    require(attention_math_onnx.get('is_attention_with_to_out') is False, f'attention math must not claim to_out connection: {attention_math_onnx}')
    require(attention_math_onnx.get('is_real_transformer_block') is False, f'attention math must not claim real transformer block: {attention_math_onnx}')
    require(int(attention_math_onnx.get('block_index', -1)) == 0, f'attention math unexpected block index: {attention_math_onnx}')
    require(attention_math_onnx.get('layer') == 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', f'attention math unexpected layer: {attention_math_onnx}')
    validate_qkv_split_schema(attention_math_onnx, 'attention-math', hidden_dim, projection_output_dim)
    attn_head_schema, attn_num_heads, attn_head_dim = validate_head_schema(attention_math_onnx, 'attention-math', hidden_dim)
    require(attn_num_heads == num_heads and attn_head_dim == head_dim, f'attention math head schema mismatch against previous head reshape: {attention_math_onnx}')
    require(attn_head_schema.get('layout') == 'batch_heads_seq_head_dim', f'attention math unexpected layout: {attention_math_onnx}')
    attn_math = attention_math_onnx.get('attention_math', {})
    require(attn_math.get('softmax_dim') == -1, f'attention math unexpected softmax dim: {attention_math_onnx}')
    require(attn_math.get('mask') is None, f'attention math should not include mask yet: {attention_math_onnx}')
    require(int(attention_math_onnx.get('sequence_length', 0)) == seq_len, f'attention math sequence length mismatch: {attention_math_onnx}')
    require_output_shape(attention_math_onnx, 'context', 'attention-math')
    require_output_shape(attention_math_onnx, 'weights', 'attention-math')
    require_output_shape(attention_math_onnx, 'scores', 'attention-math')
    require_output_shape(attention_math_onnx, 'q_heads', 'attention-math')
    require_output_shape(attention_math_onnx, 'k_heads', 'attention-math')
    require_output_shape(attention_math_onnx, 'v_heads', 'attention-math')
    validate_multi_output_report(attention_math_onnx, 'attention-math')

    summary = {
        'ok': True,
        'lowbit_milestone_reached': lowbit.get('milestone_reached'),
        'compared_layers': compare_all.get('total_compared_layers'),
        'all_layers_exact_equal': compare_all.get('all_exact_equal'),
        'runtime_linear_onnx_layer': runtime_linear_onnx.get('layer'),
        'runtime_linear_onnx_max_abs_error': runtime_linear_onnx.get('max_abs_error'),
        'multi_linear_block_kind': multi_linear_onnx.get('block_kind'),
        'multi_linear_layer_count': multi_linear_onnx.get('layer_count'),
        'multi_linear_max_abs_error': multi_linear_onnx.get('max_abs_error'),
        'same_block_projection_bundle_kind': same_block_projection_onnx.get('bundle_kind'),
        'same_block_projection_block_index': same_block_projection_onnx.get('block_index'),
        'same_block_projection_is_real_transformer_block': same_block_projection_onnx.get('is_real_transformer_block'),
        'same_block_projection_layers': same_block_projection_onnx.get('layers'),
        'same_block_projection_max_abs_error': same_block_projection_onnx.get('max_abs_error'),
        'qkv_mlp_split_graph_kind': qkv_mlp_split_onnx.get('graph_kind'),
        'qkv_mlp_split_layer': qkv_mlp_split_onnx.get('layer'),
        'qkv_mlp_split_hidden_dim': qkv_mlp_split_onnx.get('hidden_dim'),
        'qkv_mlp_split_projection_output_dim': qkv_mlp_split_onnx.get('projection_output_dim'),
        'qkv_mlp_split_schema': qkv_mlp_split_onnx.get('split_schema'),
        'qkv_mlp_split_max_abs_error': qkv_mlp_split_onnx.get('max_abs_error'),
        'qkv_head_reshape_graph_kind': qkv_head_reshape_onnx.get('graph_kind'),
        'qkv_head_reshape_head_schema': qkv_head_reshape_onnx.get('head_schema'),
        'qkv_head_reshape_split_schema': qkv_head_reshape_onnx.get('split_schema'),
        'qkv_head_reshape_max_abs_error': qkv_head_reshape_onnx.get('max_abs_error'),
        'qkv_seq_head_layout_graph_kind': qkv_seq_head_layout_onnx.get('graph_kind'),
        'qkv_seq_head_layout_sequence_length': qkv_seq_head_layout_onnx.get('sequence_length'),
        'qkv_seq_head_layout_head_schema': qkv_seq_head_layout_onnx.get('head_schema'),
        'qkv_seq_head_layout_split_schema': qkv_seq_head_layout_onnx.get('split_schema'),
        'qkv_seq_head_layout_max_abs_error': qkv_seq_head_layout_onnx.get('max_abs_error'),
        'attention_math_graph_kind': attention_math_onnx.get('graph_kind'),
        'attention_math_sequence_length': attention_math_onnx.get('sequence_length'),
        'attention_math_head_schema': attention_math_onnx.get('head_schema'),
        'attention_math': attention_math_onnx.get('attention_math'),
        'attention_math_max_abs_error': attention_math_onnx.get('max_abs_error'),
        'claim': 'qkv_projection_scaled_dot_product_attention_math_onnxruntime_cpu_verified_not_to_out_or_transformer_block_or_full_bonsai_pipeline',
    }
    out = ROOT / 'bonsai-lowbit-validation'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        out = ROOT / 'bonsai-lowbit-validation'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(json.dumps({'ok': False, 'error': str(exc)}, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'ok': False, 'error': str(exc)}, indent=2))
        sys.exit(1)
