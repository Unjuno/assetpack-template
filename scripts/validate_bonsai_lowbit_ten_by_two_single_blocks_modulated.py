#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('reports')
SOURCE = ROOT / 'bonsai-lowbit-ten-by-two-single-blocks-modulated-onnx' / 'report.json'
OUT = ROOT / 'bonsai-lowbit-ten-by-two-single-blocks-modulated-validation' / 'report.json'
BLOCK_INDICES = list(range(20))
SEGMENTS = [[index, index + 1] for index in range(0, 20, 2)]
CRITICAL_OUTPUTS = [
    *(f'segment{a}_{b}_block0_output' for a, b in SEGMENTS),
    *(f'segment{a}_{b}_block1_output' for a, b in SEGMENTS),
    'chained_final_block19_output',
]
DIAGNOSTIC_SUFFIXES = [
    'block0_semantic_to_out_input',
    'block1_semantic_to_out_input',
    'block0_gate',
    'block1_gate',
    'block0_weights',
    'block1_weights',
]


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f'missing report: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise AssertionError(f'report is not an object: {path}')
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def outputs_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get('name'): item for item in report.get('outputs', []) if isinstance(item, dict)}


def require_output_present(report: dict[str, Any], name: str) -> dict[str, Any]:
    outputs = outputs_by_name(report)
    require(name in outputs, f'missing output {name}: {report}')
    return outputs[name]


def require_critical_output(report: dict[str, Any], name: str) -> dict[str, Any]:
    item = require_output_present(report, name)
    require(item.get('category') == 'critical', f'{name} is not marked critical: {item}')
    require(item.get('allclose_rtol_1e_4_atol_1e_5') is True, f'{name} critical output not allclose: {item}')
    return item


def require_diagnostic_output(report: dict[str, Any], name: str) -> dict[str, Any]:
    item = require_output_present(report, name)
    require(item.get('category') == 'diagnostic', f'{name} is not marked diagnostic: {item}')
    require('diagnostic_allclose_rtol_1e_4_atol_1e_4' in item, f'{name} missing diagnostic allclose field: {item}')
    return item


def main() -> None:
    source = load(SOURCE)
    require(source.get('uses_lowbit_source') is True, f'source probe did not use lowbit source: {source}')
    require(source.get('writes_expanded_checkpoint') is False, f'source probe wrote expanded checkpoint: {source}')
    require(source.get('constant_folding_disabled') is True, f'source probe constant folding not disabled: {source}')
    require(source.get('unpack_lowering') == 'arithmetic_floor_div_mod_no_bitshift', f'unexpected unpack lowering: {source}')
    require(source.get('graph_kind') == 'ten_by_two_single_blocks_modulated_attention_to_out_residual_stack', f'unexpected graph kind: {source}')
    require(source.get('is_ten_by_two_single_blocks_modulated_stack') is True, f'missing ten-by-two flag: {source}')
    require(source.get('block_indices') == BLOCK_INDICES, f'bad block indices: {source}')
    require(source.get('segment_block_indices') == SEGMENTS, f'bad segment block indices: {source}')
    require(int(source.get('sequence_block_count', 0)) == 20, f'bad sequence block count: {source}')
    require(int(source.get('onnx_segment_count', 0)) == 10, f'bad onnx segment count: {source}')
    require(source.get('is_single_monolithic_onnx') is False, f'must not be monolithic ONNX: {source}')
    require(source.get('is_attention_with_to_out') is True, f'must include to_out path: {source}')
    require(source.get('to_out_connection_attempted') is True, f'to_out must be connected: {source}')
    require(source.get('has_modulation') is True, f'modulation must be present: {source}')
    require(source.get('has_gate_residual') is True, f'gate residual must be present: {source}')
    require(source.get('is_real_transformer_block') is False, f'must not claim real transformer block: {source}')
    require(source.get('is_full_bonsai_pipeline') is False, f'must not claim full bonsai pipeline: {source}')

    hidden_dim = int(source.get('hidden_dim', 0))
    require(hidden_dim == 3072, f'unexpected hidden_dim: {source}')
    require(int(source.get('sequence_length', 0)) == 4, f'unexpected sequence length: {source}')
    require(int(source.get('semantic_input_width', 0)) == 12288, f'bad semantic input width: {source}')

    residual = source.get('residual_schema', {})
    require(residual.get('method') == 'sequential_hidden_plus_gate_times_to_out_across_ten_onnx_segments', f'bad residual method: {source}')
    require(residual.get('final_output') == 'block19_output', f'bad final output: {source}')

    policy = source.get('pass_fail_policy', {})
    require('critical_outputs' in policy and 'diagnostic_outputs' in policy, f'missing pass/fail policy: {source}')

    layers = source.get('lowbit_layers', [])
    require(len(layers) == 20, f'expected twenty lowbit layer entries: {source}')
    for expected, item in zip(BLOCK_INDICES, layers):
        require(int(item.get('block_index', -1)) == expected, f'bad lowbit layer block index: {source}')
        require(item.get('qkv_mlp_proj') == f'single_transformer_blocks.{expected}.attn.to_qkv_mlp_proj', f'bad qkv layer: {source}')
        require(item.get('to_out') == f'single_transformer_blocks.{expected}.attn.to_out', f'bad to_out layer: {source}')

    for name in CRITICAL_OUTPUTS:
        require_critical_output(source, name)
    for segment in SEGMENTS:
        prefix = f'segment{segment[0]}_{segment[1]}'
        for suffix in DIAGNOSTIC_SUFFIXES:
            require_diagnostic_output(source, f'{prefix}_{suffix}')

    require(source.get('critical_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'critical outputs not allclose: {source}')
    require(float(source.get('critical_max_abs_error', 1.0)) <= 1e-3, f'critical max_abs_error too high: {source}')
    require(int(source.get('total_onnx_segment_size_bytes', 0)) > 0, f'empty onnx segment artifact size: {source}')

    summary = {
        'ok': True,
        'graph_kind': source.get('graph_kind'),
        'block_indices': source.get('block_indices'),
        'segment_block_indices': source.get('segment_block_indices'),
        'sequence_block_count': source.get('sequence_block_count'),
        'onnx_segment_count': source.get('onnx_segment_count'),
        'is_single_monolithic_onnx': source.get('is_single_monolithic_onnx'),
        'critical_max_abs_error': source.get('critical_max_abs_error'),
        'diagnostic_max_abs_error': source.get('diagnostic_max_abs_error'),
        'critical_outputs_allclose_rtol_1e_4_atol_1e_5': source.get('critical_outputs_allclose_rtol_1e_4_atol_1e_5'),
        'all_outputs_allclose_rtol_1e_4_atol_1e_5': source.get('all_outputs_allclose_rtol_1e_4_atol_1e_5'),
        'claim': 'ten_by_two_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_path_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'ok': False, 'error': str(exc)}, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'ok': False, 'error': str(exc)}, indent=2))
        sys.exit(1)
