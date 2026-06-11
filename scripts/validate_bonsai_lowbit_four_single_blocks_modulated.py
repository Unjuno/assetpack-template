#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('reports')
SOURCE = ROOT / 'bonsai-lowbit-four-single-blocks-modulated-onnx' / 'report.json'
OUT = ROOT / 'bonsai-lowbit-four-single-blocks-modulated-validation' / 'report.json'
BLOCK_INDICES = [0, 1, 2, 3]


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


def require_output(report: dict[str, Any], name: str) -> dict[str, Any]:
    outputs = outputs_by_name(report)
    require(name in outputs, f'missing output {name}: {report}')
    item = outputs[name]
    require(item.get('output_shape') == item.get('expected_shape'), f'{name} shape mismatch: {item}')
    require(item.get('allclose_rtol_1e_4_atol_1e_5') is True, f'{name} not allclose: {item}')
    return item


def main() -> None:
    source = load(SOURCE)
    require(source.get('uses_lowbit_source') is True, f'source probe did not use lowbit source: {source}')
    require(source.get('writes_expanded_checkpoint') is False, f'source probe wrote expanded checkpoint: {source}')
    require(source.get('constant_folding_disabled') is True, f'source probe constant folding not disabled: {source}')
    require(source.get('unpack_lowering') == 'arithmetic_floor_div_mod_no_bitshift', f'unexpected unpack lowering: {source}')
    require(source.get('graph_kind') == 'four_single_blocks_modulated_attention_to_out_residual_stack', f'unexpected graph kind: {source}')
    require(source.get('is_four_single_blocks_modulated_stack') is True, f'missing stack flag: {source}')
    require(source.get('block_indices') == BLOCK_INDICES, f'bad block indices: {source}')
    require(int(source.get('sequence_block_count', 0)) == 4, f'bad sequence block count: {source}')
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

    modulation = source.get('modulation_schema', {})
    require(modulation.get('method') == 'shared_single_stream_modulation_linear_temb_chunk_shift_scale_gate', f'bad modulation method: {source}')
    require(modulation.get('layer') == 'single_stream_modulation.linear.weight', f'bad modulation layer: {source}')
    require(modulation.get('chunks') == ['shift', 'scale', 'gate'], f'bad modulation chunks: {source}')
    require(int(modulation.get('input_width', 0)) == hidden_dim, f'bad modulation input width: {source}')
    require(int(modulation.get('output_width', 0)) == 3 * hidden_dim, f'bad modulation output width: {source}')

    residual = source.get('residual_schema', {})
    require(residual.get('method') == 'sequential_hidden_plus_gate_times_to_out', f'bad residual method: {source}')
    require(residual.get('inputs') == ['hidden', 'block0_output', 'block1_output', 'block2_output'], f'bad residual inputs: {source}')
    require(residual.get('final_output') == 'block3_output', f'bad residual final output: {source}')
    require(int(residual.get('block_output_width', 0)) == hidden_dim, f'block output width mismatch: {source}')

    layers = source.get('lowbit_layers', [])
    require(len(layers) == 4, f'expected four lowbit layer entries: {source}')
    for expected, item in zip(BLOCK_INDICES, layers):
        require(int(item.get('block_index', -1)) == expected, f'bad lowbit layer block index: {source}')
        require(item.get('qkv_mlp_proj') == f'single_transformer_blocks.{expected}.attn.to_qkv_mlp_proj', f'bad qkv layer: {source}')
        require(item.get('to_out') == f'single_transformer_blocks.{expected}.attn.to_out', f'bad to_out layer: {source}')

    for index in BLOCK_INDICES:
        require_output(source, f'block{index}_output')
        require_output(source, f'block{index}_semantic_to_out_input')
        require_output(source, f'block{index}_gate')
        require_output(source, f'block{index}_weights')
    require(source.get('all_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'source probe outputs not allclose: {source}')
    require(float(source.get('max_abs_error', 1.0)) <= 1e-3, f'source probe max_abs_error too high: {source}')
    require(int(source.get('onnx_size_bytes', 0)) > 0, f'empty onnx artifact: {source}')

    summary = {
        'ok': True,
        'graph_kind': source.get('graph_kind'),
        'block_indices': source.get('block_indices'),
        'sequence_block_count': source.get('sequence_block_count'),
        'head_schema': source.get('head_schema'),
        'modulation_schema': source.get('modulation_schema'),
        'residual_schema': source.get('residual_schema'),
        'max_abs_error': source.get('max_abs_error'),
        'claim': 'four_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline',
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
