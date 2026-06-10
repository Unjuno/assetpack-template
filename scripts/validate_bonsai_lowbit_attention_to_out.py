#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('reports')
SOURCE = ROOT / 'bonsai-lowbit-attention-to-out-onnx' / 'report.json'
OUT = ROOT / 'bonsai-lowbit-attention-to-out-validation' / 'report.json'


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
    require(source.get('graph_kind') == 'qkv_projection_attention_to_out', f'unexpected graph kind: {source}')
    require(source.get('is_attention_to_out_path') is True, f'missing attention to_out path flag: {source}')
    require(source.get('is_attention_with_to_out') is True, f'must claim to_out connection: {source}')
    require(source.get('to_out_connection_attempted') is True, f'to_out connection must be attempted: {source}')
    require(source.get('semantic_to_out_input_verified') is True, f'semantic input must be verified: {source}')
    require(source.get('is_real_transformer_block') is False, f'must not claim real transformer block: {source}')
    require(source.get('is_full_bonsai_pipeline') is False, f'must not claim full bonsai pipeline: {source}')

    hidden_dim = int(source.get('hidden_dim', 0))
    require(hidden_dim > 0, f'invalid hidden_dim: {source}')
    semantic = source.get('semantic_input_schema', {})
    require(semantic.get('method') == 'concat_normed_attention_context_with_swiglu_mlp', f'bad semantic method: {source}')
    require(int(semantic.get('context_flat_width', 0)) == hidden_dim, f'context width mismatch: {source}')
    require(semantic.get('mlp_activation') == 'swiglu', f'bad mlp activation: {source}')
    require(int(semantic.get('semantic_input_width', 0)) == int(semantic.get('to_out_expected_in_features', -1)), f'semantic width mismatch: {source}')
    require(semantic.get('semantic_width_matches_to_out') is True, f'semantic width flag mismatch: {source}')

    to_out = source.get('to_out_schema', {})
    require(to_out.get('layer') == 'single_transformer_blocks.0.attn.to_out', f'bad to_out layer: {source}')
    require(to_out.get('connected') is True, f'to_out must be connected: {source}')
    require(int(to_out.get('in_features', 0)) == int(semantic.get('semantic_input_width', 0)), f'to_out in width mismatch: {source}')
    require(int(to_out.get('out_features', 0)) == hidden_dim, f'to_out output width mismatch: {source}')

    for name in ['context_flat', 'mlp_swiglu', 'semantic_to_out_input', 'to_out_output', 'weights', 'scores']:
        require_output(source, name)
    require(source.get('all_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'source probe outputs not allclose: {source}')
    require(float(source.get('max_abs_error', 1.0)) <= 1e-3, f'source probe max_abs_error too high: {source}')
    require(int(source.get('onnx_size_bytes', 0)) > 0, f'empty onnx artifact: {source}')

    summary = {
        'ok': True,
        'graph_kind': source.get('graph_kind'),
        'sequence_length': source.get('sequence_length'),
        'head_schema': source.get('head_schema'),
        'semantic_input_schema': source.get('semantic_input_schema'),
        'to_out_schema': source.get('to_out_schema'),
        'max_abs_error': source.get('max_abs_error'),
        'claim': 'attention_to_out_path_onnxruntime_cpu_verified_not_real_transformer_block_or_full_bonsai_pipeline',
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
