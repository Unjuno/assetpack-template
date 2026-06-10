#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('reports')
SAME_BLOCK = ROOT / 'bonsai-lowbit-same-block-projection-onnx' / 'report.json'
QKV_SEQ = ROOT / 'bonsai-lowbit-qkv-seq-head-layout-onnx' / 'report.json'
ATTENTION_MATH = ROOT / 'bonsai-lowbit-attention-math-onnx' / 'report.json'
CONTEXT = ROOT / 'bonsai-lowbit-attention-context-composition-onnx' / 'report.json'
OUT = ROOT / 'bonsai-lowbit-attention-context-composition-validation' / 'report.json'


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


def require_output(report: dict[str, Any], name: str) -> None:
    outputs = outputs_by_name(report)
    require(name in outputs, f'missing output {name}: {report}')
    item = outputs[name]
    require(item.get('output_shape') == item.get('expected_shape'), f'{name} shape mismatch: {item}')
    require(item.get('allclose_rtol_1e_4_atol_1e_5') is True, f'{name} not allclose: {item}')


def main() -> None:
    same_block = load(SAME_BLOCK)
    qkv_seq = load(QKV_SEQ)
    attention_math = load(ATTENTION_MATH)
    context = load(CONTEXT)

    require(context.get('uses_lowbit_source') is True, f'context composition did not use lowbit source: {context}')
    require(context.get('writes_expanded_checkpoint') is False, f'context composition wrote expanded checkpoint: {context}')
    require(context.get('constant_folding_disabled') is True, f'context composition constant folding not disabled: {context}')
    require(context.get('unpack_lowering') == 'arithmetic_floor_div_mod_no_bitshift', f'unexpected unpack lowering: {context}')
    require(context.get('graph_kind') == 'qkv_projection_attention_context_composition', f'unexpected graph kind: {context}')
    require(context.get('is_attention_math') is True, f'missing attention math flag: {context}')
    require(context.get('is_attention_context_composition') is True, f'missing context composition flag: {context}')
    require(context.get('is_attention_with_to_out') is False, f'must not claim to_out connection: {context}')
    require(context.get('to_out_connection_attempted') is False, f'must not attempt to_out connection: {context}')
    require(context.get('is_real_transformer_block') is False, f'must not claim real transformer block: {context}')
    require(context.get('layer') == 'single_transformer_blocks.0.attn.to_qkv_mlp_proj', f'unexpected layer: {context}')
    require(context.get('inspected_to_out_layer') == 'single_transformer_blocks.0.attn.to_out', f'unexpected inspected to_out layer: {context}')

    hidden_dim = int(qkv_seq.get('hidden_dim', 0))
    seq_len = int(qkv_seq.get('sequence_length', 0))
    require(int(context.get('hidden_dim', 0)) == hidden_dim, f'hidden_dim mismatch: {context}')
    require(int(context.get('projection_output_dim', 0)) == int(qkv_seq.get('projection_output_dim', 0)), f'projection_output_dim mismatch: {context}')
    require(int(context.get('sequence_length', 0)) == seq_len, f'sequence_length mismatch: {context}')
    require(context.get('head_schema', {}).get('layout') == 'batch_heads_seq_head_dim', f'unexpected head layout: {context}')
    require(context.get('head_schema', {}).get('num_heads') == attention_math.get('head_schema', {}).get('num_heads'), f'num_heads mismatch: {context}')
    require(context.get('head_schema', {}).get('head_dim') == attention_math.get('head_schema', {}).get('head_dim'), f'head_dim mismatch: {context}')

    composition = context.get('composition_schema', {})
    require(composition.get('input_layout') == 'batch_heads_seq_head_dim', f'bad input layout: {context}')
    require(composition.get('intermediate_layout') == 'batch_seq_heads_head_dim', f'bad intermediate layout: {context}')
    require(composition.get('flat_layout') == 'batch_seq_hidden', f'bad flat layout: {context}')
    require(int(composition.get('context_flat_width', 0)) == hidden_dim, f'flat width mismatch: {context}')

    to_out = context.get('to_out_input_schema', {})
    same_block_inputs = same_block.get('inputs', {}) if isinstance(same_block.get('inputs', {}), dict) else {}
    same_block_attn_context_shape = same_block_inputs.get('attn_context_shape', [])
    require(to_out.get('layer') == 'single_transformer_blocks.0.attn.to_out', f'bad to_out schema layer: {context}')
    require(int(to_out.get('candidate_context_width', 0)) == hidden_dim, f'candidate width mismatch: {context}')
    require(to_out.get('candidate_width_matches') is False, f'candidate must not match to_out yet: {context}')
    require(to_out.get('connection_attempted') is False, f'to_out connection must not be attempted: {context}')
    if isinstance(same_block_attn_context_shape, list) and len(same_block_attn_context_shape) >= 2:
        require(int(to_out.get('expected_in_features', 0)) == int(same_block_attn_context_shape[1]), f'to_out width mismatch against same-block probe: {context}')
    require(int(to_out.get('expected_in_features', 0)) > hidden_dim, f'to_out expected width should be wider than flattened context: {context}')

    for name in ['context_heads', 'context_seq_heads', 'context_flat', 'weights', 'scores']:
        require_output(context, name)
    require(context.get('all_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'context composition outputs not allclose: {context}')
    require(float(context.get('max_abs_error', 1.0)) <= 1e-3, f'context composition max_abs_error too high: {context}')
    require(int(context.get('onnx_size_bytes', 0)) > 0, f'empty onnx artifact: {context}')

    summary = {
        'ok': True,
        'graph_kind': context.get('graph_kind'),
        'sequence_length': context.get('sequence_length'),
        'head_schema': context.get('head_schema'),
        'composition_schema': context.get('composition_schema'),
        'to_out_input_schema': context.get('to_out_input_schema'),
        'max_abs_error': context.get('max_abs_error'),
        'claim': 'attention_context_composition_onnxruntime_cpu_verified_not_connected_to_to_out_or_transformer_block_or_full_bonsai_pipeline',
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
