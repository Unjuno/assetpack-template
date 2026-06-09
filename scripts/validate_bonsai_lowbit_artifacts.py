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


def main() -> None:
    lowbit = load(LOWBIT)
    layer = load(LAYER)
    compare = load(COMPARE)
    compare_layers = load(COMPARE_LAYERS)
    compare_all = load(COMPARE_ALL)
    layer_onnx = load(LAYER_ONNX)

    require(lowbit.get('status') == 'passed', f'lowbit probe did not pass: {lowbit}')
    require(lowbit.get('milestone_reached') in {'state_dict_inspect', 'cpu_conversion_plan'}, f'lowbit milestone too early: {lowbit}')

    require(layer.get('layer'), f'missing layer name: {layer}')
    require(layer.get('orig_shape') is not None, f'missing layer orig_shape: {layer}')

    require(compare.get('exact_equal') is True, f'single layer unpacked compare not exact: {compare}')
    require(compare_layers.get('all_exact_equal') is True, f'sampled layer compare not exact: {compare_layers}')
    require(compare_all.get('all_exact_equal') is True, f'all layer compare not exact: {compare_all}')
    require(int(compare_all.get('total_compared_layers', 0)) > 0, f'no layers compared: {compare_all}')
    require(int(compare_all.get('failure_count', -1)) == 0, f'all layer failures present: {compare_all}')

    onnx_path = Path(str(layer_onnx.get('onnx_path', '')))
    require(layer_onnx.get('allclose_rtol_1e_4_atol_1e_5') is True, f'onnxruntime output not close to pytorch: {layer_onnx}')
    require(float(layer_onnx.get('max_abs_error', 1.0)) <= 1e-3, f'onnx max_abs_error too high: {layer_onnx}')
    require(int(layer_onnx.get('total_onnx_artifact_size_bytes', 0)) > 0, f'empty onnx artifact: {layer_onnx}')
    require(onnx_path.is_file(), f'onnx file missing from workspace: {onnx_path}')

    summary = {
        'ok': True,
        'lowbit_milestone_reached': lowbit.get('milestone_reached'),
        'compared_layers': compare_all.get('total_compared_layers'),
        'all_layers_exact_equal': compare_all.get('all_exact_equal'),
        'onnx_layer': layer_onnx.get('layer'),
        'onnx_output_shape': layer_onnx.get('output_shape'),
        'onnx_max_abs_error': layer_onnx.get('max_abs_error'),
        'onnx_total_artifact_size_bytes': layer_onnx.get('total_onnx_artifact_size_bytes'),
        'claim': 'single_dequantized_layer_onnxruntime_cpu_verified_not_full_bonsai_pipeline',
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
