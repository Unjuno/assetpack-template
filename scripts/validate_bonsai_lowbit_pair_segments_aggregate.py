#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_SEGMENTS = [(index, index + 1) for index in range(0, 20, 2)]
OUT = Path('reports/bonsai-lowbit-pair-segments-aggregate-validation/report.json')


def load_reports(root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob('**/report.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('graph_kind') == 'pair_segment_single_blocks_modulated_attention_to_out_residual_stack':
            data['_report_path'] = str(path)
            reports.append(data)
    return reports


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    reports = load_reports(root)
    by_segment: dict[tuple[int, int], dict[str, Any]] = {}
    for report in reports:
        indices = report.get('block_indices')
        if isinstance(indices, list) and len(indices) == 2:
            by_segment[(int(indices[0]), int(indices[1]))] = report

    missing = [list(segment) for segment in EXPECTED_SEGMENTS if segment not in by_segment]
    require(not missing, f'missing pair segment reports: {missing}')

    segment_summaries = []
    for segment in EXPECTED_SEGMENTS:
        report = by_segment[segment]
        require(report.get('uses_lowbit_source') is True, f'{segment} did not use lowbit source')
        require(report.get('writes_expanded_checkpoint') is False, f'{segment} wrote expanded checkpoint')
        require(report.get('constant_folding_disabled') is True, f'{segment} constant folding not disabled')
        require(report.get('unpack_lowering') == 'arithmetic_floor_div_mod_no_bitshift', f'{segment} bad unpack lowering')
        require(report.get('block_indices') == list(segment), f'{segment} bad block indices')
        require(report.get('segment_block_indices') == [list(segment)], f'{segment} bad segment block indices')
        require(report.get('sequence_block_count') == 2, f'{segment} bad sequence block count')
        require(report.get('onnx_segment_count') == 1, f'{segment} bad ONNX segment count')
        require(report.get('is_single_monolithic_onnx') is False, f'{segment} must not be monolithic')
        require(report.get('is_real_transformer_block') is False, f'{segment} must not claim real transformer block')
        require(report.get('is_full_bonsai_pipeline') is False, f'{segment} must not claim full Bonsai pipeline')
        require(report.get('critical_outputs_allclose_rtol_1e_4_atol_1e_5') is True, f'{segment} critical outputs not allclose')
        require(float(report.get('critical_max_abs_error', 1.0)) <= 1e-3, f'{segment} critical max_abs_error too high')
        segment_summaries.append({
            'block_indices': list(segment),
            'report_path': report.get('_report_path'),
            'critical_max_abs_error': report.get('critical_max_abs_error'),
            'diagnostic_max_abs_error': report.get('diagnostic_max_abs_error'),
            'critical_outputs_allclose_rtol_1e_4_atol_1e_5': report.get('critical_outputs_allclose_rtol_1e_4_atol_1e_5'),
            'diagnostic_outputs_allclose_rtol_1e_4_atol_1e_4': report.get('diagnostic_outputs_allclose_rtol_1e_4_atol_1e_4'),
            'all_outputs_allclose_rtol_1e_4_atol_1e_5': report.get('all_outputs_allclose_rtol_1e_4_atol_1e_5'),
        })

    critical_max = max(float(item['critical_max_abs_error']) for item in segment_summaries)
    diagnostic_values = [item['diagnostic_max_abs_error'] for item in segment_summaries if item.get('diagnostic_max_abs_error') is not None]
    diagnostic_max = max(float(value) for value in diagnostic_values) if diagnostic_values else None
    summary = {
        'ok': True,
        'graph_kind': 'pair_segments_aggregate_single_blocks_modulated_attention_to_out_residual_stack',
        'block_indices': list(range(20)),
        'segment_block_indices': [list(segment) for segment in EXPECTED_SEGMENTS],
        'sequence_block_count': 20,
        'onnx_segment_count': 10,
        'is_single_monolithic_onnx': False,
        'is_real_transformer_block': False,
        'is_full_bonsai_pipeline': False,
        'critical_outputs_allclose_rtol_1e_4_atol_1e_5': True,
        'critical_max_abs_error': critical_max,
        'diagnostic_max_abs_error': diagnostic_max,
        'segments': segment_summaries,
        'claim': 'all_pair_segments_single_blocks_modulated_attention_to_out_residual_stack_onnxruntime_cpu_critical_paths_verified_not_single_monolithic_onnx_not_real_transformer_block_or_full_bonsai_pipeline',
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
