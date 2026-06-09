#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch

from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM

DEFAULT_OUT = Path('reports/bonsai-prompt-projection-load-contract')
DEFAULT_PATH = Path('artifacts/bonsai_prompt_projection.pt')


def report_dir():
    return Path(os.environ.get('BONSAI_PROMPT_PROJECTION_REPORT_DIR', str(DEFAULT_OUT)))


def write_report(data):
    out = report_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(data, sort_keys=True) + '\n')


def tensor_shape(obj):
    if hasattr(obj, 'shape'):
        return list(obj.shape)
    return None


def main():
    path = Path(os.environ.get('BONSAI_PROMPT_PROJECTION_PATH', str(DEFAULT_PATH)))
    expected = {
        'weight': [BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM],
        'bias': [BONSAI_CONTEXT_DIM],
    }

    base = {
        'adapter_kind': 'linear_projection',
        'expected_shapes': expected,
        'load_path': str(path),
        'projection_wired': False,
        'used_in_generation_path': False,
    }

    if not path.is_file():
        write_report({
            **base,
            'ok': True,
            'load_available': False,
            'load_attempted': False,
            'expected_missing': True,
            'error': 'projection artifact not found',
        })
        return

    try:
        obj = torch.load(path, map_location='cpu')
    except Exception as exc:
        write_report({
            **base,
            'ok': False,
            'load_available': True,
            'load_attempted': True,
            'error': f'load failed: {type(exc).__name__}: {exc}',
        })
        return

    state = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        write_report({
            **base,
            'ok': False,
            'load_available': True,
            'load_attempted': True,
            'error': f'loaded object is not a state dict: {type(state).__name__}',
        })
        return

    shapes = {key: tensor_shape(state.get(key)) for key in expected}
    missing = [key for key in expected if key not in state]
    bad_shapes = {
        key: {'expected': expected[key], 'actual': shapes.get(key)}
        for key in expected
        if shapes.get(key) != expected[key]
    }
    ok = not missing and not bad_shapes
    write_report({
        **base,
        'ok': ok,
        'load_available': True,
        'load_attempted': True,
        'keys_checked': sorted(expected),
        'missing_keys': missing,
        'shapes': shapes,
        'bad_shapes': bad_shapes,
        'error': None if ok else 'projection state dict contract mismatch',
    })


if __name__ == '__main__':
    main()
