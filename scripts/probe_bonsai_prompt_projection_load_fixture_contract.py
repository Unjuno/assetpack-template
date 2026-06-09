#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from bonsai_prompt_projection import project_prompt_hidden
from bonsai_text_consts import BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM

OUT = Path('reports/bonsai-prompt-projection-load-fixture-contract')
FIXTURE = Path('artifacts/bonsai_prompt_projection_fixture.pt')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'weight': torch.zeros((BONSAI_CONTEXT_DIM, TEXT_HIDDEN_DIM), dtype=torch.float16),
        'bias': torch.zeros((BONSAI_CONTEXT_DIM,), dtype=torch.float16),
    }
    torch.save({'state_dict': state, 'fixture': True}, FIXTURE)

    env = dict(os.environ)
    env['BONSAI_PROMPT_PROJECTION_PATH'] = str(FIXTURE)
    env['BONSAI_PROMPT_PROJECTION_REPORT_DIR'] = str(OUT)
    proc = subprocess.run(
        [sys.executable, 'scripts/probe_bonsai_prompt_projection_load_contract.py'],
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        (OUT / 'report.json').write_text(json.dumps({
            'ok': False,
            'error': f'load contract probe exited {proc.returncode}',
            'fixture_path': str(FIXTURE),
        }, sort_keys=True) + '\n')
        sys.exit(proc.returncode)

    old_path = os.environ.get('BONSAI_PROMPT_PROJECTION_PATH')
    os.environ['BONSAI_PROMPT_PROJECTION_PATH'] = str(FIXTURE)
    try:
        projected = project_prompt_hidden(torch.ones(2, 3, TEXT_HIDDEN_DIM))
    finally:
        if old_path is None:
            os.environ.pop('BONSAI_PROMPT_PROJECTION_PATH', None)
        else:
            os.environ['BONSAI_PROMPT_PROJECTION_PATH'] = old_path
    projected_shape = list(projected.shape)
    projected_finite = bool(torch.isfinite(projected).all())
    projected_zero = bool(torch.all(projected == 0).item())

    report_path = OUT / 'report.json'
    data = json.loads(report_path.read_text())
    data['fixture_generated'] = True
    data['fixture_path'] = str(FIXTURE)
    data['fixture_size_bytes'] = FIXTURE.stat().st_size
    data['expected_positive_path'] = True
    data['project_prompt_hidden_checked'] = True
    data['project_prompt_hidden_shape'] = projected_shape
    data['project_prompt_hidden_finite'] = projected_finite
    data['project_prompt_hidden_zero_output'] = projected_zero
    data['ok'] = (
        data.get('ok') is True
        and data.get('load_available') is True
        and data.get('load_attempted') is True
        and data.get('projection_wired') is False
        and data.get('used_in_generation_path') is False
        and projected_shape == [2, 3, BONSAI_CONTEXT_DIM]
        and projected_finite
        and projected_zero
    )
    if not data['ok']:
        data['error'] = 'synthetic projection load positive path failed'
    report_path.write_text(json.dumps(data, sort_keys=True) + '\n')
    if not data['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()
