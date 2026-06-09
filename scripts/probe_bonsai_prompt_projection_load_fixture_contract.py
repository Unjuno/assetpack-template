#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

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

    report_path = OUT / 'report.json'
    data = json.loads(report_path.read_text())
    data['fixture_generated'] = True
    data['fixture_path'] = str(FIXTURE)
    data['fixture_size_bytes'] = FIXTURE.stat().st_size
    data['expected_positive_path'] = True
    data['ok'] = (
        data.get('ok') is True
        and data.get('load_available') is True
        and data.get('load_attempted') is True
        and data.get('projection_wired') is False
        and data.get('used_in_generation_path') is False
    )
    if not data['ok']:
        data['error'] = 'synthetic projection load positive path failed'
    report_path.write_text(json.dumps(data, sort_keys=True) + '\n')
    if not data['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()
