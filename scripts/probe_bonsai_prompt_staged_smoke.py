#!/usr/bin/env python3
import json
from pathlib import Path
from bonsai_prompt_staged_core import run_prompt_staged_once
p = Path('reports/bonsai-prompt-staged-smoke')
p.mkdir(parents=True, exist_ok=True)
r = {'target': 'prompt context staged transformer smoke', 'ok': True}
r.update(run_prompt_staged_once())
(p / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({'ok': True, 'ctx': r['ctx']['shape'], 'pred': r['pred']['shape']}))
