#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

p = argparse.ArgumentParser()
p.add_argument('--config', default='assetpack.yml')
p.add_argument('--request-json', required=True)
p.add_argument('--out-dir', required=True)
a = p.parse_args()

cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
req_path = Path(a.request_json)
req = json.loads(req_path.read_text(encoding='utf-8'))
need = [str(x).strip() for x in cfg.get('prompt_policy', {}).get('required_terms', []) if str(x).strip()]
miss = [x for x in need if x not in str(req.get('prompt', ''))]
req['required_terms'] = need
req['missing_terms'] = miss
req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if not miss:
    subprocess.run([
        sys.executable,
        'scripts/run_issue_asset_generation.py',
        '--config', a.config,
        '--request-json', a.request_json,
        '--out-dir', a.out_dir,
    ])
else:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'not_generated', 'summary': {'passed': 0, 'failed': 1}, 'reason': 'missing required terms'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

raise SystemExit(0)
