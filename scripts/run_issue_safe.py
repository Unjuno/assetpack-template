#!/usr/bin/env python3
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--config', default='assetpack.yml')
p.add_argument('--request-json', required=True)
p.add_argument('--out-dir', required=True)
a = p.parse_args()

subprocess.run([
    sys.executable,
    'scripts/run_issue_asset_generation.py',
    '--config', a.config,
    '--request-json', a.request_json,
    '--out-dir', a.out_dir,
])
raise SystemExit(0)
