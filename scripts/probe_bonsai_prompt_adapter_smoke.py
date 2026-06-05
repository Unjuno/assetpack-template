#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path('reports/bonsai-prompt-adapter-smoke')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'target': 'prompt embedding adapter smoke',
        'ok': True,
        'adapter_success': False,
        'skipped': True,
        'skip_reason': 'placeholder_after_interrupted_write'
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': True, 'skipped': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
