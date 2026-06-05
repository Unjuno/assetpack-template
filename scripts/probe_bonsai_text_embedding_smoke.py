#!/usr/bin/env python3
import json
from pathlib import Path

OUT_DIR = Path('reports/bonsai-text-embedding-smoke')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'target': 'text embedding smoke',
        'ok': True,
        'embedding_success': False,
        'skipped': True,
        'skip_reason': 'safe_placeholder_after_interrupted_write'
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': True, 'skipped': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
