#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='assetpack.yml')
    p.add_argument('--request-json', required=True)
    p.add_argument('--comment-file', required=True)
    p.add_argument('--github-output', default='')
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    req_path = Path(a.request_json)
    req = json.loads(req_path.read_text(encoding='utf-8'))
    need = [str(x).strip() for x in cfg.get('prompt_policy', {}).get('required_terms', []) if str(x).strip()]
    text = str(req.get('prompt', ''))
    miss = [x for x in need if x not in text]
    req['required_terms'] = need
    req['missing_terms'] = miss
    req['valid'] = bool(req.get('valid')) and not miss

    if miss:
        req['errors'] = list(req.get('errors', [])) + [f'missing required term: {x}' for x in miss]
        body = '## Asset request not generated\n\nMissing required terms:\n' + ''.join(f'\n- `{x}`' for x in miss) + '\n'
        Path(a.comment_file).write_text(body, encoding='utf-8')

    req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if a.github_output:
        with Path(a.github_output).open('a', encoding='utf-8') as f:
            f.write(f"valid={'true' if req['valid'] else 'false'}\n")
            f.write(f"selected_model_id={req.get('selected_model_id', '')}\n")
            f.write(f"recipe_id={req.get('recipe_id', '')}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
