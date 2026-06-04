#!/usr/bin/env python3
import json
from pathlib import Path
from huggingface_hub import list_repo_files

OUT_DIR = Path('reports/bonsai-text-encoder-inventory')
REPOS = ['prism-ml/bonsai-image-binary-4B-gemlite-1bit','prism-ml/bonsai-image-binary-4B-unpacked']
KEYS = ('text_encoder', 'tokenizer', 'qmodel', 'model.safetensors', 'model-')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'target': 'text encoder and tokenizer inventory', 'repos': []}
    for repo in REPOS:
        item = {'repo': repo}
        try:
            files = list_repo_files(repo_id=repo, repo_type='model')
            item['file_count'] = len(files)
            item['text_encoder_like_files'] = [f for f in files if any(k in f.lower() for k in KEYS)]
            item['tokenizer_files'] = [f for f in files if 'tokenizer' in f.lower()]
            item['has_text_encoder_like_files'] = bool(item['text_encoder_like_files'])
            item['has_tokenizer_files'] = bool(item['tokenizer_files'])
        except Exception as e:
            item['error'] = type(e).__name__ + ': ' + str(e)[:800]
            item['has_text_encoder_like_files'] = False
            item['has_tokenizer_files'] = False
        report['repos'].append(item)
    report['has_any_text_encoder_like_files'] = any(x.get('has_text_encoder_like_files') for x in report['repos'])
    report['has_any_tokenizer_files'] = any(x.get('has_tokenizer_files') for x in report['repos'])
    report['ok'] = True
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': report['ok'], 'has_any_text_encoder_like_files': report['has_any_text_encoder_like_files'], 'has_any_tokenizer_files': report['has_any_tokenizer_files']}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
