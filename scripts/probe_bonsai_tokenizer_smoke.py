#!/usr/bin/env python3
import json, time
from pathlib import Path

OUT_DIR = Path('reports/bonsai-tokenizer-smoke')
PROMPTS = ['a tiny bonsai tree in a ceramic pot', '赤い鉢に入った小さな盆栽']
CANDIDATES = [
    ('prism-ml/bonsai-image-binary-4B-gemlite-1bit', 'text_encoder-hqq-4bit/tokenizer'),
    ('prism-ml/bonsai-image-binary-4B-unpacked', 'tokenizer'),
]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'target': 'tokenizer smoke', 'prompts': PROMPTS, 'candidates': []}
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        report.update({'ok': False, 'error': type(e).__name__ + ': ' + str(e)[:800]})
        (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'ok': False, 'error': report['error']}, indent=2)); return 1
    started = time.time()
    for repo, subfolder in CANDIDATES:
        item = {'repo': repo, 'subfolder': subfolder}
        try:
            tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, trust_remote_code=True)
            enc = tok(PROMPTS, padding=True, truncation=True, return_tensors='pt')
            ids = enc['input_ids']
            item.update({
                'ok': True,
                'tokenizer_class': tok.__class__.__name__,
                'vocab_size': getattr(tok, 'vocab_size', None),
                'model_max_length': getattr(tok, 'model_max_length', None),
                'input_ids_shape': list(ids.shape),
                'attention_mask_shape': list(enc['attention_mask'].shape) if 'attention_mask' in enc else None,
                'first_prompt_token_count': int((enc['attention_mask'][0].sum() if 'attention_mask' in enc else ids[0].numel()).item()),
                'decoded_first_prefix': tok.decode(ids[0][: min(16, ids.shape[1])].tolist())[:200],
            })
        except Exception as e:
            item.update({'ok': False, 'error': type(e).__name__ + ': ' + str(e)[:1000]})
        report['candidates'].append(item)
    ok_items = [x for x in report['candidates'] if x.get('ok')]
    report.update({'ok': bool(ok_items), 'success_count': len(ok_items), 'seconds': round(time.time() - started, 3)})
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'ok': report['ok'], 'success_count': report['success_count']}, indent=2))
    return 0 if report['ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
