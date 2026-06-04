#!/usr/bin/env python3
import json, os, time
from pathlib import Path

OUT_DIR = Path('reports/bonsai-tokenizer-smoke')
PROMPTS = ['a tiny bonsai tree in a ceramic pot', '赤い鉢に入った小さな盆栽']
CANDIDATES = [
    ('prism-ml/bonsai-image-binary-4B-gemlite-1bit', 'text_encoder-hqq-4bit/tokenizer'),
    ('prism-ml/bonsai-image-binary-4B-unpacked', 'tokenizer'),
]

def is_rate_limit_error(text):
    lowered = text.lower()
    return '429' in lowered or 'too many requests' in lowered or 'ratelimit' in lowered or 'rate limit' in lowered

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'target': 'tokenizer smoke', 'prompts': PROMPTS, 'candidates': []}
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        report.update({'ok': True, 'tokenizer_success': False, 'blocked_by_dependency': True, 'error': type(e).__name__ + ': ' + str(e)[:800]})
        (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'ok': True, 'tokenizer_success': False, 'error': report['error']}, indent=2)); return 0
    started = time.time()
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    for repo, subfolder in CANDIDATES:
        item = {'repo': repo, 'subfolder': subfolder}
        try:
            tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, trust_remote_code=True, token=token)
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
            err = type(e).__name__ + ': ' + str(e)[:1000]
            item.update({'ok': False, 'error': err, 'blocked_by_hf_429': is_rate_limit_error(err)})
        report['candidates'].append(item)
    ok_items = [x for x in report['candidates'] if x.get('ok')]
    blocked_429 = any(x.get('blocked_by_hf_429') for x in report['candidates']) and not ok_items
    report.update({
        'ok': True,
        'tokenizer_success': bool(ok_items),
        'success_count': len(ok_items),
        'blocked_by_hf_429': blocked_429,
        'used_hf_token_env': bool(token),
        'seconds': round(time.time() - started, 3),
    })
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'ok': report['ok'], 'tokenizer_success': report['tokenizer_success'], 'success_count': report['success_count'], 'blocked_by_hf_429': report['blocked_by_hf_429']}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
