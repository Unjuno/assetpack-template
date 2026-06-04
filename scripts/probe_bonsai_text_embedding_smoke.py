#!/usr/bin/env python3
import json, os, time
from pathlib import Path

import torch

OUT_DIR = Path('reports/bonsai-text-embedding-smoke')
PROMPTS = ['a tiny bonsai tree in a ceramic pot', '赤い鉢に入った小さな盆栽']
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'
TOKENIZER_SUBFOLDER = 'tokenizer'
TEXT_ENCODER_SUBFOLDER = 'text_encoder'

def is_rate_limit_error(text):
    s = text.lower()
    return '429' in s or 'too many requests' in s or 'rate limit' in s or 'ratelimit' in s

def stat_tensor(x):
    y = x.detach().float()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': bool(torch.isfinite(y).all()), 'mean_abs': float(y.abs().mean()), 'max_abs': float(y.abs().max())}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'target': 'text embedding smoke', 'repo': REPO, 'tokenizer_subfolder': TOKENIZER_SUBFOLDER, 'text_encoder_subfolder': TEXT_ENCODER_SUBFOLDER, 'prompts': PROMPTS}
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    started = time.time()
    try:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, subfolder=TOKENIZER_SUBFOLDER, trust_remote_code=True, token=token)
        enc = tok(PROMPTS, padding=True, truncation=True, return_tensors='pt')
        model = AutoModel.from_pretrained(REPO, subfolder=TEXT_ENCODER_SUBFOLDER, trust_remote_code=True, torch_dtype=torch.float32, token=token).to('cpu')
        model.eval()
        with torch.inference_mode():
            out = model(**enc)
        hidden = getattr(out, 'last_hidden_state', None)
        pooled = getattr(out, 'pooler_output', None)
        report.update({
            'ok': True,
            'embedding_success': hidden is not None,
            'model_class': model.__class__.__name__,
            'tokenizer_class': tok.__class__.__name__,
            'input_ids_shape': list(enc['input_ids'].shape),
            'attention_mask_shape': list(enc['attention_mask'].shape) if 'attention_mask' in enc else None,
            'last_hidden_state': stat_tensor(hidden) if hidden is not None else None,
            'pooler_output': stat_tensor(pooled) if pooled is not None else None,
            'seconds': round(time.time() - started, 3),
            'used_hf_token_env': bool(token),
        })
    except Exception as e:
        err = type(e).__name__ + ': ' + str(e)[:1200]
        report.update({'ok': True, 'embedding_success': False, 'error': err, 'blocked_by_hf_429': is_rate_limit_error(err), 'used_hf_token_env': bool(token), 'seconds': round(time.time() - started, 3)})
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'ok': report.get('ok'), 'embedding_success': report.get('embedding_success'), 'blocked_by_hf_429': report.get('blocked_by_hf_429'), 'error': report.get('error')}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
