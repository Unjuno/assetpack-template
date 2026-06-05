#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch

OUT_DIR = Path('reports/bonsai-text-embedding-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'
PROMPTS = ['a tiny bonsai tree in a ceramic pot', 'red bonsai pot']

def tstat(x):
    y = x.detach().float()
    return {
        'shape': list(x.shape),
        'dtype': str(x.dtype),
        'finite': bool(torch.isfinite(y).all()),
        'mean_abs': float(y.abs().mean()),
        'max_abs': float(y.abs().max()),
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    report = {'target': 'text embedding smoke with 2560_to_7680_adapter', 'repo': REPO}
    started = time.time()
    try:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, subfolder='tokenizer', trust_remote_code=True, token=token)
        enc = tok(PROMPTS, padding=True, truncation=True, return_tensors='pt')
        model = AutoModel.from_pretrained(REPO, subfolder='text_encoder', trust_remote_code=True, torch_dtype=torch.float32, token=token).to('cpu')
        model.eval()
        with torch.inference_mode():
            out = model(**enc)
        hidden = out.last_hidden_state
        ctx = torch.cat([hidden, hidden, hidden], dim=-1)
        report.update({
            'ok': True,
            'embedding