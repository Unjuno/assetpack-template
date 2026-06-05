#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch

OUT_DIR = Path('reports/bonsai-text-embedding-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'
PROMPTS = ['a tiny bonsai tree in a ceramic pot', 'red bonsai pot']

def stat_shape(x):
    y = x.detach().float()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': bool(torch.isfinite(y).all())}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    report = {'target': 'text embedding smoke with adapter', 'repo': REPO}
    t0 = time.time()
    try:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(REPO, subfolder='