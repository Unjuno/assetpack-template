#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch

OUT=Path('reports/bonsai-text-embedding-smoke')
REPO='prism-ml/bonsai-image-binary-4B-unpacked'
PROMPTS=['a tiny bonsai tree in a ceramic pot','赤い鉢に入った小さな盆栽']

def st(x):
    y=x.detach().float()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(torch.isfinite(y).all()),'mean_abs':float(y.abs().mean()),'max_abs':float(y.abs().max())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.time(); token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING