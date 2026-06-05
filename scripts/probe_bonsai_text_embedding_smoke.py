#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch

OUT_DIR = Path('reports/bonsai-text-embedding-smoke')
PROMPTS = ['a tiny bonsai tree in a ceramic pot', '赤い鉢に入った小さな盆栽']
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'


def stat(x):
    y = x.detach().float()
    return {
        'shape': list(x.shape),
        'dtype': str(x.dtype),
        'finite': bool(torch.isfinite(y).all