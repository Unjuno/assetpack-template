#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch

OUT = Path('reports/bonsai-prompt-adapter-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'
PROMPTS = ['a tiny bonsai tree in a ceramic pot', '赤い鉢に入った小さな盆栽']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = {'target': 'prompt embedding adapter smoke', 'repo': REPO, 'prompts': PROMPTS}
    t0 = time.time()
    try:
        from transformers import AutoModel, Auto