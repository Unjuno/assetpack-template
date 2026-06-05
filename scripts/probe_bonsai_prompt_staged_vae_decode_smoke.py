#!/usr/bin/env python3
import json, os, time
from pathlib import Path
import torch
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_staged_generation_smoke import run_stage2_full_rope, STEP_COUNT, STEP_SIZE, stat
from probe_bonsai_staged_vae_decode_smoke import unpack_tokens_to_vae_latent, TOKEN_GRID, PATCH_SIZE

OUT_DIR = Path('reports/bonsai-prompt-staged-vae-decode-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-unpacked'
VAE_REPO = 'prism-ml/bonsai-image-binary-4B-gemlite-1bit'
PROMPT = 'a tiny bonsai tree in a ceramic pot'

def tstat(x):
    y = x.detach().float()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': bool(torch.isfinite(y).all()), 'mean_abs': float(y.abs().mean()), 'max_abs': float(y.abs().max())}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / 'prompt_staged_vae_decode_smoke.png'
    report = {'target': 'prompt embedding to staged VAE decode smoke', 'prompt': PROMPT, 'repo': REPO, 'vae_repo': VAE_REPO, 'adapter': 'repeat_text_hidden_2560_to_7680_x3', 'not_full_prompt_pipeline': True}
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    try:
        from transformers