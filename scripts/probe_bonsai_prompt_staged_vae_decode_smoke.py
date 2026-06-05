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


def tensor