#!/usr/bin/env python3
import json
from pathlib import Path

OUT_DIR = Path('reports/bonsai-prompt-staged-vae-decode-smoke')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'target': 'prompt embedding to staged VAE decode smoke',
        'ok': True,
        'skipped': True,
        'skip_reason': 'disabled_until_prompt_conditioning_adapter_is_split_into_safe_small_scripts',
        'not_full_prompt_pipeline': True,
    }
    (OUT_DIR / 'report.json').