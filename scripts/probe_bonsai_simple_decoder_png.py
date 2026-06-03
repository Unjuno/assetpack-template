#!/usr/bin/env python3
import json, struct, zlib
from pathlib import Path
import torch
import torch.nn.functional as F
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run, stat

OUT_DIR = Path('reports/bonsai-simple-decoder-png')
STEP_SIZE = 0.05
STEP_COUNT = 4
PNG_NAME = 'simple_decoder_preview.png'

def png_chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

def write_rgb_png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))
    data = b'\x89PNG\r\n\x1a\n'
    data += png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    data += png_chunk(b'IDAT', zlib.compress(raw, 9))
    data += png_chunk(b'IEND', b'')
    path.write_bytes(data)

def fixed_decoder(latent):
    # Deterministic lightweight decoder smoke, not VAE.
    x = latent.detach().float()[0]
    seed = torch.Generator(device='cpu'); seed.manual_seed(210000)
    w1 = torch.randn((128, 64), generator=seed) / 8.0
    w2 = torch.randn((64, 3), generator=seed) / 8.0
    y = torch.tanh(x @ w1)
    rgb_tokens = torch.sigmoid(y @ w2)
    side = 64
    n = rgb_tokens.shape[0]
    img = rgb_tokens.repeat_interleave((side * side + n - 1) // n, dim=0)[:side * side]
    return (img.reshape(side, side, 3) * 255).clamp(0, 255).byte().cpu().numpy()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    g = torch.Generator(device='cpu'); g.manual_seed(200000)
    latent0 = torch.randn((1, 2, 128), generator=g) * 0.01
    ctx = torch.randn((1, 3, 7680), generator=g) * 0.01
    tf = torch.randn((1, 256), generator=g) * 0.01
    latent = latent0.clone()
    steps = []
    with torch.inference_mode():
        for i in range(STEP_COUNT):
            pred, *_ = run(sd, latent, ctx, tf, True)
            next_latent = latent - STEP_SIZE * pred
            steps.append({'index': i, 'latent_after': stat(next_latent), 'prediction': stat(pred), 'all_finite': bool(torch.isfinite(next_latent).all() and torch.isfinite(pred).all())})
            latent = next_latent
    png_path = OUT_DIR / PNG_NAME
    write_rgb_png(png_path, fixed_decoder(latent))
    report = {
        'source_model_ref': LOWBIT_REF,
        'uses_lowbit_source': True,
        'writes_expanded_checkpoint': False,
        'target': 'simple deterministic decoder PNG smoke from 4-step wrapper latent',
        'not_vae_decode': True,
        'decoder': 'fixed_random_mlp_128_64_3_seed_210000',
        'step_count': STEP_COUNT,
        'step_size': STEP_SIZE,
        'all_steps_finite': all(s['all_finite'] for s in steps),
        'has_png_artifact': png_path.exists(),
        'png_name': PNG_NAME,
        'png_size_bytes': png_path.stat().st_size,
        'initial_latent': stat(latent0),
        'final_latent': stat(latent),
        'steps': steps,
        'lowbit_path': str(path),
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'has_png_artifact': report['has_png_artifact'], 'png_size_bytes': report['png_size_bytes'], 'step_count': STEP_COUNT, 'all_steps_finite': report['all_steps_finite'], 'not_vae_decode': True}, indent=2))
    return 0 if report['has_png_artifact'] and report['png_size_bytes'] > 0 and report['all_steps_finite'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
