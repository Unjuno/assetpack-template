#!/usr/bin/env python3
import json, os, struct, zlib
from pathlib import Path
import torch
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run, stat

OUT_DIR = Path('reports/bonsai-latent-preview-png')
STEP_SIZE = float(os.getenv('BONSAI_PREVIEW_STEP_SIZE', '0.05'))
STEP_COUNT = int(os.getenv('BONSAI_PREVIEW_STEPS', '4'))
PNG_NAME = 'latent_preview.png'

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

def latent_to_rgb(x):
    y = x.detach().float()[0]
    n, c = y.shape
    side = 64
    base = y[:, :3] if c >= 3 else y.repeat(1, 3)[:, :3]
    base = base.repeat_interleave((side * side + n - 1) // n, dim=0)[:side * side]
    base = base.reshape(side, side, 3)
    lo = torch.quantile(base, 0.01)
    hi = torch.quantile(base, 0.99)
    base = ((base - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return (base * 255).byte().cpu().numpy()

def finite(x):
    return bool(torch.isfinite(x).all())

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
            delta = next_latent - latent
            steps.append({
                'index': i,
                'prediction': stat(pred),
                'latent_before': stat(latent),
                'latent_after': stat(next_latent),
                'delta_abs_mean': float(delta.abs().mean()),
                'delta_abs_max': float(delta.abs().max()),
                'all_finite': finite(pred) and finite(next_latent),
            })
            latent = next_latent
    png_path = OUT_DIR / PNG_NAME
    write_rgb_png(png_path, latent_to_rgb(latent))
    total_delta = latent - latent0
    report = {
        'source_model_ref': LOWBIT_REF,
        'uses_lowbit_source': True,
        'writes_expanded_checkpoint': False,
        'target': 'multi-step latent preview PNG smoke from wrapper prediction',
        'not_vae_decode': True,
        'has_png_artifact': png_path.exists(),
        'png_name': PNG_NAME,
        'png_size_bytes': png_path.stat().st_size,
        'step_size': STEP_SIZE,
        'step_count': STEP_COUNT,
        'all_steps_finite': all(s['all_finite'] for s in steps),
        'initial_latent': stat(latent0),
        'final_latent': stat(latent),
        'total_delta_abs_mean': float(total_delta.abs().mean()),
        'total_delta_abs_max': float(total_delta.abs().max()),
        'steps': steps,
        'lowbit_path': str(path),
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'has_png_artifact': report['has_png_artifact'], 'png_size_bytes': report['png_size_bytes'], 'step_count': STEP_COUNT, 'all_steps_finite': report['all_steps_finite'], 'not_vae_decode': True}, indent=2))
    return 0 if report['has_png_artifact'] and report['png_size_bytes'] > 0 and report['all_steps_finite'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
