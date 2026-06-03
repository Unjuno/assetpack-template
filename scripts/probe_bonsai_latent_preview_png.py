#!/usr/bin/env python3
import json, struct, zlib
from pathlib import Path
import torch
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run, stat

OUT_DIR = Path('reports/bonsai-latent-preview-png')
STEP_SIZE = 0.05
PNG_NAME = 'latent_preview.png'

def png_chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

def write_rgb_png(path, img):
    # img: uint8 H x W x 3
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))
    data = b'\x89PNG\r\n\x1a\n'
    data += png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    data += png_chunk(b'IDAT', zlib.compress(raw, 9))
    data += png_chunk(b'IEND', b'')
    path.write_bytes(data)

def latent_to_rgb(x):
    # x: [1, N, C]. Deterministic preview, not VAE decode.
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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    g = torch.Generator(device='cpu'); g.manual_seed(200000)
    latent = torch.randn((1, 2, 128), generator=g) * 0.01
    ctx = torch.randn((1, 3, 7680), generator=g) * 0.01
    tf = torch.randn((1, 256), generator=g) * 0.01
    with torch.inference_mode():
        pred, *_ = run(sd, latent, ctx, tf, True)
        next_latent = latent - STEP_SIZE * pred
    png_path = OUT_DIR / PNG_NAME
    write_rgb_png(png_path, latent_to_rgb(next_latent))
    report = {
        'source_model_ref': LOWBIT_REF,
        'uses_lowbit_source': True,
        'writes_expanded_checkpoint': False,
        'target': 'latent preview PNG smoke from one-step wrapper prediction',
        'not_vae_decode': True,
        'has_png_artifact': png_path.exists(),
        'png_name': PNG_NAME,
        'png_size_bytes': png_path.stat().st_size,
        'step_size': STEP_SIZE,
        'latent': stat(latent),
        'prediction': stat(pred),
        'next_latent': stat(next_latent),
        'lowbit_path': str(path),
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'has_png_artifact': report['has_png_artifact'], 'png_size_bytes': report['png_size_bytes'], 'not_vae_decode': True}, indent=2))
    return 0 if report['has_png_artifact'] and report['png_size_bytes'] > 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
