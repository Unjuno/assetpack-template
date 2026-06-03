#!/usr/bin/env python3
import json, time
from pathlib import Path
import torch

OUT_DIR = Path('reports/bonsai-vae-decode-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-gemlite-1bit'

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / 'vae_decode_smoke.png'
    report = {'target': 'VAE decode smoke attempt', 'repo': REPO, 'subfolder': 'vae'}
    try:
        from diffusers import AutoencoderKL
        from PIL import Image
        started = time.time()
        vae = AutoencoderKL.from_pretrained(REPO, subfolder='vae', torch_dtype=torch.float32).to('cpu')
        vae.eval()
        cfg = dict(getattr(vae, 'config', {}))
        latent_channels = int(cfg.get('latent_channels', 16))
        scaling_factor = float(cfg.get('scaling_factor', 1.0))
        g = torch.Generator(device='cpu'); g.manual_seed(250000)
        latents = torch.randn((1, latent_channels, 8, 8), generator=g) * 0.1
        with torch.inference_mode():
            decoded = vae.decode(latents / scaling_factor).sample
        img = decoded[0].detach().float().clamp(-1, 1)
        img = ((img + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(png_path)
        report.update({
            'ok': True,
            'vae_decode_success': True,
            'has_png_artifact': png_path.exists(),
            'png_size_bytes': png_path.stat().st_size,
            'latent_shape': list(latents.shape),
            'decoded_shape': list(decoded.shape),
            'latent_channels': latent_channels,
            'scaling_factor': scaling_factor,
            'seconds': round(time.time() - started, 3),
            'config_subset': {k: cfg.get(k) for k in ['latent_channels','scaling_factor','sample_size','block_out_channels','layers_per_block','act_fn'] if k in cfg},
        })
    except Exception as e:
        report.update({'ok': True, 'vae_decode_success': False, 'error': type(e).__name__ + ': ' + str(e)[:1200], 'has_png_artifact': png_path.exists()})
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': report.get('ok'), 'vae_decode_success': report.get('vae_decode_success'), 'error': report.get('error')}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
