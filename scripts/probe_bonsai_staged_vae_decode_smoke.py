#!/usr/bin/env python3
import json, math, time
from pathlib import Path
import torch

from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_staged_generation_smoke import run_stage2_full_rope, STEP_COUNT, STEP_SIZE, stat

OUT_DIR = Path('reports/bonsai-staged-vae-decode-smoke')
REPO = 'prism-ml/bonsai-image-binary-4B-gemlite-1bit'
TOKEN_GRID = 4
PATCH_SIZE = 2

def unpack_tokens_to_vae_latent(tokens, latent_channels, token_grid, patch_size):
    b, n, d = tokens.shape
    expected_d = latent_channels * patch_size * patch_size
    if n != token_grid * token_grid:
        raise ValueError(f'token count {n} does not match token_grid^2 {token_grid * token_grid}')
    if d != expected_d:
        raise ValueError(f'token dim {d} does not match latent_channels*patch_size^2 {expected_d}')
    x = tokens.detach().float().reshape(b, token_grid, token_grid, patch_size, patch_size, latent_channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.reshape(b, latent_channels, token_grid * patch_size, token_grid * patch_size)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / 'staged_vae_decode_smoke.png'
    report = {
        'target': 'staged transformer packed-token output to VAE decode smoke',
        'source_model_ref': LOWBIT_REF,
        'vae_repo': REPO,
        'subfolder': 'vae',
        'not_full_prompt_pipeline': True,
        'token_grid': TOKEN_GRID,
        'patch_size': PATCH_SIZE,
    }
    try:
        from diffusers import AutoencoderKL
        from PIL import Image
        started = time.time()
        path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
        vae = AutoencoderKL.from_pretrained(REPO, subfolder='vae', torch_dtype=torch.float32).to('cpu')
        vae.eval()
        cfg = dict(getattr(vae, 'config', {}))
        latent_channels = int(cfg.get('latent_channels', 32))
        scaling_factor = float(cfg.get('scaling_factor', 1.0))
        g = torch.Generator(device='cpu'); g.manual_seed(220000)
        latent = torch.randn((1, TOKEN_GRID * TOKEN_GRID, latent_channels * PATCH_SIZE * PATCH_SIZE), generator=g) * 0.01
        ctx = torch.randn((1, 3, 7680), generator=g) * 0.01
        tf = torch.randn((1, 256), generator=g) * 0.01
        steps = []
        with torch.inference_mode():
            for i in range(STEP_COUNT):
                pred = run_stage2_full_rope(sd, latent, ctx, tf)
                next_latent = latent - STEP_SIZE * pred
                steps.append({'index': i, 'prediction': stat(pred), 'latent_after': stat(next_latent), 'all_finite': bool(torch.isfinite(pred).all() and torch.isfinite(next_latent).all())})
                latent = next_latent
            vae_latent = unpack_tokens_to_vae_latent(latent, latent_channels, TOKEN_GRID, PATCH_SIZE)
            decoded = vae.decode(vae_latent / scaling_factor).sample
        img = decoded[0].detach().float().clamp(-1, 1)
        img = ((img + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(png_path)
        report.update({
            'ok': True,
            'decode_success': True,
            'has_png_artifact': png_path.exists(),
            'png_size_bytes': png_path.stat().st_size,
            'stage_step_count': STEP_COUNT,
            'stage_all_steps_finite': all(s['all_finite'] for s in steps),
            'token_latent_shape': list(latent.shape),
            'vae_latent_shape': list(vae_latent.shape),
            'decoded_shape': list(decoded.shape),
            'latent_channels': latent_channels,
            'scaling_factor': scaling_factor,
            'seconds': round(time.time() - started, 3),
            'projection': 'native_packed_token_unpatchify_no_random_projection',
            'packing_identity': f'token_dim={latent_channels}*{PATCH_SIZE}*{PATCH_SIZE}',
            'steps': steps,
        })
    except Exception as e:
        report.update({'ok': True, 'decode_success': False, 'error': type(e).__name__ + ': ' + str(e)[:1200], 'has_png_artifact': png_path.exists()})
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': report.get('ok'), 'decode_success': report.get('decode_success'), 'has_png_artifact': report.get('has_png_artifact'), 'projection': report.get('projection'), 'error': report.get('error')}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
