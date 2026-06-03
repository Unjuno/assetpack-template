#!/usr/bin/env python3
import json, struct, time, zlib
from pathlib import Path
import torch
import torch.nn.functional as F
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run as run_wrapper, stat
from probe_bonsai_lowbit_modulated_block_cores import stack as run_modulated_stack

OUT_DIR = Path('reports/bonsai-staged-generation-smoke')
STEP_SIZE = 0.05
STEP_COUNT = 4

def png_chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

def write_png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))
    data = b'\x89PNG\r\n\x1a\n'
    data += png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    data += png_chunk(b'IDAT', zlib.compress(raw, 9))
    data += png_chunk(b'IEND', b'')
    path.write_bytes(data)

def decoder(latent, seed_value):
    x = latent.detach().float()[0]
    g = torch.Generator(device='cpu'); g.manual_seed(seed_value)
    w1 = torch.randn((128, 64), generator=g) / 8.0
    w2 = torch.randn((64, 3), generator=g) / 8.0
    rgb = torch.sigmoid(torch.tanh(x @ w1) @ w2)
    side = 64
    n = rgb.shape[0]
    img = rgb.repeat_interleave((side * side + n - 1) // n, dim=0)[:side * side]
    return (img.reshape(side, side, 3) * 255).clamp(0, 255).byte().cpu().numpy()

def lin(sd, key, x):
    return F.linear(x, sd[key].to(dtype=x.dtype, device=x.device))

def tmlp(sd, x):
    a = lin(sd, 'time_guidance_embed.timestep_embedder.linear_1.weight', x)
    return lin(sd, 'time_guidance_embed.timestep_embedder.linear_2.weight', F.silu(a))

def final(sd, img, temb):
    shift, scale = lin(sd, 'norm_out.linear.weight', temb).chunk(2, dim=-1)
    y = F.layer_norm(img.float(), (img.shape[-1],), eps=1e-6)
    y = y * (1 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)
    return lin(sd, 'proj_out.weight', y.to(img.dtype))

def run_stage1(sd, latent, ctx, tf):
    pred, *_ = run_wrapper(sd, latent, ctx, tf, True)
    return pred

def run_stage2(sd, latent, ctx, tf):
    img = lin(sd, 'x_embedder.weight', latent)
    txt = lin(sd, 'context_embedder.weight', ctx)
    temb = tmlp(sd, tf)
    img_out, _, _ = run_modulated_stack(sd, img, txt, temb, True)
    return final(sd, img_out, temb)

def execute_stage(sd, name, runner, seed_value):
    g = torch.Generator(device='cpu'); g.manual_seed(220000)
    latent0 = torch.randn((1, 2, 128), generator=g) * 0.01
    ctx = torch.randn((1, 3, 7680), generator=g) * 0.01
    tf = torch.randn((1, 256), generator=g) * 0.01
    latent = latent0.clone()
    steps = []
    started = time.time()
    with torch.inference_mode():
        for i in range(STEP_COUNT):
            pred = runner(sd, latent, ctx, tf)
            next_latent = latent - STEP_SIZE * pred
            delta = next_latent - latent
            steps.append({'index': i, 'prediction': stat(pred), 'latent_after': stat(next_latent), 'delta_abs_mean': float(delta.abs().mean()), 'delta_abs_max': float(delta.abs().max()), 'all_finite': bool(torch.isfinite(pred).all() and torch.isfinite(next_latent).all())})
            latent = next_latent
    png_name = f'{name}.png'
    png_path = OUT_DIR / png_name
    write_png(png_path, decoder(latent, seed_value))
    total_delta = latent - latent0
    return {
        'name': name,
        'step_count': STEP_COUNT,
        'step_size': STEP_SIZE,
        'seconds': round(time.time() - started, 3),
        'all_steps_finite': all(s['all_finite'] for s in steps),
        'has_png_artifact': png_path.exists(),
        'png_name': png_name,
        'png_size_bytes': png_path.stat().st_size,
        'initial_latent': stat(latent0),
        'final_latent': stat(latent),
        'total_delta_abs_mean': float(total_delta.abs().mean()),
        'total_delta_abs_max': float(total_delta.abs().max()),
        'steps': steps,
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    stage1 = execute_stage(sd, 'stage1_wrapper_rope_1double', run_stage1, 230001)
    stage2 = execute_stage(sd, 'stage2_modulated_stack_5double_20single', run_stage2, 230002)
    report = {
        'source_model_ref': LOWBIT_REF,
        'uses_lowbit_source': True,
        'writes_expanded_checkpoint': False,
        'target': 'batched staged generation smoke with logs and PNG artifacts',
        'not_vae_decode': True,
        'stages': [
            {**stage1, 'description': '1 double block RoPE wrapper path'},
            {**stage2, 'description': '5 double + 20 single modulated stack path; no RoPE in full stack yet'},
        ],
        'all_stages_passed': all(s['all_steps_finite'] and s['has_png_artifact'] and s['png_size_bytes'] > 0 for s in [stage1, stage2]),
        'lowbit_path': str(path),
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_stages_passed': report['all_stages_passed'], 'stage_count': len(report['stages']), 'pngs': [s['png_name'] for s in report['stages']]}, indent=2))
    return 0 if report['all_stages_passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
