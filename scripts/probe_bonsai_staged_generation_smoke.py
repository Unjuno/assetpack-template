#!/usr/bin/env python3
import json, struct, time, zlib
from pathlib import Path
import torch
import torch.nn.functional as F
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run as run_wrapper, stat
from probe_bonsai_lowbit_rope_smoke import rope
from probe_bonsai_lowbit_modulated_block_cores import explicit_attention, layer_norm_modulated, lb, modulation_chunks, rms_norm_per_head, swiglu

OUT_DIR = Path('reports/bonsai-staged-generation-smoke')
STEP_SIZE = 0.05
STEP_COUNT = 4
DOUBLE_BLOCKS = 5
SINGLE_BLOCKS = 20
ROPE_FREQUENCY_SOURCE = 'diffusers.get_1d_rotary_pos_embed(pos * 0.01, dim=head_dim, theta=10000, use_real=True, repeat_interleave_real=True)'
ROPE_APPLICATION_SOURCE = 'diffusers.apply_rotary_emb tuple_full_dim_repeat_interleave equivalent'

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

def lin_plain(sd, key, x):
    return F.linear(x, sd[key].to(dtype=x.dtype, device=x.device))

def tmlp(sd, x):
    a = lin_plain(sd, 'time_guidance_embed.timestep_embedder.linear_1.weight', x)
    return lin_plain(sd, 'time_guidance_embed.timestep_embedder.linear_2.weight', F.silu(a))

def final(sd, img, temb):
    shift, scale = lin_plain(sd, 'norm_out.linear.weight', temb).chunk(2, dim=-1)
    y = F.layer_norm(img.float(), (img.shape[-1],), eps=1e-6)
    y = y * (1 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)
    return lin_plain(sd, 'proj_out.weight', y.to(img.dtype))

def run_stage1(sd, latent, ctx, tf):
    pred, *_ = run_wrapper(sd, latent, ctx, tf, True)
    return pred

def run_rope_double(sd, index, img, txt, temb):
    p = f'transformer_blocks.{index}'
    im = modulation_chunks(sd, 'double_stream_modulation_img.linear.weight', temb, 6)
    tx = modulation_chunks(sd, 'double_stream_modulation_txt.linear.weight', temb, 6)
    img_in = layer_norm_modulated(img, im[0], im[1])
    txt_in = layer_norm_modulated(txt, tx[0], tx[1])
    qi = lb(sd, f'{p}.attn.to_q', img_in)
    ki = lb(sd, f'{p}.attn.to_k', img_in)
    vi = lb(sd, f'{p}.attn.to_v', img_in)
    qt = lb(sd, f'{p}.attn.add_q_proj', txt_in)
    kt = lb(sd, f'{p}.attn.add_k_proj', txt_in)
    vt = lb(sd, f'{p}.attn.add_v_proj', txt_in)
    h = img.shape[-1]
    hd = int(sd[f'{p}.attn.norm_q.weight'].numel())
    heads = h // hd
    def shp(x):
        return x.view(x.shape[0], x.shape[1], heads, hd)
    qi = rms_norm_per_head(shp(qi), sd[f'{p}.attn.norm_q.weight'])
    ki = rms_norm_per_head(shp(ki), sd[f'{p}.attn.norm_k.weight'])
    qt = rms_norm_per_head(shp(qt), sd[f'{p}.attn.norm_added_q.weight'])
    kt = rms_norm_per_head(shp(kt), sd[f'{p}.attn.norm_added_k.weight'])
    q = rope(torch.cat([qt, qi], dim=1))
    k = rope(torch.cat([kt, ki], dim=1))
    v = torch.cat([shp(vt), shp(vi)], dim=1)
    y = explicit_attention(q, k, v, h, hd)
    yt, yi = y.split([txt.shape[1], img.shape[1]], dim=1)
    img = img + im[2].unsqueeze(1) * lb(sd, f'{p}.attn.to_out.0', yi)
    txt = txt + tx[2].unsqueeze(1) * lb(sd, f'{p}.attn.to_add_out', yt)
    img = img + im[5].unsqueeze(1) * lb(sd, f'{p}.ff.linear_out', swiglu(lb(sd, f'{p}.ff.linear_in', layer_norm_modulated(img, im[3], im[4]))))
    txt = txt + tx[5].unsqueeze(1) * lb(sd, f'{p}.ff_context.linear_out', swiglu(lb(sd, f'{p}.ff_context.linear_in', layer_norm_modulated(txt, tx[3], tx[4]))))
    return img, txt

def run_rope_single(sd, index, hidden, temb):
    p = f'single_transformer_blocks.{index}'
    shift, scale, gate = modulation_chunks(sd, 'single_stream_modulation.linear.weight', temb, 3)
    hidden_in = layer_norm_modulated(hidden, shift, scale)
    fused = lb(sd, f'{p}.attn.to_qkv_mlp_proj', hidden_in)
    h = hidden.shape[-1]
    mlp_size = fused.shape[-1] - 3 * h
    q, k, v, mlp = torch.split(fused, [h, h, h, mlp_size], dim=-1)
    hd = int(sd[f'{p}.attn.norm_q.weight'].numel())
    heads = h // hd
    q = rms_norm_per_head(q.view(q.shape[0], q.shape[1], heads, hd), sd[f'{p}.attn.norm_q.weight'])
    k = rms_norm_per_head(k.view(k.shape[0], k.shape[1], heads, hd), sd[f'{p}.attn.norm_k.weight'])
    v = v.view(v.shape[0], v.shape[1], heads, hd)
    q = rope(q)
    k = rope(k)
    attn = explicit_attention(q, k, v, h, hd)
    joined = torch.cat([attn, swiglu(mlp)], dim=-1)
    return hidden + gate.unsqueeze(1) * lb(sd, f'{p}.attn.to_out', joined)

def run_stage2_full_rope(sd, latent, ctx, tf):
    img = lin_plain(sd, 'x_embedder.weight', latent)
    txt = lin_plain(sd, 'context_embedder.weight', ctx)
    temb = tmlp(sd, tf)
    for index in range(DOUBLE_BLOCKS):
        img, txt = run_rope_double(sd, index, img, txt, temb)
    hidden = torch.cat([txt, img], dim=1)
    for index in range(SINGLE_BLOCKS):
        hidden = run_rope_single(sd, index, hidden, temb)
    _, img_out = hidden.split([txt.shape[1], img.shape[1]], dim=1)
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
    stage2 = execute_stage(sd, 'stage2_full_rope_stack_5double_20single', run_stage2_full_rope, 230002)
    report = {
        'source_model_ref': LOWBIT_REF,
        'uses_lowbit_source': True,
        'writes_expanded_checkpoint': False,
        'target': 'batched staged generation smoke with Diffusers-equivalent scaled-position RoPE on all wide blocks, logs, and PNG artifacts',
        'not_vae_decode': True,
        'rope_application_source': ROPE_APPLICATION_SOURCE,
        'rope_frequency_source': ROPE_FREQUENCY_SOURCE,
        'rope_frequency_equivalence_report': 'reports/bonsai-diffusers-rope-equivalence/report.json',
        'stages': [
            {**stage1, 'description': '1 double block RoPE wrapper path', 'rope_enabled': True, 'rope_scope': 'single_double_block_wrapper', 'rope_application_source': ROPE_APPLICATION_SOURCE, 'rope_frequency_source': ROPE_FREQUENCY_SOURCE},
            {**stage2, 'description': '5 double + 20 single stack with Diffusers-equivalent scaled-position pairwise RoPE on every tested attention block', 'rope_enabled': True, 'rope_scope': 'double_blocks_0_4_and_single_blocks_0_19_diffusers_equivalent_scaled_position_rope', 'rope_application_source': ROPE_APPLICATION_SOURCE, 'rope_frequency_source': ROPE_FREQUENCY_SOURCE},
        ],
        'all_stages_passed': all(s['all_steps_finite'] and s['has_png_artifact'] and s['png_size_bytes'] > 0 for s in [stage1, stage2]),
        'lowbit_path': str(path),
    }
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_stages_passed': report['all_stages_passed'], 'stage_count': len(report['stages']), 'rope_frequency_source': ROPE_FREQUENCY_SOURCE, 'pngs': [s['png_name'] for s in report['stages']]}, indent=2))
    return 0 if report['all_stages_passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
