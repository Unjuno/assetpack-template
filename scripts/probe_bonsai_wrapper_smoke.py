#!/usr/bin/env python3
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_lowbit_rope_smoke import run_block

OUT_DIR = Path('reports/bonsai-wrapper-smoke')

def lin(sd, key, x):
    return F.linear(x, sd[key].to(dtype=x.dtype, device=x.device))

def tmlp(sd, x):
    return lin(sd, 'time_guidance_embed.timestep_embedder.linear_2.weight', F.silu(lin(sd, 'time_guidance_embed.timestep_embedder.linear_1.weight', x)))

def stat(x):
    y=x.detach().float(); f=torch.isfinite(y)
    return {'shape':list(x.shape),'finite':bool(f.all()),'min':float(y[f].min()),'max':float(y[f].max()),'mean':float(y[f].mean())}

def final(sd, hidden, temb):
    shift, scale = lin(sd, 'norm_out.linear.weight', temb).chunk(2, dim=-1)
    y = F.layer_norm(hidden.float(), (hidden.shape[-1],), eps=1e-6)
    y = y * (1 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)
    return lin(sd, 'proj_out.weight', y.to(hidden.dtype))

def run(sd, xf, cf, tf, runtime):
    img = lin(sd, 'x_embedder.weight', xf)
    txt = lin(sd, 'context_embedder.weight', cf)
    temb = tmlp(sd, tf)
    img, txt = run_block(sd, img, txt, temb, runtime)
    out = final(sd, img, temb)
    return out, img, txt, temb

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    g=torch.Generator(device='cpu'); g.manual_seed(180000)
    xf=torch.randn((1,2,128),generator=g)*0.01
    cf=torch.randn((1,3,7680),generator=g)*0.01
    tf=torch.randn((1,256),generator=g)*0.01
    with torch.inference_mode():
        r=run(sd,xf,cf,tf,True); e=run(sd,xf,cf,tf,False)
    diffs=[a-b for a,b in zip(r,e)]
    ok=all(torch.allclose(a,b,rtol=1e-4,atol=1e-5) for a,b in zip(r,e))
    report={'source_model_ref':LOWBIT_REF,'uses_lowbit_source':True,'writes_expanded_checkpoint':False,'target':'minimal wrapper: embedders + one RoPE double block + final projection','not_full_pipeline':True,'double_block_count':1,'single_block_count':0,'output':stat(r[0]),'max_abs_error':max(float(d.abs().max()) for d in diffs),'mean_abs_error':max(float(d.abs().mean()) for d in diffs),'allclose_rtol_1e_4_atol_1e_5':bool(ok),'lowbit_path':str(path)}
    (OUT_DIR/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'allclose':bool(ok),'max_abs_error':report['max_abs_error'],'output_shape':report['output']['shape']},indent=2))
    return 0 if ok else 1
if __name__=='__main__':
    raise SystemExit(main())
