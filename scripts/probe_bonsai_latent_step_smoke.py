#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from probe_bonsai_wrapper_smoke import run, stat

OUT_DIR = Path('reports/bonsai-latent-step-smoke')
STEP_SIZE = 0.05

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    g=torch.Generator(device='cpu'); g.manual_seed(190000)
    latent=torch.randn((1,2,128),generator=g)*0.01
    ctx=torch.randn((1,3,7680),generator=g)*0.01
    tf=torch.randn((1,256),generator=g)*0.01
    with torch.inference_mode():
        pred_r,*_=run(sd,latent,ctx,tf,True)
        pred_e,*_=run(sd,latent,ctx,tf,False)
        next_r=latent-STEP_SIZE*pred_r
        next_e=latent-STEP_SIZE*pred_e
    dp=pred_r-pred_e; dn=next_r-next_e
    ok=torch.allclose(pred_r,pred_e,rtol=1e-4,atol=1e-5) and torch.allclose(next_r,next_e,rtol=1e-4,atol=1e-5)
    report={'source_model_ref':LOWBIT_REF,'uses_lowbit_source':True,'writes_expanded_checkpoint':False,'target':'one-step latent denoise smoke using wrapper prediction','not_full_pipeline':True,'has_scheduler_like_update':True,'step_size':STEP_SIZE,'latent':stat(latent),'prediction':stat(pred_r),'next_latent':stat(next_r),'max_abs_error':max(float(dp.abs().max()),float(dn.abs().max())),'mean_abs_error':max(float(dp.abs().mean()),float(dn.abs().mean())),'allclose_rtol_1e_4_atol_1e_5':bool(ok),'lowbit_path':str(path)}
    (OUT_DIR/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'allclose':bool(ok),'max_abs_error':report['max_abs_error'],'next_latent_shape':report['next_latent']['shape']},indent=2))
    return 0 if ok else 1
if __name__=='__main__':
    raise SystemExit(main())
