import torch
from bonsai_lowbit_recover import LOWBIT_REF, load_lowbit_transformer_state_dict
from bonsai_prompt_context import load_prompt_context
from bonsai_prompt_context_select import select_staged_context
from bonsai_tensor_stats import tensor_stat
from probe_bonsai_staged_generation_smoke import STEP_SIZE, run_stage2_full_rope

def run_prompt_staged_once():
    _, sd = load_lowbit_transformer_state_dict(LOWBIT_REF)
    g = torch.Generator(device='cpu'); g.manual_seed(270000)
    latent = torch.randn((1, 16, 128), generator=g) * 0.01
    tf = torch.randn((1, 256), generator=g) * 0.01
    ctx = select_staged_context(load_prompt_context()).to(latent.dtype)
    with torch.inference_mode():
        pred = run_stage2_full_rope(sd, latent, ctx, tf)
        nxt = latent - STEP_SIZE * pred
    return {'ctx': tensor_stat(ctx), 'pred': tensor_stat(pred), 'latent_after': tensor_stat(nxt)}
