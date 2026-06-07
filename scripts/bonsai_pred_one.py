from bonsai_ctx_cast import ctx_like
from bonsai_lowbit_step import load_sd,step
from bonsai_prompt_seed import seeded_inputs


def pred_once():
    sd=load_sd(); z,t=seeded_inputs(); c=ctx_like(z)
    return z,c,step(sd,z,c,t)
