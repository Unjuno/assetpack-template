from bonsai_ctx_cast import ctx_like
from bonsai_lowbit_step import load_sd,step
from bonsai_next import apply_delta
from bonsai_prompt_seed import seeded_inputs
from probe_bonsai_staged_generation_smoke import STEP_COUNT,STEP_SIZE


def prompt_z():
    sd=load_sd(); z,t=seeded_inputs(); c=ctx_like(z)
    p=None
    for _ in range(STEP_COUNT):
        p=step(sd,z,c,t)
        z=apply_delta(z,p,STEP