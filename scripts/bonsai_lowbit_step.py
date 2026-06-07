from bonsai_lowbit_recover import LOWBIT_REF,load_lowbit_transformer_state_dict
from probe_bonsai_staged_generation_smoke import run_stage2_full_rope


def load_sd():
    return load_lowbit_transformer_state_dict(LOWBIT_REF)[1]


def step(sd,z,c,t):
    return run_stage2_full_rope(sd,z,c,t)
