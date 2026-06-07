from bonsai_next import apply_delta
from bonsai_pred_one import pred_once
from probe_bonsai_staged_generation_smoke import STEP_COUNT,STEP_SIZE


def prompt_z():
    z,c,p=pred_once()
    for _ in range(STEP_COUNT):
        z=apply_delta(z,p,STEP_SIZE)
    return z,c,p
