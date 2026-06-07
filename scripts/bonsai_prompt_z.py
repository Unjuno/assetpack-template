from bonsai_next import apply_delta
from bonsai_pred_one import pred_once
from probe_bonsai_staged_generation_smoke import STEP_SIZE


def prompt_z():
    z,c,p=pred_once()
    return apply_delta(z,p,STEP_SIZE),c,p
