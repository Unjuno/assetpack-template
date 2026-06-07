from bonsai_ctx3 import ctx3


def ctx_like(x):
    return ctx3().to(x.dtype)
