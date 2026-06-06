def select_staged_context(ctx):
    if ctx.ndim != 3:
        raise ValueError('ctx must be [B,T,C]')
    if ctx.shape[0] < 1 or ctx.shape[1] < 3:
        raise ValueError('ctx must have B>=1 and T>=3')
    return ctx[:1, :3, :]
