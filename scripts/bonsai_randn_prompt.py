from pathlib import Path
from bonsai_ctx3 import ctx3


def make_randn(orig):
    def f(shape,*a,**k):
        if tuple(shape)==(1,3,7680):
            return ctx3().float()
        return orig(shape,*a,**k)
    return f


def run(m):
    m.torch.randn=make_randn(m.torch.randn)
    m.OUT_DIR=Path('reports/bonsai-prompt-staged-vae-smoke')
    return m.main()
