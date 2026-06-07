from bonsai_ctx3 import ctx3


def make_randn(orig):
    def f(shape,*a,**k):
        if tuple(shape)==(1,3,7680):
            return ctx3().float()
        return orig(shape,*a,**k)
    return f
