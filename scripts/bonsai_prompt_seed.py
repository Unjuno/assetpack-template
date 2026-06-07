import torch


def seeded_inputs(seed=270000):
    g=torch.Generator(device='cpu')
    g.manual_seed(seed)
    z=torch.randn((1,16,128),generator=g)*0.01
    t=torch.randn((1,256),generator=g)*0.01
    return z,t
