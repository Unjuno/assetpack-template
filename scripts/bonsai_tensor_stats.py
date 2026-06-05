def tensor_stat(x):
    import torch
    y = x.detach().float()
    return {
        'shape': list(x.shape),
        'dtype': str(x.dtype),
        'finite': bool(torch.isfinite(y).all()),
        'mean_abs': float(y.abs().mean()),
        'max_abs': float(y.abs().max()),
    }
