def load_akl(repo):
    import torch
    from diffusers import AutoencoderKL
    v=AutoencoderKL.from_pretrained(repo,subfolder='vae',torch_dtype=torch.float32)
    return v.to('cpu').eval()
