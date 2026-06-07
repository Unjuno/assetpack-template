def vae_sample(v,z,s):
    return v.decode(z.detach().float()/s).sample
