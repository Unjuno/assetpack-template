def vae_sample(v,z,s):
    return v.decode(z/s).sample
