def vae_cfg(v):
    c=dict(getattr(v,'config',{}))
    ch=int(c.get('latent_channels',32))
    sf=float(c.get('scaling_factor',1.0))
    return ch,sf
