def unpack_tokens_to_vae_latent(t,c,g,p):
    b,n,d=t.shape
    if n!=g*g:
        raise ValueError('bad token count')
    if d!=c*p*p:
        raise ValueError('bad token dim')
    x=t.detach().float().reshape(b,g,g,p,p,c)
    x=x.permute(0,5,1,3,2,4).contiguous()
    return x.reshape(b,c,g*p,g*p)
