def save_img(x,path):
    from PIL import Image
    y=x[0].detach().float().clamp(-1,1)
    y=((y+1)*127.5).clamp(0,255).byte()
    y=y.permute(1,2,0).cpu().numpy()
    Image.fromarray(y).save(path)
