from pathlib import Path
from bonsai_akl import load_akl
from bonsai_img_save import save_img
from bonsai_native_unpack import unpack_tokens_to_vae_latent
from bonsai_prompt_z import prompt_z
from bonsai_tensor_stats import tensor_stat
from bonsai_vae_apply import vae_sample
from bonsai_vae_cfg import vae_cfg
from bonsai_vae_shape import PATCH_SIZE,TOKEN_GRID

REPO='prism-ml/bonsai-image-binary-4B-gemlite-1bit'


def run(out):
    z,c,p=prompt_z()
    v=load_akl(REPO)
    ch,s=vae_cfg(v)
    vz=unpack_tokens_to_vae_latent(z,ch,TOKEN_GRID,PATCH_SIZE)
    y=vae_sample(v,vz,s)
    png=Path(out)/'prompt_staged_vae.png'
    save_img(y,png)
    return {'decode_success':png.exists(),'png_size_bytes':png.stat().st_size,'ctx':tensor_stat(c),'pred':tensor_stat(p),'token_latent':tensor_stat(z),'vae_latent':tensor_stat(vz),'decoded':tensor_stat(y),'latent_channels':ch,'scaling_factor':s}
