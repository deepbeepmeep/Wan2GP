import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS
from einops import rearrange, repeat


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)
        img_cond = ae.encode(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict


def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_list: list,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
    neg_prompt: str | list[str] = None,
    real_guidance_scale = False,

) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond_seq = None
    img_cond_seq_ids = None
    if img_cond_list == None: img_cond_list = []
    for cond_no, img_cond in enumerate(img_cond_list): 
        width, height = img_cond.size
        aspect_ratio = width / height

        # Kontext is trained on specific resolutions, using one of them is recommended
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)

        width = 2 * int(width / 16)
        height = 2 * int(height / 16)

        img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
        img_cond = np.array(img_cond)
        img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
        img_cond = rearrange(img_cond, "h w c -> 1 c h w")
        with torch.no_grad():
            img_cond_latents = ae.encode(img_cond.to(device))

        img_cond_latents = img_cond_latents.to(torch.bfloat16)
        img_cond_latents = rearrange(img_cond_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img_cond.shape[0] == 1 and bs > 1:
            img_cond_latents = repeat(img_cond_latents, "1 ... -> bs ...", bs=bs)
        img_cond = None

        # image ids are the same as base image with the first dimension set to 1
        # instead of 0
        img_cond_ids = torch.zeros(height // 2, width // 2, 3)
        img_cond_ids[..., 0] = cond_no + 1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

        if target_width is None:
            target_width = 8 * width
        if target_height is None:
            target_height = 8 * height
        img_cond_ids = img_cond_ids.to(device)
        if cond_no == 0:
            img_cond_seq, img_cond_seq_ids  = img_cond_latents, img_cond_ids
        else:
            img_cond_seq, img_cond_seq_ids  =  torch.cat([img_cond_seq, img_cond_latents], dim=1), torch.cat([img_cond_seq_ids, img_cond_ids], dim=1)
        
    img = get_noise(
        bs,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    if real_guidance_scale != 1:
        temp_dict = prepare(t5, clip, img, neg_prompt)
        return_dict["neg_txt"]=  temp_dict["txt"]
        return_dict["neg_txt_ids"]=  temp_dict["txt_ids"]
    else:
        return_dict["neg_txt"]=  None
        return_dict["neg_txt_ids"]=  None
    return_dict["img_cond_seq"] = img_cond_seq
    return_dict["img_cond_seq_ids"] = img_cond_seq_ids

    return return_dict, target_height, target_width


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    real_guidance_scale = None,
    # extra img tokens (channel-wise)
    neg_txt: Tensor = None,
    neg_txt_ids: Tensor= None,
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    callback=None,
    pipeline=None,
    loras_slists=None,
    unpack_latent = None,
    joint_pass= False,
):

    kwargs = {'pipeline': pipeline, 'callback': callback}
    if callback != None:
        callback(-1, None, True)

    updated_num_steps= len(timesteps) -1
    if callback != None:
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(model, loras_slists, updated_num_steps)
        callback(-1, None, True, override_num_inference_steps = updated_num_steps)
    from mmgp import offload
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        offload.set_step_no_for_lora(model, i)
        if pipeline._interrupt:
            return None

        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        if not joint_pass or real_guidance_scale == 1:
            pred = model(
                img=img_input,
                img_ids=img_input_ids,
                txt_list=[txt],
                txt_ids_list=[txt_ids],
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                **kwargs
            )[0]
            if pred == None: return None
            if real_guidance_scale> 1:
                neg_pred = model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt_list=[neg_txt],
                    txt_ids_list=[neg_txt_ids],
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    **kwargs
                )[0]
                if neg_pred == None: return None
        else:
            pred, neg_pred = model(
                img=img_input,
                img_ids=img_input_ids,
                txt_list=[txt, neg_txt],
                txt_ids_list=[txt_ids, neg_txt_ids],
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                **kwargs
            )
            if pred == None: return None


        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]
        if real_guidance_scale > 1:
            if img_input_ids is not None:
                neg_pred = neg_pred[:, : img.shape[1]]
            pred = neg_pred + real_guidance_scale * (pred - neg_pred)

        img += (t_prev - t_curr) * pred
        if callback is not None:
            preview = unpack_latent(img).transpose(0,1)
            callback(i, preview, False)         


    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
