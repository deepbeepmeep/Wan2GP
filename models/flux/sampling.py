import math
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS
import torchvision.transforms.functional as TVF


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    *,
    channels: int = 16,
    patch_size: int = 2,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    if channels == 16 and patch_size == 2:
        return torch.randn(
            num_samples,
            channels,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    return torch.randn(
        num_samples,
        channels,
        height,
        width,
        dtype=dtype,
        device=device,
        generator=generator,
    )


def prepare_prompt(t5: HFEmbedder, clip: HFEmbedder, bs: int, prompt: str | list[str], neg: bool = False, device: str = "cuda") -> dict[str, Tensor]:
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

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
        "neg_txt" if neg else "txt": txt.to(device),
        "neg_txt_ids" if neg else "txt_ids": txt_ids.to(device),
        "neg_vec" if neg else "vec": vec.to(device),
    }


def prepare_img(img: Tensor, patch_size: int = 2) -> dict[str, Tensor]:
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
    }





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

def resizeinput(img):
    multiple_of = 16
    image_height, image_width = img.height, img.width
    aspect_ratio = image_width / image_height
    _, image_width, image_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
    )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of
    if (image_width, image_height) != img.size:
        img = img.resize((image_width, image_height), Image.LANCZOS)
    return img


def build_mask(target_width, target_height, img_mask, device):
    from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
    # image_height, image_width = calculate_new_dimensions(ref_height, ref_width, image_height, image_width, False, block_size=multiple_of)
    image_mask_latents = convert_image_to_tensor(img_mask.resize((target_width // 16, target_height // 16), resample=Image.Resampling.LANCZOS))
    image_mask_latents = torch.where(image_mask_latents>-0.5, 1., 0. )[0:1]
    image_mask_rebuilt = image_mask_latents.repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2).unsqueeze(0)
    # convert_tensor_to_image( image_mask_rebuilt.squeeze(0).repeat(3,1,1)).save("mmm.png")
    image_mask_latents = image_mask_latents.reshape(1, -1, 1).to(device)        
    return {
        "img_msk_latents": image_mask_latents,
        "img_msk_rebuilt": image_mask_rebuilt,
    }


def prepare_kontext(
    ae: AutoEncoder | None,
    img_cond_list: list,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
    img_mask = None,
    *,
    patch_size: int = 2,
    noise_channels: int = 16,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image

    res_match_output = img_mask is not None

    img_cond_seq = None
    img_cond_seq_ids = None
    if img_cond_list == None: img_cond_list = []
    height_offset = 0
    width_offset = 0
    for cond_no, img_cond in enumerate(img_cond_list): 
        if res_match_output:
            if img_cond.size != (target_width, target_height):
                img_cond = img_cond.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            img_cond = resizeinput(img_cond)
        width, height = img_cond.size
        width, height = width // 8, height // 8

        img_cond = np.array(img_cond)
        img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
        img_cond = rearrange(img_cond, "h w c -> 1 c h w")
        if ae is None:
            raise ValueError("Image conditioning is not supported for this model.")
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
        img_cond_ids[..., 0] = 1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None] + height_offset
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :] + width_offset
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)
        height_offset +=  height // 2 
        width_offset +=  width // 2

        if target_width is None:
            target_width = 8 * width
        if target_height is None:
            target_height = 8 * height
        img_cond_ids = img_cond_ids.to(device)
        if cond_no == 0:
            img_cond_seq, img_cond_seq_ids  = img_cond_latents, img_cond_ids
        else:
            img_cond_seq, img_cond_seq_ids  =  torch.cat([img_cond_seq, img_cond_latents], dim=1), torch.cat([img_cond_seq_ids, img_cond_ids], dim=1)
        
    return_dict = {
        "img_cond_seq": img_cond_seq,
        "img_cond_seq_ids": img_cond_seq_ids,
    }
    if img_mask is not None:
        return_dict.update(build_mask(target_width, target_height, img_mask), device)

    img = get_noise(
        bs,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
        channels=noise_channels,
        patch_size=patch_size,
    )
    return_dict.update(prepare_img(img, patch_size=patch_size))

    return return_dict, target_height, target_width


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b



def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule_flux2(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

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
    neg_vec: Tensor = None,
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    siglip_embedding = None,
    siglip_embedding_ids = None,
    callback=None,
    pipeline=None,
    loras_slists=None,
    unpack_latent = None,
    joint_pass= False,
    img_msk_latents = None,
    img_msk_rebuilt = None,
    denoising_strength = 1,
    masking_strength = 1,
    preview_meta = None,
    original_image_latents = None,
):

    kwargs = {
        'pipeline': pipeline,
        'callback': callback,
        "img_len": img.shape[1],
        "siglip_embedding": siglip_embedding,   
        "siglip_embedding_ids": siglip_embedding_ids,
    }

    if callback != None:
        callback(-1, None, True)

    original_timesteps = timesteps
    
    morph, first_step = False, 0
    if img_msk_latents is not None:
        if original_image_latents is None: original_image_latents= img_cond_seq.clone() 
        randn = torch.randn_like(original_image_latents)
        if denoising_strength < 1.:
            first_step = int(len(timesteps[:-1]) * (1. - denoising_strength))
        masked_steps = math.ceil(len(timesteps[:-1]) * masking_strength)
        if not morph:
            latent_noise_factor = timesteps[first_step]
            latents  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor
            img = latents.to(img)
            latents = None
            timesteps = timesteps[first_step:]


    updated_num_steps= len(timesteps) -1
    if callback != None:
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(model, loras_slists, len(original_timesteps))
        callback(-1, None, True, override_num_inference_steps = updated_num_steps)
    from mmgp import offload
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        offload.set_step_no_for_lora(model, first_step  + i)
        if pipeline._interrupt:
            return None

        if img_msk_latents is not None and denoising_strength <1. and i == first_step and morph:
            latent_noise_factor = t_curr/1000
            img  = original_image_latents  * (1.0 - latent_noise_factor) + img * latent_noise_factor 

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
                y_list=[vec],
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
                    y_list=[neg_vec],
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
                y_list=[vec, neg_vec],
                timesteps=t_vec,
                guidance=guidance_vec,
                **kwargs
            )
            if pred == None: return None

        if real_guidance_scale > 1:
            pred = neg_pred + real_guidance_scale * (pred - neg_pred)

        img += (t_prev - t_curr) * pred

        if img_msk_latents is not None and i < masked_steps:
            latent_noise_factor = t_prev
            # noisy_image  = original_image_latents  * (1.0 - latent_noise_factor) + torch.randn_like(original_image_latents) * latent_noise_factor 
            noisy_image  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor 
            img  =  noisy_image * (1-img_msk_latents)  + img_msk_latents * img
            noisy_image = None

        if callback is not None:
            preview = unpack_latent(img).transpose(0,1)
            callback(i, preview, False, preview_meta=preview_meta)


    return img

def prepare_multi_ip(
    ae: AutoEncoder,
    img_cond_list: list,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
    pe: Literal["d", "h", "w", "o"] = "d",
    conditions_zero_start = False,
    set_cond_index = False,
    res_match_output = True,
    patch_size: int = 2    
    
) -> dict[str, Tensor]:

    assert pe in ["d", "h", "w", "o"]

    if img_cond_list == None: img_cond_list = []

    if not res_match_output:
        for i, img_cond in enumerate(img_cond_list):
            img_cond_list[i]= resizeinput(img_cond)

    ref_imgs = [
        ae.encode(
            (TVF.to_tensor(ref_img) * 2.0 - 1.0)
            .unsqueeze(0)
            .to(device, torch.float32)
        ).to(torch.bfloat16)
        for ref_img in img_cond_list
    ]

    img = get_noise( bs, target_height, target_width, device=device, dtype=torch.bfloat16, seed=seed)
    bs, c, h, w = img.shape
    # tgt img
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    img_cond_seq = img_cond_seq_ids = None
    if conditions_zero_start:
        pe_shift_w = pe_shift_h = 0
    else:
        pe_shift_w, pe_shift_h = w // patch_size, h // patch_size
    for cond_no, ref_img in enumerate(ref_imgs):
        _, _, ref_h1, ref_w1 = ref_img.shape
        ref_img = rearrange(
            ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3)
        if set_cond_index:
            ref_img_ids1[..., 0] = cond_no + 1
        h_offset = pe_shift_h if pe in {"d", "h"} else 0
        w_offset = pe_shift_w if pe in {"d", "w"} else 0
        ref_img_ids1[..., 1] = (
            ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2)[:, None] + h_offset
        )
        ref_img_ids1[..., 2] = (
            ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2)[None, :] + w_offset
        )
        ref_img_ids1 = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)

        if target_width is None:
            target_width = 8 * ref_w1
        if target_height is None:
            target_height = 8 * ref_h1
        ref_img_ids1 = ref_img_ids1.to(device)
        if cond_no == 0:
            img_cond_seq, img_cond_seq_ids  = ref_img, ref_img_ids1
        else:
            img_cond_seq, img_cond_seq_ids  =  torch.cat([img_cond_seq, ref_img], dim=1), torch.cat([img_cond_seq_ids, ref_img_ids1], dim=1)


        # 更新pe shift
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "img_cond_seq": img_cond_seq,
        "img_cond_seq_ids": img_cond_seq_ids,
    }, target_height, target_width


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def patches_to_image(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    tokens = x.transpose(1, 2)
    return F.fold(
        tokens,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size,
    )
