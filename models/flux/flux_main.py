import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from mmgp import offload as offload
import torch
from shared.utils.utils import calculate_new_dimensions
from .sampling import denoise, get_schedule, prepare_kontext, unpack
from .modules.layers import get_linear_split_map
from .util import (
    aspect_ratio_to_height_width,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)

from PIL import Image

def stitch_images(img1, img2):
    # Resize img2 to match img1's height
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width2 = int(width2 * height1 / height2)
    img2_resized = img2.resize((new_width2, height1), Image.Resampling.LANCZOS)
    
    stitched = Image.new('RGB', (width1 + new_width2, height1))
    stitched.paste(img1, (0, 0))
    stitched.paste(img2_resized, (width1, 0))
    return stitched

class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        self.device = torch.device(f"cuda")
        self.VAE_dtype = VAE_dtype
        self.dtype = dtype
        torch_device = "cpu"
        self.guidance_max_phases = model_def.get("guidance_max_phases", 0) 

        # model_filename = ["c:/temp/flux1-schnell.safetensors"] 
        
        self.t5 = load_t5(torch_device, text_encoder_filename, max_length=512)
        self.clip = load_clip(torch_device)
        self.name = model_def.get("flux-model", "flux-dev")
        # self.name= "flux-dev-kontext"
        # self.name= "flux-dev"
        # self.name= "flux-schnell"
        source =  model_def.get("source", None)
        self.model = load_flow_model(self.name, model_filename[0] if source is None else source, torch_device)

        self.vae = load_ae(self.name, device=torch_device)

        # offload.change_dtype(self.model, dtype, True)
        # offload.save_model(self.model, "flux-dev.safetensors")

        if not source is None:
            from wgp import save_model
            save_model(self.model, model_type, dtype, None)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.model, model_type, model_filename[0], dtype, None)

        split_linear_modules_map = get_linear_split_map()
        self.model.split_linear_modules_map = split_linear_modules_map
        offload.split_linear_modules(self.model, split_linear_modules_map )

    
    def generate(
            self,
            seed: int | None = None,
            input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
            n_prompt: str = None,
            sampling_steps: int = 20,
            input_ref_images = None,
            width= 832,
            height=480,
            embedded_guidance_scale: float = 2.5,
            guide_scale = 2.5,
            fit_into_canvas = None,
            callback = None,
            loras_slists = None,
            batch_size = 1,
            video_prompt_type = "",
            joint_pass = False,            
            **bbargs
    ):
            if self._interrupt:
                return None
            if self.guidance_max_phases < 1:
                guide_scale = 1
            if n_prompt is None or len(n_prompt) == 0:
                n_prompt = "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
            device="cuda"
            if "I" in video_prompt_type and input_ref_images != None and len(input_ref_images) > 0: 
                if "K" in video_prompt_type and False :
                    # image latents tiling method
                    w, h = input_ref_images[0].size
                    height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)
                else:
                    # image stiching method
                    stiched = input_ref_images[0]
                    if "K" in video_prompt_type :
                        w, h = input_ref_images[0].size
                        height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)

                    for new_img in input_ref_images[1:]:
                        stiched = stitch_images(stiched, new_img)
                    input_ref_images  = [stiched]
            else:
                input_ref_images = None

            inp, height, width = prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=input_prompt,
                neg_prompt= n_prompt,
                ae=self.vae,
                img_cond_list=input_ref_images,
                target_width=width,
                target_height=height,
                bs=batch_size,
                seed=seed,
                device=device,
                real_guidance_scale=guide_scale,
            )

            timesteps = get_schedule(sampling_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
            def unpack_latent(x):
                return unpack(x.float(), height, width) 
            # denoise initial noise
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=embedded_guidance_scale, real_guidance_scale =guide_scale, callback=callback, pipeline=self, loras_slists= loras_slists, unpack_latent = unpack_latent, joint_pass = joint_pass)
            if x==None: return None
            # decode latents to pixel space
            x = unpack_latent(x)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x = self.vae.decode(x)

            x = x.clamp(-1, 1)
            x = x.transpose(0, 1)
            return x

