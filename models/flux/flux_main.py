import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from mmgp import offload as offload
import torch
from shared.utils.utils import calculate_new_dimensions
from .sampling import denoise, get_schedule, get_schedule_flux2, prepare_kontext, prepare_prompt, prepare_multi_ip, unpack, resizeinput, patches_to_image, build_mask
from .modules.layers import get_linear_split_map
from transformers import SiglipVisionModel, SiglipImageProcessor
import torchvision.transforms.functional as TVF
import math
from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
from shared.utils import files_locator as fl 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
from .modules.autoencoder_flux2 import AutoencoderKLFlux2, AutoEncoderParamsFlux2

from .util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from .flux2_adapter import (
    scatter_ids ,
    batched_prc_img, 
    batched_prc_txt,
    encode_image_refs,
    )
from .modules.autoencoder_flux2 import AutoencoderKLFlux2

from PIL import Image
def preprocess_ref(raw_image: Image.Image, long_size: int = 512):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image

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
        self._interrupt = False
        self.VAE_dtype = VAE_dtype
        self.dtype = dtype
        torch_device = "cpu"
        self.model_def = model_def 
        self.name = model_def.get("flux-model", "flux-dev")
        self.guidance_max_phases = model_def.get("guidance_max_phases", 0) 
        self.is_flux2 = self.name.startswith("flux2")

        # model_filename = ["c:/temp/flux1-schnell.safetensors"] 
        source = model_def.get("source", None)
        self.clip = self.t5 = self.vision_encoder = self.mistal = None
        if self.is_flux2:
            self.model = load_flow_model(self.name, model_filename[0] if source is None else source, torch_device)

            from .modules.text_encoder_mistral import Mistral3SmallEmbedder
            self.mistral = Mistral3SmallEmbedder( model_spec = text_encoder_filename)
    
            with torch.device("meta"):
                self.vae  = AutoencoderKLFlux2(AutoEncoderParamsFlux2())

            offload.load_model_data(self.vae, fl.locate_file("flux2_vae.safetensors"), writable_tensors= False, )
            self.vae_scale_factor = 8
        else:
            self.t5 = load_t5(torch_device, text_encoder_filename, max_length=512)
            self.clip = load_clip(torch_device)
            self.name = model_def.get("flux-model", "flux-dev")
            # self.name= "flux-dev-kontext"
            # self.name= "flux-dev"
            # self.name= "flux-schnell"
            source =  model_def.get("source", None)
            self.model = load_flow_model(self.name, model_filename[0] if source is None else source, torch_device)
            self.model_def = model_def 
            self.vae = None if getattr(self.model, "radiance", False) else load_ae(self.name, device=torch_device)

        siglip_processor = siglip_model = feature_embedder = None
        if self.name == 'flux-dev-uso':
            siglip_path =  fl.locate_folder("siglip-so400m-patch14-384")
            siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
            siglip_model = SiglipVisionModel.from_pretrained(siglip_path)
            siglip_model.eval().to("cpu")
            if len(model_filename) > 1:
                from .modules.layers import SigLIPMultiFeatProjModel                
                feature_embedder = SigLIPMultiFeatProjModel(
                    siglip_token_nums=729,
                    style_token_nums=64,
                    siglip_token_dims=1152,
                    hidden_size=3072, #self.hidden_size,
                    context_layer_norm=True,
                )
                offload.load_model_data(feature_embedder, model_filename[1])
        self.vision_encoder = siglip_model
        self.vision_encoder_processor = siglip_processor
        self.feature_embedder = feature_embedder

        if self.name in ['flux-dev-kontext-dreamomni2']:
            self.processor = Qwen2VLProcessor.from_pretrained(fl.locate_folder("Qwen2.5-VL-7B-DreamOmni2"))
            self.vlm_model = offload.fast_load_transformers_model(fl.locate_file( os.path.join("Qwen2.5-VL-7B-DreamOmni2","Qwen2.5-VL-7B-DreamOmni2_quanto_bf16_int8.safetensors")),  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration,  defaultConfigPath= fl.locate_file(os.path.join("Qwen2.5-VL-7B-DreamOmni2", "config.json")))
        else:
            self.processor = None
            self.vlm_model = None
        # offload.change_dtype(self.model, dtype, True)
        # offload.save_model(self.model, "flux-dev.safetensors")

        if not source is None:
            from wgp import save_model
            save_model(self.model, model_type, dtype, None)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.model, model_type, model_filename[0], dtype, None)

        split_linear_modules_map = get_linear_split_map(
            self.model.hidden_size,
            getattr(self.model.params, "mlp_ratio", 4.0),
            getattr(self.model.params, "single_linear1_mlp_ratio", None),
            getattr(self.model.params, "double_linear1_mlp_ratio", None),
        )
        self.model.split_linear_modules_map = split_linear_modules_map
        offload.split_linear_modules(self.model, split_linear_modules_map )

    def infer_vlm(self, input_img_path,input_instruction,prefix):
        tp=[]
        for path in input_img_path:
            tp.append({"type": "image", "image": path})
        tp.append({"type": "text", "text": input_instruction+prefix})
        messages = [
                {
                    "role": "user",
                    "content": tp,
                }
            ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # from .vprocess import process_vision_info
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=input_img_path,
            # images=image_inputs,
            # videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cpu")

        # Inference
        generated_ids = self.vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    
    def generate(
            self,
            seed: int | None = None,
            input_prompt: str = "replace the logo with the text 'Black Forest Labs'",            
            n_prompt: str = None,
            sampling_steps: int = 20,
            input_ref_images = None,
            input_frames= None,
            input_masks= None,
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
            image_refs_relative_size = 100,
            denoising_strength = 1.,
            masking_strength = 1.,
            **bbargs
    ):
            if self._interrupt:
                return None
            device="cuda"
            flux2 = self.is_flux2
            if flux2:
                guide_scale = 1.0
            if self.guidance_max_phases < 1: guide_scale = 1
            if n_prompt is None or len(n_prompt) == 0: n_prompt = "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
            flux_dev_uso = self.name in ['flux-dev-uso']
            flux_dev_umo = self.name in ['flux-dev-umo']
            radiance = self.name in ['flux-chroma-radiance']
            flux_kontext_dreamomni2 = self.name in ['flux-dev-kontext-dreamomni2']

            if flux2:
                if input_frames is not None:
                    input_ref_images = [convert_tensor_to_image(input_frames) ] + (input_ref_images or [])
                
                shape = (batch_size, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
                img, img_ids = batched_prc_img(randn)                
                ctx = self.mistral([input_prompt]).to(torch.bfloat16)
                txt_embeds, txt_ids = batched_prc_txt(ctx)
                txt_embeds, txt_ids = txt_embeds.expand(batch_size, -1, -1), txt_ids.expand(batch_size, -1, -1)
                vec = torch.zeros(batch_size, 1, device=device, dtype=self.dtype)
                inp = { "img": img, "img_ids": img_ids, "txt": txt_embeds.to(device), "txt_ids": txt_ids.to(device), "vec": vec }
                if guide_scale != 1:
                    ctx = self.mistral([n_prompt]).to(torch.bfloat16)
                    txt_embeds, txt_ids = batched_prc_txt(ctx)
                    txt_embeds, txt_ids = txt_embeds.expand(batch_size, -1, -1), txt_ids.expand(batch_size, -1, -1)
                    inp.update({ "neg_txt": txt_embeds.to(device), "neg_txt_ids": txt_ids.to(device), "neg_vec": vec })

                if input_masks is not None:
                    inp.update( build_mask(width, height, convert_tensor_to_image(input_masks, mask_levels= True), device))
                    inp["original_image_latents"], _ = encode_image_refs(self.vae, [input_ref_images[0].resize((width, height), resample=Image.Resampling.LANCZOS)]) 

                if input_ref_images is not None and len(input_ref_images):
                    cond_latents, cond_ids = encode_image_refs(self.vae, input_ref_images)
                    cond_latents, cond_ids = cond_latents.expand(batch_size, -1, -1), cond_ids.expand(batch_size, -1, -1)
                    inp.update({"img_cond_seq": cond_latents, "img_cond_seq_ids": cond_ids})

                noise_patch_size = 2
                timesteps = get_schedule_flux2(sampling_steps, inp["img"].shape[1])
                unpack_latent = lambda x : self.vae.pre_decode(torch.cat(scatter_ids(x, inp["img_ids"])).squeeze(2))
                ref_style_imgs = []
                image_mask = None

            else:
                latent_stiching = flux_dev_uso or  flux_dev_umo or flux_kontext_dreamomni2
                lock_dimensions=  False
                input_ref_images = [] if input_ref_images is None else input_ref_images[:]
                if flux_dev_umo:
                    ref_long_side = 512 if len(input_ref_images) <= 1 else 320
                    input_ref_images = [preprocess_ref(img, ref_long_side) for img in input_ref_images]
                    lock_dimensions = True

                elif flux_kontext_dreamomni2:
                    for i, img in enumerate(input_ref_images):
                        input_ref_images[i] = resizeinput(img)
                    input_prompt= self.infer_vlm(input_ref_images,input_prompt, " It is editing task." if "K"  in video_prompt_type else " It is generation task." )
                    input_prompt = input_prompt[6:-7]
                    print(input_prompt)
                    lock_dimensions = True

                ref_style_imgs = []
                if "I" in video_prompt_type and len(input_ref_images) > 0: 
                    if flux_dev_uso :
                        if "J" in video_prompt_type:
                            ref_style_imgs = input_ref_images
                            input_ref_images = []
                        elif len(input_ref_images) > 1 :
                            ref_style_imgs = input_ref_images[-1:]
                            input_ref_images = input_ref_images[:-1]

                    if latent_stiching:
                        # latents stiching with resize 
                        if not lock_dimensions :
                            for i in range(len(input_ref_images)):
                                w, h = input_ref_images[i].size
                                image_height, image_width = calculate_new_dimensions(int(height*image_refs_relative_size/100), int(width*image_refs_relative_size/100), h, w, 0)
                                input_ref_images[i] = input_ref_images[i].resize((image_width, image_height), resample=Image.Resampling.LANCZOS) 
                    else:
                        # image stiching method
                        stiched = input_ref_images[0]
                        for new_img in input_ref_images[1:]:
                            stiched = stitch_images(stiched, new_img)
                        input_ref_images  = [stiched]
                elif input_frames is not None:
                    input_ref_images = [convert_tensor_to_image(input_frames) ] 
                else:
                    input_ref_images = None
                image_mask = None if input_masks is None else convert_tensor_to_image(input_masks, mask_levels= True) 

                noise_patch_size = self.model.patch_size if radiance else 2
                noise_channels = self.model.out_channels if radiance else 16

                if latent_stiching  :
                    inp, height, width = prepare_multi_ip(
                        ae=self.vae,
                        img_cond_list=input_ref_images,
                        target_width=width,
                        target_height=height,
                        bs=batch_size,
                        seed=seed,
                        device=device,
                        res_match_output= flux_dev_uso or flux_dev_umo,
                        pe = 'w' if flux_kontext_dreamomni2 else 'd',
                        set_cond_index = flux_kontext_dreamomni2,
                        conditions_zero_start= flux_kontext_dreamomni2
                    )
                else:
                    inp, height, width = prepare_kontext(
                        ae=self.vae,
                        img_cond_list=input_ref_images,
                        target_width=width,
                        target_height=height,
                        bs=batch_size,
                        seed=seed,
                        device=device,
                        img_mask=image_mask,
                        patch_size=noise_patch_size,
                        noise_channels=noise_channels,
                    )

                inp.update(prepare_prompt(self.t5, self.clip, batch_size, input_prompt))
                if guide_scale != 1:
                    inp.update(prepare_prompt(self.t5, self.clip, batch_size, n_prompt, neg = True, device=device))

                timesteps = get_schedule(sampling_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

                ref_style_imgs = [self.vision_encoder_processor(img, return_tensors="pt").to(self.device) for img in ref_style_imgs]
                if self.feature_embedder is not None and ref_style_imgs is not None and len(ref_style_imgs) > 0 and self.vision_encoder is not None:
                    # processing style feat into textural hidden space
                    siglip_embedding = [self.vision_encoder(**emb, output_hidden_states=True) for emb in ref_style_imgs]
                    siglip_embedding = torch.cat([self.feature_embedder(emb) for emb in siglip_embedding], dim=1)
                    siglip_embedding_ids = torch.zeros( siglip_embedding.shape[0], siglip_embedding.shape[1], 3 ).to(device)
                    inp["siglip_embedding"] = siglip_embedding
                    inp["siglip_embedding_ids"] = siglip_embedding_ids

                if radiance:
                    def unpack_latent(x):
                        return patches_to_image(x.float(), height, width, noise_patch_size)
                else:
                    def unpack_latent(x):
                        return unpack(x.float(), height, width) 

            # denoise initial noise
            x = denoise(
                self.model,
                **inp,
                timesteps=timesteps,
                guidance=embedded_guidance_scale,
                real_guidance_scale=guide_scale,
                callback=callback,
                pipeline=self,
                loras_slists=loras_slists,
                unpack_latent=unpack_latent,
                joint_pass=joint_pass,
                denoising_strength=denoising_strength,
                masking_strength=masking_strength,
            )
            if x==None: return None
            # decode latents to pixel space
            x = unpack_latent(x)
            if self.vae is not None:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    x = self.vae.decode(x)

            if image_mask is not None and masking_strength == 1 and not flux2:
                img_msk_rebuilt = inp["img_msk_rebuilt"]
                img= input_frames.squeeze(1).unsqueeze(0) # convert_image_to_tensor(image_guide) 
                x = img * (1 - img_msk_rebuilt) + x.to(img) * img_msk_rebuilt 

            x = x.clamp(-1, 1)
            x = x.transpose(0, 1)
            return x

    def get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, video_prompt_type, **kwargs):
        if model_type != "flux_dev_kontext_dreamomni2": return [], []

        preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
        if len(preloadURLs) < 2: return [], []
        edit = "K" in video_prompt_type
        return [ fl.locate_file(os.path.basename(preloadURLs[0 if edit else 1]))] , [1]
