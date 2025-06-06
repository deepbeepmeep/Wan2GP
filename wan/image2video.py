# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import json
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.posemb_layers import get_rotary_pos_embed
from wan.utils.utils import resize_lanczos, calculate_new_dimensions

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star



class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        model_filename ="",
        text_encoder_filename="",
        quantizeTransformer = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        self.device = torch.device(f"cuda")
        self.config = config
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        # shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), dtype = VAE_dtype,
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir , 
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir ,  config.clip_tokenizer))

        logging.info(f"Creating WanModel from {model_filename[-1]}")
        from mmgp import offload

        # fantasy = torch.load("c:/temp/fantasy.ckpt")
        # proj_model = fantasy["proj_model"]
        # audio_processor = fantasy["audio_processor"]
        # offload.safetensors2.torch_write_file(proj_model, "proj_model.safetensors")
        # offload.safetensors2.torch_write_file(audio_processor, "audio_processor.safetensors")
        # for k,v in audio_processor.items():
        #     audio_processor[k] = v.to(torch.bfloat16)
        # with open("fantasy_config.json", "r", encoding="utf-8") as reader:
        #     config_text = reader.read()
        # config_json = json.loads(config_text)
        # offload.safetensors2.torch_write_file(audio_processor, "audio_processor_bf16.safetensors", config=config_json)
        # model_filename = [model_filename, "audio_processor_bf16.safetensors"] 
        # model_filename = "c:/temp/i2v480p/diffusion_pytorch_model-00001-of-00007.safetensors"
        # dtype = torch.float16
        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False) #, forcedConfigPath= "c:/temp/i2v720p/config.json")
        self.model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
        offload.change_dtype(self.model, dtype, True)
        # offload.save_model(self.model, "wan2.1_image2video_720p_14B_mbf16.safetensors", config_file_path="c:/temp/i2v720p/config.json")
        # offload.save_model(self.model, "wan2.1_image2video_720p_14B_quanto_mbf16_int8.safetensors",do_quantize=True, config_file_path="c:/temp/i2v720p/config.json")
        # offload.save_model(self.model, "wan2.1_image2video_720p_14B_quanto_mfp16_int8.safetensors",do_quantize=True, config_file_path="c:/temp/i2v720p/config.json")

        # offload.save_model(self.model, "wan2.1_Fun_InP_1.3B_bf16_bis.safetensors")
        self.model.eval().requires_grad_(False)


        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
        input_prompt,
        image_start,
        image_end = None,
        height =720,
        width = 1280,
        fit_into_canvas = True,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        callback = None,
        enable_RIFLEx = False,
        VAE_tile_size= 0,
        joint_pass = False,
        slg_layers = None,
        slg_start = 0.0,
        slg_end = 1.0,
        cfg_star_switch = True,
        cfg_zero_step = 5,
        audio_scale=None,
        audio_cfg_scale=None,
        audio_proj=None,
        audio_context_lens=None,
        model_filename = None,
        **bbargs
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            image_start (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        add_frames_for_end_image = "image2video" in model_filename or "fantasy" in model_filename

        image_start = TF.to_tensor(image_start)
        lat_frames = int((frame_num - 1) // self.vae_stride[0] + 1)
        any_end_frame = image_end !=None 
        if any_end_frame:
            any_end_frame = True
            image_end = TF.to_tensor(image_end) 
            if add_frames_for_end_image:
                frame_num +=1
                lat_frames = int((frame_num - 2) // self.vae_stride[0] + 2)
        
        h, w = image_start.shape[1:]

        h, w = calculate_new_dimensions(height, width, h, w, fit_into_canvas)
 
        lat_h = round(
            h // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            w // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        
        clip_image_size = self.clip.model.image_size
        img_interpolated = resize_lanczos(image_start, h, w).sub_(0.5).div_(0.5).unsqueeze(0).transpose(0,1).to(self.device) #, self.dtype
        image_start = resize_lanczos(image_start, clip_image_size, clip_image_size)
        image_start = image_start.sub_(0.5).div_(0.5).to(self.device) #, self.dtype
        if image_end!= None:
            img_interpolated2 = resize_lanczos(image_end, h, w).sub_(0.5).div_(0.5).unsqueeze(0).transpose(0,1).to(self.device) #, self.dtype
            image_end = resize_lanczos(image_end, clip_image_size, clip_image_size)
            image_end = image_end.sub_(0.5).div_(0.5).to(self.device) #, self.dtype

        max_seq_len = lat_frames * lat_h * lat_w // ( self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_frames, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)        

        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        if any_end_frame:
            msk[:, 1: -1] = 0
            if add_frames_for_end_image:
                msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:-1], torch.repeat_interleave(msk[:, -1:], repeats=4, dim=1) ], dim=1)
            else:
                msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)

        else:
            msk[:, 1:] = 0
            msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if self._interrupt:
            return None

        # preprocess
        context = self.text_encoder([input_prompt], self.device)[0]
        context_null = self.text_encoder([n_prompt], self.device)[0]
        context  = context.to(self.dtype)
        context_null  = context_null.to(self.dtype)

        if self._interrupt:
            return None

        clip_context = self.clip.visual([image_start[:, None, :, :]])

        from mmgp import offload
        offload.last_offload_obj.unload_all()
        if any_end_frame:
            mean2 = 0
            enc= torch.concat([
                    img_interpolated,
                    torch.full( (3, frame_num-2,  h, w), mean2, device=self.device, dtype= self.VAE_dtype),
                    img_interpolated2,
            ], dim=1).to(self.device)
        else:
            enc= torch.concat([
                    img_interpolated,
                    torch.zeros(3, frame_num-1, h, w, device=self.device, dtype= self.VAE_dtype)
            ], dim=1).to(self.device)
        image_start, image_end, img_interpolated, img_interpolated2 = None, None, None, None

        lat_y = self.vae.encode([enc], VAE_tile_size, any_end_frame= any_end_frame and add_frames_for_end_image)[0]
        y = torch.concat([msk, lat_y])
        lat_y = None


        # evaluation mode

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # sample videos
        latent = noise
        batch_size  = 1
        freqs = get_rotary_pos_embed(latent.shape[1:],  enable_RIFLEx= enable_RIFLEx) 

        kwargs = {  'clip_fea': clip_context, 'y': y, 'freqs' : freqs, 'pipeline' : self, 'callback' : callback }

        if audio_proj != None:
            kwargs.update({
            "audio_proj": audio_proj.to(self.dtype),
            "audio_context_lens": audio_context_lens,
            }) 

        if self.model.enable_teacache:
            self.model.previous_residual = [None] * (3 if audio_cfg_scale !=None else 2)
            self.model.compute_teacache_threshold(self.model.teacache_start_step, timesteps, self.model.teacache_multiplier)

        # self.model.to(self.device)
        if callback != None:
            callback(-1, None, True)
        latent = latent.to(self.device)
        for i, t in enumerate(tqdm(timesteps)):
            offload.set_step_no_for_lora(self.model, i)
            kwargs["slg_layers"] = slg_layers if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps) else None
            latent_model_input = latent
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)
            kwargs.update({
                't' :timestep,
                'current_step' :i,
            })
              
            if guide_scale == 1:
                noise_pred = self.model( [latent_model_input], context=[context], audio_scale = None if audio_scale == None else [audio_scale], x_id=0, **kwargs, )[0]
                if self._interrupt:
                    return None      
            elif joint_pass:
                if audio_proj == None:
                    noise_pred_cond, noise_pred_uncond = self.model(
                        [latent_model_input, latent_model_input],
                        context=[context, context_null],
                        **kwargs)
                else:
                    noise_pred_cond, noise_pred_noaudio, noise_pred_uncond = self.model(
                        [latent_model_input, latent_model_input, latent_model_input],
                        context=[context, context, context_null],
                        audio_scale = [audio_scale, None, None ],
                        **kwargs)

                if self._interrupt:
                    return None                
            else:
                noise_pred_cond = self.model( [latent_model_input], context=[context], audio_scale = None if audio_scale == None else [audio_scale], x_id=0, **kwargs, )[0]
                if self._interrupt:
                    return None
                
                if audio_proj != None:
                    noise_pred_noaudio = self.model(
                        [latent_model_input],
                        x_id=1,
                        context=[context],
                        **kwargs,
                    )[0]
                    if self._interrupt:
                        return None

                noise_pred_uncond = self.model(
                    [latent_model_input],
                    x_id=1 if audio_scale == None else 2,
                    context=[context_null],
                    **kwargs,
                )[0]
                if self._interrupt:
                    return None                
            del latent_model_input

            if guide_scale > 1:
                # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
                if cfg_star_switch:
                    positive_flat = noise_pred_cond.view(batch_size, -1)  
                    negative_flat = noise_pred_uncond.view(batch_size, -1)  

                    alpha = optimized_scale(positive_flat,negative_flat)
                    alpha = alpha.view(batch_size, 1, 1, 1)

                    if (i <= cfg_zero_step):
                        noise_pred = noise_pred_cond*0.  # it would be faster not to compute noise_pred...
                    else:
                        noise_pred_uncond *= alpha
                if audio_scale == None:
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)            
                else:
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_noaudio - noise_pred_uncond) + audio_cfg_scale * (noise_pred_cond  - noise_pred_noaudio) 
                              
            noise_pred_uncond, noise_pred_noaudio = None, None
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            del temp_x0
            del timestep

            if callback is not None:
                callback(i, latent, False) 

        x0 = [latent]        
        video = self.vae.decode(x0, VAE_tile_size, any_end_frame= any_end_frame and add_frames_for_end_image)[0]

        if any_end_frame and add_frames_for_end_image:
            # video[:,  -1:] = img_interpolated2
            video = video[:,  :-1]  

        del noise, latent
        del sample_scheduler

        return video
