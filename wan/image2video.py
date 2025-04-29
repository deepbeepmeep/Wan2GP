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
from wan.utils.utils import resize_lanczos

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
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {model_filename}")
        from mmgp import offload

        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False)
        self.model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype, True)
        offload.change_dtype(self.model, dtype, True)
        # offload.save_model(self.model, "i2v_720p_fp16.safetensors",do_quantize=True)

        # offload.save_model(self.model, "wan2.1_Fun_InP_1.3B_bf16_bis.safetensors")
        self.model.eval().requires_grad_(False)


        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
        input_prompt,
        img,
        img2 = None,
        max_area=720 * 1280,
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
        add_frames_for_end_image = True
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
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
        img = TF.to_tensor(img)
        lat_frames = int((frame_num - 1) // self.vae_stride[0] + 1)
        any_end_frame = img2 !=None 
        if any_end_frame:
            any_end_frame = True
            img2 = TF.to_tensor(img2) 
            if add_frames_for_end_image:
                frame_num +=1
                lat_frames = int((frame_num - 2) // self.vae_stride[0] + 2)
                
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        clip_image_size = self.clip.model.image_size
        img_interpolated = resize_lanczos(img, h, w).sub_(0.5).div_(0.5).unsqueeze(0).transpose(0,1).to(self.device) #, self.dtype
        img = resize_lanczos(img, clip_image_size, clip_image_size)
        img = img.sub_(0.5).div_(0.5).to(self.device) #, self.dtype
        if img2!= None:
            img_interpolated2 = resize_lanczos(img2, h, w).sub_(0.5).div_(0.5).unsqueeze(0).transpose(0,1).to(self.device) #, self.dtype
            img2 = resize_lanczos(img2, clip_image_size, clip_image_size)
            img2 = img2.sub_(0.5).div_(0.5).to(self.device) #, self.dtype

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

        clip_context = self.clip.visual([img[:, None, :, :]])

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
        img, img2, img_interpolated, img_interpolated2 = None, None, None, None

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
        batch_size  = latent.shape[0]
        freqs = get_rotary_pos_embed(latent.shape[1:],  enable_RIFLEx= enable_RIFLEx) 

        arg_c = {
            'context': [context],
            'clip_fea': clip_context,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self,
            'callback' : callback
        }

        arg_null = {
            'context': [context_null],
            'clip_fea': clip_context,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self,
            'callback' : callback
        }

        arg_both= {
            'context': [context, context_null],
            'clip_fea': clip_context,
            'y': [y],
            'freqs' : freqs,
            'pipeline' : self,
            'callback' : callback
        }

        if self.model.enable_teacache:
            self.model.compute_teacache_threshold(self.model.teacache_start_step, timesteps, self.model.teacache_multiplier)

        # self.model.to(self.device)
        if callback != None:
            callback(-1, None, True)

        for i, t in enumerate(tqdm(timesteps)):
            offload.set_step_no_for_lora(self.model, i)
            slg_layers_local = None
            if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps):
                slg_layers_local = slg_layers

            latent_model_input = [latent.to(self.device)]
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)
            if joint_pass:
                noise_pred_cond, noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, current_step=i, slg_layers=slg_layers_local, **arg_both)
                if self._interrupt:
                    return None                
            else:
                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    current_step=i,
                    is_uncond=False,
                    **arg_c,
                )[0]
                if self._interrupt:
                    return None                
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    current_step=i,
                    is_uncond=True,
                    slg_layers=slg_layers_local,
                    **arg_null,
                )[0]
                if self._interrupt:
                    return None                
            del latent_model_input

            # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
            noise_pred_text = noise_pred_cond
            if cfg_star_switch:
                positive_flat = noise_pred_text.view(batch_size, -1)  
                negative_flat = noise_pred_uncond.view(batch_size, -1)  

                alpha = optimized_scale(positive_flat,negative_flat)
                alpha = alpha.view(batch_size, 1, 1, 1)


                if (i <= cfg_zero_step):
                    noise_pred = noise_pred_text*0.  # it would be faster not to compute noise_pred...
                else:
                    noise_pred_uncond *= alpha
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_text - noise_pred_uncond)            

            del noise_pred_uncond

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


        # x0 = [latent.to(self.device, dtype=self.dtype)]

        x0 = [latent]

        # x0 = [lat_y]
        video = self.vae.decode(x0, VAE_tile_size, any_end_frame= any_end_frame and add_frames_for_end_image)[0]

        if any_end_frame and add_frames_for_end_image:
            # video[:,  -1:] = img_interpolated2
            video = video[:,  :-1]  

        del noise, latent
        del sample_scheduler

        return video
