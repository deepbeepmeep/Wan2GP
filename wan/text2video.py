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
from mmgp import offload
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.posemb_layers import get_rotary_pos_embed
from .utils.vace_preprocessor import VaceVideoProcessor


def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star
    

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        model_filename = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        dtype = torch.bfloat16
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.dtype = dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn= None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size 

        
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {model_filename}")
        from mmgp import offload

        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False)
        # offload.load_model_data(self.model, "recam.ckpt")
        # self.model.cpu()
        # offload.save_model(self.model, "recam.safetensors")
        if self.dtype == torch.float16 and not "fp16" in model_filename:
            self.model.to(self.dtype) 
        # offload.save_model(self.model, "t2v_fp16.safetensors",do_quantize=True)
        if self.dtype == torch.float16:
            self.vae.model.to(self.dtype)
        self.model.eval().requires_grad_(False)


        self.sample_neg_prompt = config.sample_neg_prompt

        if "Vace" in model_filename:
            self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
                                            min_area=480*832,
                                            max_area=480*832,
                                            min_fps=config.sample_fps,
                                            max_fps=config.sample_fps,
                                            zero_start=True,
                                            seq_len=32760,
                                            keep_last=True)

            self.adapt_vace_model()

    def vace_encode_frames(self, frames, ref_images, masks=None, tile_size = 0):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, tile_size = tile_size)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, tile_size = tile_size)
            reactive = self.vae.encode(reactive, tile_size = tile_size)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                else:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                self.vae_stride[1] * self.vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, total_frames, image_size,  device, original_video = False, keep_frames= [], start_frame = 0, pre_src_video = None):
        image_sizes = []
        trim_video = len(keep_frames)

        for i, (sub_src_video, sub_src_mask, sub_pre_src_video) in enumerate(zip(src_video, src_mask,pre_src_video)):
            prepend_count = 0 if sub_pre_src_video == None else sub_pre_src_video.shape[1]
            num_frames = total_frames - prepend_count 
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask, max_frames= num_frames, trim_video = trim_video - prepend_count, start_frame = start_frame)
                # src_video is [-1, 1], 0 = inpainting area (in fact 127  in [0, 255])
                # src_mask is [-1, 1], 0 = preserve original video (in fact 127  in [0, 255]) and 1 = Inpainting (in fact 255  in [0, 255])
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, src_video[i]], dim=1)
                    src_mask[i] =  torch.cat( [torch.zeros_like(sub_pre_src_video), src_mask[i]] ,1)
                src_video_shape = src_video[i].shape
                if src_video_shape[1] != total_frames:
                    src_video[i] =  torch.cat( [src_video[i], src_video[i].new_zeros(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                    src_mask[i] =  torch.cat( [src_mask[i], src_mask[i].new_ones(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)], dim=1)
                    src_mask[i] =  torch.cat( [torch.zeros_like(sub_pre_src_video), torch.ones((3, num_frames, image_size[0], image_size[1]), device=device)] ,1)
                else:
                    src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                    src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video, max_frames= num_frames, trim_video = trim_video - prepend_count, start_frame = start_frame)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.zeros_like(src_video[i], device=device) if original_video else torch.ones_like(src_video[i], device=device)
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, src_video[i]], dim=1)
                    src_mask[i] =  torch.cat( [torch.zeros_like(sub_pre_src_video), src_mask[i]] ,1)
                src_video_shape = src_video[i].shape
                if src_video_shape[1] != total_frames:
                    src_video[i] =  torch.cat( [src_video[i], src_video[i].new_zeros(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                    src_mask[i] =  torch.cat( [src_mask[i], src_mask[i].new_ones(src_video_shape[0], total_frames -src_video_shape[1], *src_video_shape[-2:])], dim=1)
                image_sizes.append(src_video[i].shape[2:])
            for k, keep in enumerate(keep_frames):
                if not keep:
                    src_video[i][:, k:k+1] = 0
                    src_mask[i][:, k:k+1] = 1

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    def decode_latent(self, zs, ref_images=None, tile_size= 0 ):
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return self.vae.decode(trimed_zs, tile_size= tile_size)

    def generate(self,
                input_prompt,
                input_frames= None,
                input_masks = None,
                input_ref_images = None,      
                source_video=None,
                target_camera=None,                  
                context_scale=1.0,
                size=(1280, 720),
                frame_num=81,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=5.0,
                n_prompt="",
                seed=-1,
                offload_model=True,
                callback = None,
                enable_RIFLEx = None,
                VAE_tile_size = 0,
                joint_pass = False,
                slg_layers = None,
                slg_start = 0.0,
                slg_end = 1.0,
                cfg_star_switch = True,
                cfg_zero_step = 5,
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if target_camera != None:
            size = (source_video.shape[2], source_video.shape[1])
            source_video = source_video.to(dtype=self.dtype , device=self.device)
            source_video = source_video.permute(3, 0, 1, 2).div_(127.5).sub_(1.)            
            source_latents = self.vae.encode([source_video]) #.to(dtype=self.dtype, device=self.device)
            del source_video
            # Process target camera (recammaster)
            from wan.utils.cammmaster_tools import get_camera_embedding
            cam_emb = get_camera_embedding(target_camera)       
            cam_emb = cam_emb.to(dtype=self.dtype, device=self.device)

        if input_frames != None:
            # vace context encode
            input_frames = [u.to(self.device) for u in input_frames]
            input_ref_images = [ None if u == None else [v.to(self.device) for v in u]  for u in input_ref_images]
            input_masks = [u.to(self.device) for u in input_masks]

            z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, tile_size = VAE_tile_size)
            m0 = self.vace_encode_masks(input_masks, input_ref_images)
            z = self.vace_latent(z0, m0)

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
        else:
            F = frame_num
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                            size[1] // self.vae_stride[1],
                            size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1]) 

        context  = [u.to(self.dtype) for u in context]
        context_null  = [u.to(self.dtype) for u in context_null]

        noise = [ torch.randn( *target_shape, dtype=torch.float32, device=self.device, generator=seed_g) ]

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
        latents = noise
        del noise
        batch_size =len(latents)
        if target_camera != None:
            shape = list(latents[0].shape[1:])
            shape[0] *= 2
            freqs = get_rotary_pos_embed(shape, enable_RIFLEx= False) 
        else:
            freqs = get_rotary_pos_embed(latents[0].shape[1:], enable_RIFLEx= enable_RIFLEx) 
        arg_c = {'context': context, 'freqs': freqs, 'pipeline': self}
        arg_null = {'context': context_null, 'freqs': freqs, 'pipeline': self}
        arg_both = {'context': context, 'context2': context_null,  'freqs': freqs, 'pipeline': self}

        if target_camera != None:
            recam_dict = {'cam_emb': cam_emb}
            arg_c.update(recam_dict)
            arg_null.update(recam_dict)
            arg_both.update(recam_dict)

        if input_frames != None:
            vace_dict = {'vace_context' : z, 'vace_context_scale' : context_scale}
            arg_c.update(vace_dict)
            arg_null.update(vace_dict)
            arg_both.update(vace_dict)

        if self.model.enable_teacache:
            self.model.compute_teacache_threshold(self.model.teacache_start_step, timesteps, self.model.teacache_multiplier)
        if callback != None:
            callback(-1, True)
        for i, t in enumerate(tqdm(timesteps)):
            if target_camera != None:
                latent_model_input = [torch.cat([u,v], dim=1) for u,v in zip(latents,source_latents )]
            else:
                latent_model_input = latents
            slg_layers_local = None
            if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps):
                slg_layers_local = slg_layers
            timestep = [t]
            offload.set_step_no_for_lora(self.model, i)
            timestep = torch.stack(timestep)

            if joint_pass:
                noise_pred_cond, noise_pred_uncond = self.model(
                    latent_model_input, t=timestep,  current_step=i, slg_layers=slg_layers_local, **arg_both)
                if self._interrupt:
                    return None
            else:
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep,current_step=i, is_uncond = False, **arg_c)[0]
                if self._interrupt:
                    return None               
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep,current_step=i, is_uncond = True, slg_layers=slg_layers_local, **arg_null)[0]
                if self._interrupt:
                    return None

            # del latent_model_input

            # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
            noise_pred_text = noise_pred_cond
            if cfg_star_switch:
                positive_flat = noise_pred_text.view(batch_size, -1)  
                negative_flat = noise_pred_uncond.view(batch_size, -1)  

                alpha = optimized_scale(positive_flat,negative_flat)
                alpha = alpha.view(batch_size, 1, 1, 1)

                if (i <= cfg_zero_step):
                    noise_pred = noise_pred_text*0. # it would be faster not to compute noise_pred...
                else:
                    noise_pred_uncond *= alpha
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_text - noise_pred_uncond)            
            del noise_pred_uncond

            temp_x0 = sample_scheduler.step(
                noise_pred[:, :target_shape[1]].unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]
            del temp_x0

            if callback is not None:
                callback(i, False)         

        x0 = latents

        if input_frames == None:
            videos = self.vae.decode(x0, VAE_tile_size)
        else:
            videos = self.decode_latent(x0, input_ref_images, VAE_tile_size)

        del latents
        del sample_scheduler

        return videos[0] if self.rank == 0 else None

    def adapt_vace_model(self):
        model = self.model
        modules_dict= { k: m for k, m in model.named_modules()}
        for model_layer, vace_layer in model.vace_layers_mapping.items():
            module = modules_dict[f"vace_blocks.{vace_layer}"]
            target = modules_dict[f"blocks.{model_layer}"]
            setattr(target, "vace", module )
        delattr(model, "vace_blocks")
                    
 