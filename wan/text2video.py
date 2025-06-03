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
from wan.utils.basic_flowmatch import FlowMatchScheduler

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
        model_filename = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        self.device = torch.device(f"cuda")
        self.config = config
        self.rank = rank
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
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), dtype= VAE_dtype,
            device=self.device)
        
        logging.info(f"Creating WanModel from {model_filename[-1]}")
        from mmgp import offload
        # model_filename = "c:/temp/vace1.3/diffusion_pytorch_model.safetensors"
        # model_filename = "vace14B_quanto_bf16_int8.safetensors"
        # model_filename = "c:/temp/phantom/Phantom_Wan_14B-00001-of-00006.safetensors"
        # config_filename= "c:/temp/phantom/config.json"
        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False)#, forcedConfigPath= config_filename)
        # offload.load_model_data(self.model, "e:/vace.safetensors")
        # offload.load_model_data(self.model, "c:/temp/Phantom-Wan-1.3B.pth")
        # self.model.to(torch.bfloat16)
        # self.model.cpu()
        self.model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
        # dtype = torch.bfloat16
        offload.change_dtype(self.model, dtype, True)
        # offload.save_model(self.model, "wan2.1_phantom_14B_mbf16.safetensors", config_file_path=config_filename)
        # offload.save_model(self.model, "wan2.1_phantom_14B_quanto_fp16_int8.safetensors", do_quantize= True, config_file_path=config_filename)
        self.model.eval().requires_grad_(False)


        self.sample_neg_prompt = config.sample_neg_prompt

        if "Vace" in model_filename[-1]:
            self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
                                            min_area=480*832,
                                            max_area=480*832,
                                            min_fps=config.sample_fps,
                                            max_fps=config.sample_fps,
                                            zero_start=True,
                                            seq_len=32760,
                                            keep_last=True)

            self.adapt_vace_model()

    def vace_encode_frames(self, frames, ref_images, masks=None, tile_size = 0, overlapped_latents = None):
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
            self.toto = inactive[0].clone() 
            if overlapped_latents  != None  : 
                # inactive[0][:, 0:1] = self.vae.encode([frames[0][:, 0:1]], tile_size = tile_size)[0] # redundant
                inactive[0][:, 1:overlapped_latents.shape[1] + 1] = overlapped_latents

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

    def prepare_source(self, src_video, src_mask, src_ref_images, total_frames, image_size,  device, original_video = False, keep_frames= [], start_frame = 0,  fit_into_canvas = True, pre_src_video = None):
        image_sizes = []
        trim_video = len(keep_frames)
        canvas_height, canvas_width = image_size

        for i, (sub_src_video, sub_src_mask, sub_pre_src_video) in enumerate(zip(src_video, src_mask,pre_src_video)):
            prepend_count = 0 if sub_pre_src_video == None else sub_pre_src_video.shape[1]
            num_frames = total_frames - prepend_count 
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask, max_frames= num_frames, trim_video = trim_video - prepend_count, start_frame = start_frame, canvas_height = canvas_height, canvas_width = canvas_width, fit_into_canvas = fit_into_canvas)
                # src_video is [-1, 1] (at this function output), 0 = inpainting area (in fact 127  in [0, 255])
                # src_mask is [-1, 1] (at this function output), 0 = preserve original video (in fact 127  in [0, 255]) and 1 = Inpainting (in fact 255  in [0, 255])
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                if prepend_count > 0:
                    src_video[i] =  torch.cat( [sub_pre_src_video, src_video[i]], dim=1)
                    src_mask[i] =  torch.cat( [torch.full_like(sub_pre_src_video, -1.0), src_mask[i]] ,1)
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
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video, max_frames= num_frames, trim_video = trim_video - prepend_count, start_frame = start_frame, canvas_height = canvas_height, canvas_width = canvas_width, fit_into_canvas = fit_into_canvas)
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

    def get_vae_latents(self, ref_images, device, tile_size= 0):
        ref_vae_latents = []
        for ref_image in ref_images:
            ref_image = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(self.device)
            img_vae_latent = self.vae.encode([ref_image.unsqueeze(1)], tile_size= tile_size)
            ref_vae_latents.append(img_vae_latent[0])
                    
        return torch.cat(ref_vae_latents, dim=1)
        
    def generate(self,
                input_prompt,
                input_frames= None,
                input_masks = None,
                input_ref_images = None,      
                input_video=None,
                target_camera=None,                  
                context_scale=1.0,
                width = 1280,
                height = 720,
                fit_into_canvas = True,
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
                overlapped_latents  = None,
                return_latent_slice = None,
                overlap_noise = 0,
                conditioning_latents_size = 0,
                model_filename = None,
                **bbargs
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
        vace = "Vace" in model_filename

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if self._interrupt:
            return None
        context = self.text_encoder([input_prompt], self.device)[0]
        context_null = self.text_encoder([n_prompt], self.device)[0]
        context = context.to(self.dtype)
        context_null = context_null.to(self.dtype)
        input_ref_images_neg = None
        phantom = False

        if target_camera != None:
            width = input_video.shape[2]
            height = input_video.shape[1]
            input_video = input_video.to(dtype=self.dtype , device=self.device)
            input_video = input_video.permute(3, 0, 1, 2).div_(127.5).sub_(1.)            
            source_latents = self.vae.encode([input_video])[0] #.to(dtype=self.dtype, device=self.device)
            del input_video
            # Process target camera (recammaster)
            from wan.utils.cammmaster_tools import get_camera_embedding
            cam_emb = get_camera_embedding(target_camera)       
            cam_emb = cam_emb.to(dtype=self.dtype, device=self.device)

        if vace :
            # vace context encode
            input_frames = [u.to(self.device) for u in input_frames]
            input_ref_images = [ None if u == None else [v.to(self.device) for v in u]  for u in input_ref_images]
            input_masks = [u.to(self.device) for u in input_masks]
            previous_latents = None
            # if overlapped_latents != None:
                # input_ref_images = [u[-1:] for u in input_ref_images]
            z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, tile_size = VAE_tile_size, overlapped_latents = overlapped_latents )
            m0 = self.vace_encode_masks(input_masks, input_ref_images)
            z = self.vace_latent(z0, m0)

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
        else:
            if input_ref_images != None: # Phantom Ref images
                phantom = True
                input_ref_images = self.get_vae_latents(input_ref_images, self.device)
                input_ref_images_neg = torch.zeros_like(input_ref_images)
            F = frame_num
            target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1 + (input_ref_images.shape[1] if input_ref_images != None else 0),
                            height // self.vae_stride[1],
                            width // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1]) 

        if self._interrupt:
            return None

        noise = [ torch.randn( *target_shape, dtype=torch.float32, device=self.device, generator=seed_g) ]

        # evaluation mode

        if False:
            sample_scheduler = FlowMatchScheduler(num_inference_steps=sampling_steps, shift=shift, sigma_min=0, extra_one_step=True)
            timesteps = torch.tensor([1000, 934, 862, 756, 603, 410, 250, 140, 74, 0])[:sampling_steps].to(self.device)
            sample_scheduler.timesteps =timesteps
        elif sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler( num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps( sampling_steps, device=self.device, shift=shift)
            
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
        latents = noise[0]
        del noise
        batch_size = 1
        if target_camera != None:
            shape = list(latents.shape[1:])
            shape[0] *= 2
            freqs = get_rotary_pos_embed(shape, enable_RIFLEx= False) 
        else:
            freqs = get_rotary_pos_embed(latents.shape[1:], enable_RIFLEx= enable_RIFLEx) 

        kwargs = {'freqs': freqs, 'pipeline': self, 'callback': callback}

        if target_camera != None:
            kwargs.update({'cam_emb': cam_emb})

        if vace:
            ref_images_count = len(input_ref_images[0]) if input_ref_images != None and input_ref_images[0] != None else 0 
            kwargs.update({'vace_context' : z, 'vace_context_scale' : context_scale})
            if overlapped_latents != None :
                overlapped_latents_size = overlapped_latents.shape[1] + 1
                # overlapped_latents_size = 3
                z_reactive = [  zz[0:16, 0:overlapped_latents_size + ref_images_count].clone() for zz in z]


        if self.model.enable_teacache:
            x_count = 3 if phantom else 2
            self.model.previous_residual = [None] * x_count 
            self.model.compute_teacache_threshold(self.model.teacache_start_step, timesteps, self.model.teacache_multiplier)
        if callback != None:
            callback(-1, None, True)
        prev = 50/1000
        for i, t in enumerate(tqdm(timesteps)):

            timestep = [t]
            if overlapped_latents != None :
                # overlap_noise_factor = overlap_noise *(i/(len(timesteps)-1)) / 1000
                overlap_noise_factor = overlap_noise / 1000 
                # overlap_noise_factor = (1000-t )/ 1000  #  overlap_noise / 1000 
                # latent_noise_factor = 1 #max(min(1,  (t - overlap_noise)  / 1000 ),0)
                latent_noise_factor = t / 1000
                for zz, zz_r, ll in zip(z, z_reactive, [latents]):
                    pass
                    # zz[0:16, ref_images_count:overlapped_latents_size + ref_images_count]   = zz_r[:, ref_images_count:]  * (1.0 - overlap_noise_factor) + torch.randn_like(zz_r[:, ref_images_count:] ) * overlap_noise_factor 
                    # ll[:, 0:overlapped_latents_size + ref_images_count]   = zz_r  * (1.0 - latent_noise_factor) + torch.randn_like(zz_r ) * latent_noise_factor 

            if conditioning_latents_size > 0 and overlap_noise > 0:
                pass
                overlap_noise_factor = overlap_noise / 1000 
                latents[:, conditioning_latents_size + ref_images_count:]   = latents[:, conditioning_latents_size + ref_images_count:]  * (1.0 - overlap_noise_factor) + torch.randn_like(latents[:, conditioning_latents_size + ref_images_count:]) * overlap_noise_factor 
                #timestep = [torch.tensor([t.item()] * (conditioning_latents_size + ref_images_count) + [t.item() - overlap_noise]*(len(timesteps) - conditioning_latents_size - ref_images_count))]

            if target_camera != None:
                latent_model_input = torch.cat([latents, source_latents], dim=1)
            else:
                latent_model_input = latents
            kwargs["slg_layers"] = slg_layers if int(slg_start * sampling_steps) <= i < int(slg_end * sampling_steps) else None

            offload.set_step_no_for_lora(self.model, i)
            timestep = torch.stack(timestep)
            kwargs["current_step"] = i 
            kwargs["t"] = timestep 
            if guide_scale == 1:
                noise_pred = self.model( [latent_model_input], x_id = 0, context = [context], **kwargs)[0]
                if self._interrupt:
                    return None
            elif joint_pass:
                if phantom:
                    pos_it, pos_i, neg = self.model(
                         [ torch.cat([latent_model_input[:,:-input_ref_images.shape[1]], input_ref_images], dim=1) ] * 2 +
                         [ torch.cat([latent_model_input[:,:-input_ref_images_neg.shape[1]], input_ref_images_neg], dim=1)],
                        context = [context, context_null, context_null], **kwargs)
                else:
                    noise_pred_cond, noise_pred_uncond = self.model(
                        [latent_model_input, latent_model_input], context = [context, context_null], **kwargs)
                if self._interrupt:
                    return None
            else:
                if phantom:
                    pos_it = self.model(
                        [ torch.cat([latent_model_input[:,:-input_ref_images.shape[1]], input_ref_images], dim=1) ], x_id = 0, context = [context], **kwargs
                        )[0]
                    if self._interrupt:
                        return None               
                    pos_i = self.model(
                        [ torch.cat([latent_model_input[:,:-input_ref_images.shape[1]], input_ref_images], dim=1) ], x_id = 1, context = [context_null],**kwargs
                        )[0]
                    if self._interrupt:
                        return None               
                    neg = self.model(
                           [ torch.cat([latent_model_input[:,:-input_ref_images_neg.shape[1]], input_ref_images_neg], dim=1) ], x_id = 2, context = [context_null], **kwargs
                        )[0]
                    if self._interrupt:
                        return None               
                else:
                    noise_pred_cond = self.model(
                        [latent_model_input], x_id = 0, context = [context], **kwargs)[0]
                    if self._interrupt:
                        return None               
                    noise_pred_uncond = self.model(
                        [latent_model_input], x_id = 1, context = [context_null], **kwargs)[0]
                    if self._interrupt:
                        return None

            # del latent_model_input

            # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
            if guide_scale == 1:
                pass
            elif phantom:
                guide_scale_img= 5.0
                guide_scale_text= guide_scale #7.5                
                noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i)
            else:
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
            noise_pred_uncond, noise_pred_cond, noise_pred_text, pos_it, pos_i, neg  = None, None, None, None, None, None
            scheduler_kwargs = {} if isinstance(sample_scheduler, FlowMatchScheduler) else {"generator": seed_g}
            temp_x0 = sample_scheduler.step(
                noise_pred[:, :target_shape[1]].unsqueeze(0),
                t,
                latents.unsqueeze(0),
                # return_dict=False,
                **scheduler_kwargs)[0]
            latents = temp_x0.squeeze(0)
            del temp_x0

            if callback is not None:
                callback(i, latents, False)         

        x0 = [latents]

        if return_latent_slice != None:
            if overlapped_latents != None:
                # latents [:, 1:] = self.toto
                for zz, zz_r, ll  in zip(z, z_reactive, [latents]):
                    ll[:, 0:overlapped_latents_size + ref_images_count]   = zz_r 

            latent_slice = latents[:, return_latent_slice].clone()
        if input_frames == None:
            if phantom:
                # phantom post processing
                x0 = [x0_[:,:-input_ref_images.shape[1]] for x0_ in x0]
            videos = self.vae.decode(x0, VAE_tile_size)
        else:
            # vace post processing
            videos = self.decode_latent(x0, input_ref_images, VAE_tile_size)
        if return_latent_slice != None:
            return { "x" : videos[0], "latent_slice" : latent_slice }
        return videos[0]

    def adapt_vace_model(self):
        model = self.model
        modules_dict= { k: m for k, m in model.named_modules()}
        for model_layer, vace_layer in model.vace_layers_mapping.items():
            module = modules_dict[f"vace_blocks.{vace_layer}"]
            target = modules_dict[f"blocks.{model_layer}"]
            setattr(target, "vace", module )
        delattr(model, "vace_blocks")

 
