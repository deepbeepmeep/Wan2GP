import math
import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import logging
import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from wan.modules.posemb_layers import get_rotary_pos_embed
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

class DTT2V:


    def __init__(
        self,
        config,
        checkpoint_dir,
        rank=0,
        model_filename = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        dtype = torch.bfloat16,
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
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {model_filename}")
        from mmgp import offload

        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False, forcedConfigPath="config.json")
        # offload.load_model_data(self.model, "recam.ckpt")
        # self.model.cpu()
        # offload.save_model(self.model, "recam.safetensors")
        if self.dtype == torch.float16 and not "fp16" in model_filename:
            self.model.to(self.dtype) 
        # offload.save_model(self.model, "t2v_fp16.safetensors",do_quantize=True)
        if self.dtype == torch.float16:
            self.vae.model.to(self.dtype)
        self.model.eval().requires_grad_(False)

        self.scheduler = FlowUniPCMultistepScheduler()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    def encode_image(
        self, image: PipelineImageInput, height: int, width: int, num_frames: int, tile_size = 0, causal_block_size = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # prefix_video
        prefix_video = np.array(image.resize((width, height))).transpose(2, 0, 1)
        prefix_video = torch.tensor(prefix_video).unsqueeze(1)  # .to(image_embeds.dtype).unsqueeze(1)
        if prefix_video.dtype == torch.uint8:
            prefix_video = (prefix_video.float() / (255.0 / 2.0)) - 1.0
        prefix_video = prefix_video.to(self.device)
        prefix_video = [self.vae.encode(prefix_video.unsqueeze(0), tile_size = tile_size)[0]]  # [(c, f, h, w)]
        if prefix_video[0].shape[1] % causal_block_size != 0:
            truncate_len = prefix_video[0].shape[1] % causal_block_size
            print("the length of prefix video is truncated for the casual block size alignment.")
            prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
        predix_video_latent_length = prefix_video[0].shape[1]
        return prefix_video, predix_video_latent_length

    def prepare_latents(
        self,
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        return randn_tensor(shape, generator, device=device, dtype=dtype)

    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        image: PipelineImageInput = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        shift: float = 1.0,
        guidance_scale: float = 5.0,
        seed: float = 0.0,
        overlap_history: int = 17,
        addnoise_condition: int = 0,
        base_num_frames: int = 97,
        ar_step: int = 5,
        causal_block_size: int = 1,
        causal_attention: bool = False,
        fps: int = 24,
        VAE_tile_size = 0,
        joint_pass = False,
        callback = None,
    ):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        # if base_num_frames > base_num_frames:
        #     causal_block_size = 0
        self._guidance_scale = guidance_scale

        i2v_extra_kwrags = {}
        prefix_video = None
        predix_video_latent_length = 0
        if image:
            frame_width, frame_height  = image.size
            scale = min(height / frame_height, width /  frame_width)
            height = (int(frame_height * scale) // 16) * 16
            width = (int(frame_width * scale) // 16) * 16

            prefix_video, predix_video_latent_length = self.encode_image(image, height, width, num_frames, tile_size=VAE_tile_size, causal_block_size=causal_block_size)

        latent_length = (num_frames - 1) // 4 + 1
        latent_height = height // 8
        latent_width = width // 8

        prompt_embeds = self.text_encoder([prompt], self.device)
        prompt_embeds  = [u.to(self.dtype).to(self.device) for u in prompt_embeds]
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = self.text_encoder([negative_prompt], self.device)
            negative_prompt_embeds  = [u.to(self.dtype).to(self.device) for u in negative_prompt_embeds]



        self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
        init_timesteps = self.scheduler.timesteps
        fps_embeds = [fps] * prompt_embeds[0].shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]
        transformer_dtype = self.dtype
        # with torch.cuda.amp.autocast(dtype=self.dtype), torch.no_grad():
        if overlap_history is None or base_num_frames is None or num_frames <= base_num_frames:
            # short video generation
            latent_shape = [16, latent_length, latent_height, latent_width]
            latents = self.prepare_latents(
                latent_shape, dtype=torch.float32, device=self.device, generator=generator
            )
            latents = [latents]
            if prefix_video is not None:
                latents[0][:, :predix_video_latent_length] = prefix_video[0].to(torch.float32)
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                latent_length, init_timesteps, base_num_frames, ar_step, predix_video_latent_length, causal_block_size
            )
            sample_schedulers = []
            for _ in range(latent_length):
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
                sample_schedulers.append(sample_scheduler)
            sample_schedulers_counter = [0] * latent_length

            if callback != None:
                callback(-1, None, True)

            freqs = get_rotary_pos_embed(latents[0].shape[1:], enable_RIFLEx= False) 
            for i, timestep_i in enumerate(tqdm(step_matrix)):
                update_mask_i = step_update_mask[i]
                valid_interval_i = valid_interval[i]
                valid_interval_start, valid_interval_end = valid_interval_i
                timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                    noise_factor = 0.001 * addnoise_condition
                    timestep_for_noised_condition = addnoise_condition
                    latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] * (1.0 - noise_factor)
                        + torch.randn_like(latent_model_input[0][:, valid_interval_start:predix_video_latent_length])
                        * noise_factor
                    )
                    timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                kwrags = {
                    "x" : torch.stack([latent_model_input[0]]),
                    "t" : timestep,
                    "freqs" :freqs,
                    "fps" : fps_embeds,
                    # "causal_block_size" : causal_block_size,
                    "callback" : callback,
                    "pipeline" : self
                }
                kwrags.update(i2v_extra_kwrags)


                if not self.do_classifier_free_guidance:
                    noise_pred = self.model(
                        context=prompt_embeds,
                        **kwrags,
                    )[0]
                    if self._interrupt:
                        return None                
                    noise_pred= noise_pred.to(torch.float32)                                          
                else:
                    if joint_pass:
                        noise_pred_cond, noise_pred_uncond = self.model(
                            context=prompt_embeds,
                            context2=negative_prompt_embeds,
                            **kwrags,
                        )
                        if self._interrupt:
                            return None
                    else:
                        noise_pred_cond = self.model(
                            context=prompt_embeds,
                            **kwrags,
                        )[0]
                        if self._interrupt:
                            return None                
                        noise_pred_uncond = self.model(
                            context=negative_prompt_embeds,
                            **kwrags,
                        )[0]
                        if self._interrupt:
                            return None
                    noise_pred_cond= noise_pred_cond.to(torch.float32)                                                                                 
                    noise_pred_uncond= noise_pred_uncond.to(torch.float32)                                                                                 
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    del noise_pred_cond, noise_pred_uncond
                for idx in range(valid_interval_start, valid_interval_end):
                    if update_mask_i[idx].item():
                        latents[0][:, idx] = sample_schedulers[idx].step(
                            noise_pred[:, idx - valid_interval_start],
                            timestep_i[idx],
                            latents[0][:, idx],
                            return_dict=False,
                            generator=generator,
                        )[0]
                        sample_schedulers_counter[idx] += 1
                if callback is not None:
                    callback(i, latents[0], False)         

            x0 = latents[0].unsqueeze(0)
            videos = self.vae.decode(x0, tile_size= VAE_tile_size)
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
            return videos
        else:
            # long video generation
            base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
            overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = 1 + (latent_length - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1
            print(f"n_iter:{n_iter}")
            output_video = None
            for i in range(n_iter):
                if output_video is not None:  # i !=0
                    prefix_video = output_video[:, -overlap_history:].to(self.device)
                    prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
                    if prefix_video[0].shape[1] % causal_block_size != 0:
                        truncate_len = prefix_video[0].shape[1] % causal_block_size
                        print("the length of prefix video is truncated for the casual block size alignment.")
                        prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
                    predix_video_latent_length = prefix_video[0].shape[1]
                    finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                    left_frame_num = latent_length - finished_frame_num
                    base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
                else:  # i == 0
                    base_num_frames_iter = base_num_frames
                latent_shape = [16, base_num_frames_iter, latent_height, latent_width]
                latents = self.prepare_latents(
                    latent_shape, dtype=torch.float32, device=self.device, generator=generator
                )
                latents = [latents]
                if prefix_video is not None:
                    latents[0][:, :predix_video_latent_length] = prefix_video[0].to(torch.float32)
                step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                    base_num_frames_iter,
                    init_timesteps,
                    base_num_frames_iter,
                    ar_step,
                    predix_video_latent_length,
                    causal_block_size,
                )
                sample_schedulers = []
                for _ in range(base_num_frames_iter):
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                sample_schedulers_counter = [0] * base_num_frames_iter
                if callback != None:
                    callback(-1, None, True)

                freqs = get_rotary_pos_embed(latents[0].shape[1:], enable_RIFLEx= False) 
                for i, timestep_i in enumerate(tqdm(step_matrix)):
                    update_mask_i = step_update_mask[i]
                    valid_interval_i = valid_interval[i]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    latent_model_input = [latents[0][:, valid_interval_start:valid_interval_end, :, :].clone()]
                    if addnoise_condition > 0 and valid_interval_start < predix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[0][:, valid_interval_start:predix_video_latent_length] = (
                            latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            * (1.0 - noise_factor)
                            + torch.randn_like(
                                latent_model_input[0][:, valid_interval_start:predix_video_latent_length]
                            )
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
                    kwrags = {
                        "x" : torch.stack([latent_model_input[0]]),
                        "t" : timestep,
                        "freqs" :freqs,
                        "fps" : fps_embeds,
                        "causal_block_size" : causal_block_size,
                        "causal_attention" : causal_attention,
                        "callback" : callback,
                        "pipeline" : self
                    }
                    kwrags.update(i2v_extra_kwrags)
                        
                    if not self.do_classifier_free_guidance:
                        noise_pred = self.model(
                            context=prompt_embeds,
                            **kwrags,
                        )[0]
                        if self._interrupt:
                            return None
                        noise_pred= noise_pred.to(torch.float32)                                                                  
                    else:
                        if joint_pass:
                            noise_pred_cond, noise_pred_uncond = self.model(
                                context=prompt_embeds,
                                context2=negative_prompt_embeds,
                                **kwrags,
                            )
                            if self._interrupt:
                                return None                
                        else:
                            noise_pred_cond = self.model(
                                context=prompt_embeds,
                                **kwrags,
                            )[0]
                            if self._interrupt:
                                return None                
                            noise_pred_uncond = self.model(
                                context=negative_prompt_embeds,
                            )[0]
                            if self._interrupt:
                                return None
                        noise_pred_cond= noise_pred_cond.to(torch.float32)                                          
                        noise_pred_uncond= noise_pred_uncond.to(torch.float32)                                          
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        del noise_pred_cond, noise_pred_uncond
                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[0][:, idx] = sample_schedulers[idx].step(
                                noise_pred[:, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[0][:, idx],
                                return_dict=False,
                                generator=generator,
                            )[0]
                            sample_schedulers_counter[idx] += 1
                    if callback is not None:
                        callback(i, latents[0].squeeze(0), False)         

                x0 = latents[0].unsqueeze(0)
                videos = [self.vae.decode(x0, tile_size= VAE_tile_size)[0]]
                if output_video is None:
                    output_video = videos[0].clamp(-1, 1).cpu()  # c, f, h, w
                else:
                    output_video = torch.cat(
                        [output_video, videos[0][:, overlap_history:].clamp(-1, 1).cpu()], 1
                    )  # c, f, h, w
            return output_video
