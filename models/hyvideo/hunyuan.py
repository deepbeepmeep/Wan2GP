import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

from pathlib import Path
from einops import rearrange
import torch
import torch.distributed as dist
from .constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE, NEGATIVE_PROMPT_I2V
from .vae import load_vae
from .modules import load_model
from .text_encoder import TextEncoder
from .utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from .modules.posemb_layers import get_nd_rotary_pos_embed, get_nd_rotary_pos_embed_new 
from .diffusion.schedulers import FlowMatchDiscreteScheduler
from .diffusion.pipelines import HunyuanVideoPipeline
from .diffusion.pipelines import HunyuanVideoAudioPipeline
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from shared.utils.utils import calculate_new_dimensions, convert_tensor_to_image
from .data_kits.audio_preprocessor import encode_audio, get_facemask
from transformers import WhisperModel
from transformers import AutoFeatureExtractor
from .data_kits.face_align import AlignImage
import librosa

def get_audio_feature(feature_extractor, audio_path, duration):
    audio_input, sampling_rate = librosa.load(audio_path, duration=duration, sr=16000)
    assert sampling_rate == 16000

    audio_features = []
    window = 750*640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i+window], 
                                        sampling_rate=sampling_rate, 
                                        return_tensors="pt", 
                                        device="cuda"
                                        ).input_features
        audio_features.append(audio_feature)

    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640

def pad_image(crop_img, size, color=(255, 255, 255), resize_ratio=1):
    crop_h, crop_w = crop_img.shape[:2]
    target_w, target_h = size
    scale_h, scale_w = target_h / crop_h, target_w / crop_w
    if scale_w > scale_h:
        resize_h = int(target_h*resize_ratio)
        resize_w = int(crop_w / crop_h * resize_h)
    else:
        resize_w = int(target_w*resize_ratio)
        resize_h = int(crop_h / crop_w * resize_w)
    crop_img = cv2.resize(crop_img, (resize_w, resize_h))
    pad_left = (target_w - resize_w) // 2
    pad_top = (target_h - resize_h) // 2
    pad_right = target_w - resize_w - pad_left
    pad_bottom = target_h - resize_h - pad_top
    crop_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return crop_img




# def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
#     num_images, num_image_patches, embed_dim = image_features.shape
#     batch_size, sequence_length = input_ids.shape
#     left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
#     # 1. Create a mask to know where special image tokens are
#     special_image_token_mask = input_ids == self.config.image_token_index
#     num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
#     # Compute the maximum embed dimension
#     max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
#     batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

#     # 2. Compute the positions where text should be written
#     # Calculate new positions for text tokens in merged image-text sequence.
#     # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
#     # `torch.cumsum` computes how each image token shifts subsequent text token positions.
#     # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
#     new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
#     nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
#     if left_padding:
#         new_token_positions += nb_image_pad[:, None]  # offset for left padding
#     text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

#     # 3. Create the full embedding, already padded to the maximum position
#     final_embedding = torch.zeros(
#         batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
#     )
#     final_attention_mask = torch.zeros(
#         batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
#     )
#     if labels is not None:
#         final_labels = torch.full(
#             (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
#         )
#     # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
#     # set the corresponding tensors into their correct target device.
#     target_device = inputs_embeds.device
#     batch_indices, non_image_indices, text_to_overwrite = (
#         batch_indices.to(target_device),
#         non_image_indices.to(target_device),
#         text_to_overwrite.to(target_device),
#     )
#     attention_mask = attention_mask.to(target_device)

#     # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
#     # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
#     final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
#     final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
#     if labels is not None:
#         final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

#     # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
#     image_to_overwrite = torch.full(
#         (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
#     )
#     image_to_overwrite[batch_indices, text_to_overwrite] = False
#     image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

#     if image_to_overwrite.sum() != image_features.shape[:-1].numel():
#         raise ValueError(
#             f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
#             f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
#         )

#     final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
#     final_attention_mask |= image_to_overwrite
#     position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

#     # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
#     batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
#     indices_to_mask = new_token_positions[batch_indices, pad_indices]

#     final_embedding[batch_indices, indices_to_mask] = 0

#     if labels is None:
#         final_labels = None

#     return final_embedding, final_attention_mask, final_labels, position_ids
    
# def patched_llava_forward(
#     self,
#     input_ids: torch.LongTensor = None,
#     pixel_values: torch.FloatTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     vision_feature_layer: Optional[int] = None,
#     vision_feature_select_strategy: Optional[str] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     num_logits_to_keep: int = 0,
# ):
#     from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast


#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#     vision_feature_layer = (
#         vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
#     )
#     vision_feature_select_strategy = (
#         vision_feature_select_strategy
#         if vision_feature_select_strategy is not None
#         else self.config.vision_feature_select_strategy
#     )

#     if (input_ids is None) ^ (inputs_embeds is not None):
#         raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#     if pixel_values is not None and inputs_embeds is not None:
#         raise ValueError(
#             "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
#         )

#     if inputs_embeds is None:
#         inputs_embeds = self.get_input_embeddings()(input_ids)

#     image_features = None
#     if pixel_values is not None:
#         image_features = self.get_image_features(
#             pixel_values=pixel_values,
#             vision_feature_layer=vision_feature_layer,
#             vision_feature_select_strategy=vision_feature_select_strategy,
#         )


#     inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
#         image_features, inputs_embeds, input_ids, attention_mask, labels
#     )
#     cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)


#     outputs = self.language_model(
#         attention_mask=attention_mask,
#         position_ids=position_ids,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#         num_logits_to_keep=num_logits_to_keep,
#     )

#     logits = outputs[0]

#     loss = None

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return LlavaCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         image_hidden_states=image_features if pixel_values is not None else None,
#     )

def adapt_model(model, audio_block_name):
    modules_dict= { k: m for k, m in model.named_modules()}
    for model_layer, avatar_layer in model.double_stream_map.items():
        module = modules_dict[f"{audio_block_name}.{avatar_layer}"]
        target = modules_dict[f"double_blocks.{model_layer}"]
        setattr(target, "audio_adapter", module )
    delattr(model, audio_block_name)

class DataPreprocess(object):
    def __init__(self):
        self.llava_size = (336, 336)
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def get_batch(self, image , size, pad = False):
        image = np.asarray(image)
        if pad:
            llava_item_image = pad_image(image.copy(), self.llava_size)
        else:
            llava_item_image = image.copy()
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255

        if pad:
            cat_item_image = pad_image(image.copy(), size)
        else:
            cat_item_image = image.copy()
        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0
        # batch = {
        #     "pixel_value_llava": llava_item_tensor.unsqueeze(0),
        #     "uncond_pixel_value_llava": uncond_llava_item_tensor.unsqueeze(0),
        #     'pixel_value_ref': cat_item_tensor.unsqueeze(0), 
        # }
        return llava_item_tensor.unsqueeze(0), uncond_llava_item_tensor.unsqueeze(0), cat_item_tensor.unsqueeze(0)

class Inference(object):
    def __init__(        
        self,
        i2v,
        custom,
        avatar,
        enable_cfg,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        feature_extractor=None,
        wav2vec=None,
        align_instance=None,
        device=None,
    ):
        self.i2v = i2v
        self.custom = custom
        self.avatar = avatar
        self.enable_cfg = enable_cfg
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline

        self.feature_extractor=feature_extractor
        self.wav2vec=wav2vec
        self.align_instance=align_instance

        self.device = "cuda"


    @classmethod
    def from_pretrained(cls, model_filepath, model_type, base_model_type, text_encoder_filepath,  dtype = torch.bfloat16, VAE_dtype = torch.float16, mixed_precision_transformer =torch.bfloat16 , quantizeTransformer = False, save_quantized = False, **kwargs):

        device = "cuda" 

        import transformers
        # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = patched_llava_forward # force legacy behaviour to be able to use tansformers v>(4.47)
        # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._merge_input_ids_with_image_features = _merge_input_ids_with_image_features

        torch.set_grad_enabled(False)
        text_len = 512
        latent_channels = 16
        precision = "bf16"
        vae_precision = "fp32" if VAE_dtype == torch.float32 else "bf16" 
        embedded_cfg_scale = 6
        filepath = model_filepath[0]
        i2v_condition_type = None
        i2v_mode = False
        custom = False
        custom_audio = False
        avatar = False 
        if base_model_type == "hunyuan_i2v":
            model_id = "HYVideo-T/2"
            i2v_condition_type = "token_replace"
            i2v_mode = True
        elif base_model_type == "hunyuan_custom":
            model_id = "HYVideo-T/2-custom"
            custom = True
        elif base_model_type == "hunyuan_custom_audio":
            model_id = "HYVideo-T/2-custom-audio"
            custom_audio = True
            custom = True
        elif base_model_type == "hunyuan_custom_edit":
            model_id = "HYVideo-T/2-custom-edit"
            custom = True
        elif base_model_type == "hunyuan_avatar":
            model_id = "HYVideo-T/2-avatar"
            text_len = 256
            avatar = True
        else:
            model_id = "HYVideo-T/2-cfgdistill"


        if i2v_mode and i2v_condition_type == "latent_concat":
            in_channels = latent_channels * 2 + 1
            image_embed_interleave = 2
        elif i2v_mode and i2v_condition_type == "token_replace":
            in_channels = latent_channels
            image_embed_interleave = 4
        else:
            in_channels = latent_channels
            image_embed_interleave = 1
        out_channels = latent_channels
        pinToMemory = kwargs.pop("pinToMemory", False)
        partialPinning = kwargs.pop("partialPinning", False)        
        factor_kwargs = kwargs | {"device": "meta", "dtype": PRECISION_TO_TYPE[precision]}

        if embedded_cfg_scale and i2v_mode:
            factor_kwargs["guidance_embed"] = True

        model = load_model(
            model = model_id,
            i2v_condition_type = i2v_condition_type,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )

  
        from mmgp import offload
        # model = Inference.load_state_dict(args, model, model_filepath)

        # model_filepath ="c:/temp/hc/mp_rank_00_model_states_video.pt"
        offload.load_model_data(model, model_filepath, do_quantize= quantizeTransformer and not save_quantized, pinToMemory = pinToMemory, partialPinning = partialPinning)
        pass
        # offload.save_model(model, "hunyuan_video_avatar_edit_720_bf16.safetensors")
        # offload.save_model(model, "hunyuan_video_avatar_edit_720_quanto_bf16_int8.safetensors", do_quantize= True)
        if save_quantized:            
            from wgp import save_quantized_model
            save_quantized_model(model, model_type, filepath, dtype, None)
            
        model.mixed_precision = mixed_precision_transformer

        if model.mixed_precision :
            model._lock_dtype = torch.float32
            model.lock_layers_dtypes(torch.float32)
        model.eval()

        # ============================= Build extra models ========================
        # VAE
        if custom or avatar:
            vae_configpath = "ckpts/hunyuan_video_custom_VAE_config.json"
            vae_filepath = "ckpts/hunyuan_video_custom_VAE_fp32.safetensors"
        # elif avatar:
        #     vae_configpath = "ckpts/config_vae_avatar.json"
        #     vae_filepath = "ckpts/vae_avatar.pt"
        else:
            vae_configpath = "ckpts/hunyuan_video_VAE_config.json"
            vae_filepath = "ckpts/hunyuan_video_VAE_fp32.safetensors"

    # config = AutoencoderKLCausal3D.load_config("ckpts/hunyuan_video_VAE_config.json")
    # config = AutoencoderKLCausal3D.load_config("c:/temp/hvae/config_vae.json")

        vae, _, s_ratio, t_ratio = load_vae( "884-16c-hy", vae_path= vae_filepath, vae_config_path= vae_configpath, vae_precision= vae_precision, device= "cpu", )

        vae._model_dtype =  torch.float32 if VAE_dtype == torch.float32 else  (torch.float16 if avatar else torch.bfloat16)
        vae._model_dtype =  torch.float32 if VAE_dtype == torch.float32 else  torch.bfloat16
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
        enable_cfg = False
        # Text encoder
        if i2v_mode:
            text_encoder = "llm-i2v"
            tokenizer = "llm-i2v"
            prompt_template = "dit-llm-encode-i2v"
            prompt_template_video = "dit-llm-encode-video-i2v"
        elif custom or avatar :
            text_encoder = "llm-i2v"
            tokenizer = "llm-i2v"
            prompt_template = "dit-llm-encode"
            prompt_template_video = "dit-llm-encode-video"
            enable_cfg = True
        else:
            text_encoder = "llm"
            tokenizer = "llm"
            prompt_template = "dit-llm-encode"
            prompt_template_video = "dit-llm-encode-video"

        if prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[prompt_template_video].get( "crop_start", 0 )
        elif prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = text_len + crop_start

        # prompt_template
        prompt_template =  PROMPT_TEMPLATE[prompt_template] if prompt_template is not None else None

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE[prompt_template_video] if prompt_template_video is not None else None
        

        text_encoder = TextEncoder(
            text_encoder_type=text_encoder,
            max_length=max_length,
            text_encoder_precision="fp16",
            tokenizer_type=tokenizer,
            i2v_mode=i2v_mode,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=True,
            device="cpu",
            image_embed_interleave=image_embed_interleave,
   			text_encoder_path = text_encoder_filepath            
        )

        text_encoder_2 = TextEncoder(
            text_encoder_type="clipL",
            max_length=77,
            text_encoder_precision="fp16",
            tokenizer_type="clipL",
            reproduce=True,
            device="cpu",
        )

        feature_extractor = None
        wav2vec = None
        align_instance = None

        if avatar or custom_audio:
            feature_extractor = AutoFeatureExtractor.from_pretrained("ckpts/whisper-tiny/")
            wav2vec = WhisperModel.from_pretrained("ckpts/whisper-tiny/").to(device="cpu", dtype=torch.float32)
            wav2vec._model_dtype = torch.float32
            wav2vec.requires_grad_(False)
        if avatar:
            align_instance = AlignImage("cuda", det_path="ckpts/det_align/detface.pt")
            align_instance.facedet.model.to("cpu")
            adapt_model(model, "audio_adapter_blocks")
        elif custom_audio:
            adapt_model(model, "audio_models")

        return cls(
            i2v=i2v_mode,
            custom=custom,
            avatar=avatar,
            enable_cfg = enable_cfg,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            feature_extractor=feature_extractor,
            wav2vec=wav2vec,
            align_instance=align_instance,
            device=device,
        )

  

class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        i2v,
        custom,
        avatar,
        enable_cfg,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        feature_extractor=None,
        wav2vec=None,
        align_instance=None,
        device=0,
    ):
        super().__init__(
            i2v,
            custom,
            avatar,
            enable_cfg,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            feature_extractor=feature_extractor,
            wav2vec=wav2vec,
            align_instance=align_instance,
            device=device,
        )

        self.i2v_mode = i2v
        self.enable_cfg = enable_cfg
        self.pipeline = self.load_diffusion_pipeline(
            avatar = self.avatar,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        if self.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT

    @property
    def _interrupt(self):
        return self.pipeline._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.pipeline._interrupt =value 

    def load_diffusion_pipeline(
        self,
        avatar,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        #data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            scheduler = FlowMatchDiscreteScheduler(
                shift=6.0,
                reverse=True,
                solver="euler",
            )

        if avatar:
            pipeline = HunyuanVideoAudioPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                transformer=model,
                scheduler=scheduler,
                progress_bar_config=progress_bar_config,
            )
        else:
            pipeline = HunyuanVideoPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                transformer=model,
                scheduler=scheduler,
                progress_bar_config=progress_bar_config,
            )
 
        return pipeline

    def get_rotary_pos_embed_new(self, video_length, height, width, concat_dict={}, enable_riflex = False):
        target_ndim = 3
        ndim = 5 - 2
        latents_size = [(video_length-1)//4+1 , height//8, width//8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(rope_dim_list, 
                                                    rope_sizes, 
                                                    theta=256, 
                                                    use_real=True,
                                                    theta_rescale_factor=1,
                                                    concat_dict=concat_dict,
                                                    L_test = (video_length - 1) // 4 + 1,
                                                    enable_riflex = enable_riflex
                                                    )
        return freqs_cos, freqs_sin
        
    def get_rotary_pos_embed(self, video_length, height, width, enable_riflex = False):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        vae = "884-16c-hy"
        if "884" in vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(
                s % self.model.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=256,
            use_real=True,
            theta_rescale_factor=1,
            L_test = (video_length - 1) // 4 + 1,
            enable_riflex = enable_riflex
        )
        return freqs_cos, freqs_sin


    def generate(
        self,
        input_prompt,
        input_ref_images = None,
        audio_guide = None,
        input_frames = None,
        input_masks = None,
        input_video = None,        
        fps = 24,
        height=192,
        width=336,
        frame_num=129,
        seed=None,
        n_prompt=None,
        sampling_steps=50,
        guide_scale=1.0,
        shift=5.0,
        embedded_guidance_scale=6.0,
        batch_size=1,
        num_videos_per_prompt=1,
        image_start=None,
        enable_RIFLEx = False,
        i2v_condition_type: str = "token_replace",
        i2v_stability=True,
        VAE_tile_size = None,
        joint_pass = False,
        cfg_star_switch = False,
        fit_into_canvas = True,
        conditioning_latents_size = 0,
        **kwargs,
    ):

        if VAE_tile_size != None:
            self.vae.tile_sample_min_tsize = VAE_tile_size["tile_sample_min_tsize"]
            self.vae.tile_latent_min_tsize = VAE_tile_size["tile_latent_min_tsize"]
            self.vae.tile_sample_min_size = VAE_tile_size["tile_sample_min_size"]
            self.vae.tile_latent_min_size = VAE_tile_size["tile_latent_min_size"]
            self.vae.tile_overlap_factor = VAE_tile_size["tile_overlap_factor"]
            self.vae.enable_tiling()

        i2v_mode= self.i2v_mode
        if not self.enable_cfg:
            guide_scale=1.0

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        from shared.utils.utils import seed_everything
        seed_everything(seed)
        generator = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]
        # generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # ========================================================================
        # Arguments: target_width, target_height, target_frame_num
        # ========================================================================
        if width <= 0 or height <= 0 or frame_num <= 0:
            raise ValueError(
                f"`height` and `width` and `frame_num` must be positive integers, got height={height}, width={width}, frame_num={frame_num}"
            )
        if (frame_num - 1) % 4 != 0:
            raise ValueError(
                f"`frame_num-1` must be a multiple of 4, got {frame_num}"
            )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_frame_num = frame_num
        audio_strength = 1

        if input_ref_images  != None:
            # ip_cfg_scale = 3.0
            ip_cfg_scale = 0
            denoise_strength = 1
            # guide_scale=7.5
            # shift=13
            name = "person"
            input_ref_images = input_ref_images[0]

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(input_prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(input_prompt)}")
        input_prompt = [input_prompt.strip()]

        # negative prompt
        if n_prompt is None or n_prompt == "":
            n_prompt = self.default_negative_prompt
        if guide_scale == 1.0:
            n_prompt = ""
        if not isinstance(n_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(n_prompt)}"
            )
        n_prompt = [n_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=shift,
            reverse=True,
            solver="euler"
        )
        self.pipeline.scheduler = scheduler

        # ---------------------------------
        # Reference condition
        # ---------------------------------
        img_latents = None
        semantic_images = None
        denoise_strength = 0
        ip_cfg_scale = 0
        if i2v_mode:
            semantic_images = convert_tensor_to_image(image_start)
            semantic_image_pixel_values = image_start.unsqueeze(0).unsqueeze(2).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img_latents = self.pipeline.vae.encode(semantic_image_pixel_values).latent_dist.mode() # B, C, F, H, W
                img_latents.mul_(self.pipeline.vae.config.scaling_factor)

            target_height, target_width = image_start.shape[1:] 

        # ========================================================================
        # Build Rope freqs
        # ========================================================================

        if input_ref_images == None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(target_frame_num, target_height, target_width, enable_RIFLEx)
        else:
            if self.avatar:
                concat_dict = {'mode': 'timecat', 'bias': -1} 
                freqs_cos, freqs_sin = self.get_rotary_pos_embed_new(129, target_height, target_width, concat_dict)
            else:
                if input_frames != None:
                    target_height, target_width = input_frames.shape[-2:]
                elif input_video != None:
                    target_height, target_width = input_video.shape[-2:]

                concat_dict = {'mode': 'timecat-w', 'bias': -1} 
                freqs_cos, freqs_sin = self.get_rotary_pos_embed_new(target_frame_num, target_height, target_width, concat_dict, enable_RIFLEx)

        n_tokens = freqs_cos.shape[0]

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        # ========================================================================
        # Pipeline inference
        # ========================================================================

        pixel_value_llava, uncond_pixel_value_llava, pixel_value_ref =  None, None, None
        if input_ref_images  == None:
            name = None
        else:
            pixel_value_llava, uncond_pixel_value_llava, pixel_value_ref =  DataPreprocess().get_batch(input_ref_images, (target_width, target_height), pad = self.custom)

        ref_latents, uncond_audio_prompts, audio_prompts, face_masks, motion_exp, motion_pose = None, None, None, None, None, None


        bg_latents = None
        if input_video != None:
            pixel_value_bg = input_video.unsqueeze(0)
            pixel_value_mask =  torch.zeros_like(input_video).unsqueeze(0)
        if input_frames != None:
            pixel_value_video_bg = input_frames.unsqueeze(0) #.permute(-1,0,1,2).unsqueeze(0).float()
            # pixel_value_video_bg = pixel_value_video_bg.div_(127.5).add_(-1.)
            # pixel_value_video_mask = input_masks.unsqueeze(-1).repeat(1,1,1,3).permute(-1,0,1,2).unsqueeze(0).float()
            pixel_value_video_mask = input_masks.repeat(3,1,1,1).unsqueeze(0)
            if input_video != None:
                pixel_value_bg = torch.cat([pixel_value_bg, pixel_value_video_bg], dim=2)
                pixel_value_mask = torch.cat([ pixel_value_mask, pixel_value_video_mask], dim=2)
            else:
                pixel_value_bg = pixel_value_video_bg
                pixel_value_mask = pixel_value_video_mask
            pixel_value_video_mask, pixel_value_video_bg  = None, None
        if input_video != None or input_frames != None:
            if pixel_value_bg.shape[2] < frame_num:
                padding_shape = list(pixel_value_bg.shape[0:2]) + [frame_num-pixel_value_bg.shape[2]] +  list(pixel_value_bg.shape[3:])  
                pixel_value_bg = torch.cat([pixel_value_bg, torch.full(padding_shape, -1, dtype=pixel_value_bg.dtype, device= pixel_value_bg.device ) ], dim=2)
                # pixel_value_mask = torch.cat([ pixel_value_mask, torch.full(padding_shape, 255, dtype=pixel_value_mask.dtype, device= pixel_value_mask.device ) ], dim=2)
                pixel_value_mask = torch.cat([ pixel_value_mask, torch.full(padding_shape, 1, dtype=pixel_value_mask.dtype, device= pixel_value_mask.device ) ], dim=2)

            bg_latents = self.vae.encode(pixel_value_bg).latent_dist.sample()                
            pixel_value_mask = pixel_value_mask.mul_(2).add_(-1.)    # unmasked pixels is -1 (no 0 as usual) and masked is 1 
            mask_latents = self.vae.encode(pixel_value_mask).latent_dist.sample()
            bg_latents = torch.cat([bg_latents, mask_latents], dim=1)
            bg_latents.mul_(self.vae.config.scaling_factor)

        if self.avatar:
            if n_prompt == None or len(n_prompt) == 0:
                n_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, Lens changes"

            uncond_pixel_value_llava = pixel_value_llava.clone()

            pixel_value_ref = pixel_value_ref.unsqueeze(0)
            self.align_instance.facedet.model.to("cuda")
            face_masks = get_facemask(pixel_value_ref.to("cuda")*255, self.align_instance, area=3.0) 
            # iii = (face_masks.squeeze(0).squeeze(0).permute(1,2,0).repeat(1,1,3)*255).cpu().numpy().astype(np.uint8)
            # image = Image.fromarray(iii)
            # image.save("mask.png")
            # jjj = (pixel_value_ref.squeeze(0).squeeze(0).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)

            self.align_instance.facedet.model.to("cpu")
            # pixel_value_ref = pixel_value_ref.clone().repeat(1,129,1,1,1)

            pixel_value_ref = pixel_value_ref.repeat(1,1+4*2,1,1,1)
            pixel_value_ref = pixel_value_ref * 2 - 1 
            pixel_value_ref_for_vae = rearrange(pixel_value_ref, "b f c h w -> b c f h w")

            vae_dtype = self.vae.dtype
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                ref_latents = self.vae.encode(pixel_value_ref_for_vae).latent_dist.sample()
                ref_latents = torch.cat( [ref_latents[:,:, :1], ref_latents[:,:, 1:2].repeat(1,1,31,1,1),  ref_latents[:,:, -1:]], dim=2)
                pixel_value_ref, pixel_value_ref_for_vae = None, None

                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    ref_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                else:
                    ref_latents.mul_(self.vae.config.scaling_factor)

                # out_latents= ref_latents / self.vae.config.scaling_factor
                # image = self.vae.decode(out_latents, return_dict=False, generator=generator)[0]
                # image = image.clamp(-1, 1)
                # from shared.utils.utils import cache_video
                # cache_video( tensor=image, save_file="decode.mp4", fps=25, nrow=1, normalize=True, value_range=(-1, 1))

            motion_pose = np.array([25] * 4)
            motion_exp = np.array([30] * 4)
            motion_pose = torch.from_numpy(motion_pose).unsqueeze(0)
            motion_exp = torch.from_numpy(motion_exp).unsqueeze(0)

            face_masks = torch.nn.functional.interpolate(face_masks.float().squeeze(2), 
                                                    (ref_latents.shape[-2], 
                                                    ref_latents.shape[-1]), 
                                                    mode="bilinear").unsqueeze(2).to(dtype=ref_latents.dtype)


        if audio_guide != None:            
            audio_input, audio_len = get_audio_feature(self.feature_extractor, audio_guide, duration = frame_num/fps )
            audio_prompts = audio_input[0]
            weight_dtype = audio_prompts.dtype
            if self.custom:
                audio_len = min(audio_len, frame_num)
                audio_input = audio_input[:, :audio_len]
            audio_prompts = encode_audio(self.wav2vec, audio_prompts.to(dtype=self.wav2vec.dtype), fps, num_frames=audio_len) 
            audio_prompts = audio_prompts.to(self.model.dtype)
            segment_size = 129 if self.avatar else frame_num
            if audio_prompts.shape[1] <= segment_size:
                audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1,segment_size-audio_prompts.shape[1], 1, 1, 1)], dim=1)
            else:
                audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1, 5, 1, 1, 1)], dim=1)
            uncond_audio_prompts = torch.zeros_like(audio_prompts[:,:129])

        samples = self.pipeline(
            prompt=input_prompt,
            height=target_height,
            width=target_width,
            video_length=target_frame_num,
            num_inference_steps=sampling_steps,
            guidance_scale=guide_scale,
            negative_prompt=n_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            name = name,

            pixel_value_ref = pixel_value_ref,
            ref_latents=ref_latents,                            # [1, 16, 1, h//8, w//8]
            pixel_value_llava=pixel_value_llava,                # [1, 3, 336, 336]
            uncond_pixel_value_llava=uncond_pixel_value_llava,
            face_masks=face_masks,                              # [b f h w]
            audio_prompts=audio_prompts, 
            uncond_audio_prompts=uncond_audio_prompts, 
            motion_exp=motion_exp, 
            motion_pose=motion_pose, 
            fps= torch.from_numpy(np.array(fps)), 

            bg_latents = bg_latents,
            audio_strength = audio_strength,

            denoise_strength=denoise_strength,
            ip_cfg_scale=ip_cfg_scale,             
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_frame_num > 1 else "image",
            is_progress_bar=True,
            vae_ver="884-16c-hy",
            enable_tiling=True,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=img_latents,
            semantic_images=semantic_images,
            joint_pass = joint_pass,
            cfg_star_rescale = cfg_star_switch,
            callback = callback,
            callback_steps = callback_steps,
        )[0]

        if samples == None:
            return None
        samples = samples.squeeze(0)

        return samples



