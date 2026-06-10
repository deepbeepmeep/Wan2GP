from __future__ import annotations

import os

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask
from tqdm import tqdm

from mmgp import offload
from shared.utils import files_locator as fl

from models.flux.modules.autoencoder_flux2 import AutoencoderKLFlux2, AutoEncoderParamsFlux2
from .constants import IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS, SEQUENCE_PADDING_INDICATOR
from .latent_norm import get_latent_norm
from .modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer, get_linear_split_map
from .qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from .qwen3_vl_transformers import Qwen3VLTextModel
from .sampler_configs import PRESETS
from .scheduler import get_schedule_for_resolution, make_step_intervals

_DEFAULT_PRESET = "V4_DEFAULT_20"
_TRANSFORMER_WRAPPER_PREFIX = "model.diffusion_model."


def _strip_transformer_wrapper(state_dict, quantization_map=None, tied_weights_map=None):
    if not any(key.startswith(_TRANSFORMER_WRAPPER_PREFIX) for key in state_dict):
        return state_dict, quantization_map, tied_weights_map

    def strip_mapping(mapping):
        if mapping is None:
            return None
        prefix_len = len(_TRANSFORMER_WRAPPER_PREFIX)
        return {key[prefix_len:]: value for key, value in mapping.items() if key.startswith(_TRANSFORMER_WRAPPER_PREFIX)}

    return strip_mapping(state_dict), strip_mapping(quantization_map), strip_mapping(tied_weights_map)


def _load_transformer(filename: str, dtype: torch.dtype) -> Ideogram4Transformer:
    config = Ideogram4Config()
    split_map = get_linear_split_map(config.emb_dim)
    with init_empty_weights(include_buffers=True):
        transformer = Ideogram4Transformer(config)
    transformer.rotary_emb.reset_inv_freq()
    offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype, fused_split_map=split_map, preprocess_sd=_strip_transformer_wrapper)
    transformer.split_linear_modules_map = split_map
    transformer.eval().requires_grad_(False)
    return transformer


class Ideogram4TextEncoder(torch.nn.Module):
    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.language_model = Qwen3VLTextModel(config.text_config)


def _load_text_encoder(filename: str, config_path: str, dtype: torch.dtype) -> Ideogram4TextEncoder:
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = Ideogram4TextEncoder(config)
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(text_encoder.language_model, filename, modelPrefix="language_model", writable_tensors=False, default_dtype=dtype)
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_autoencoder(filename: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    with init_empty_weights(include_buffers=True):
        autoencoder = AutoencoderKLFlux2(AutoEncoderParamsFlux2())
    offload.load_model_data(autoencoder, filename, writable_tensors=False, default_dtype=dtype)
    autoencoder.eval().requires_grad_(False)
    return autoencoder


class Ideogram4WanPipeline:
    def __init__(self, conditional_transformer, unconditional_transformer, text_encoder, text_tokenizer, autoencoder, dtype=torch.bfloat16) -> None:
        self.conditional_transformer = conditional_transformer
        self.unconditional_transformer = unconditional_transformer
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.autoencoder = autoencoder
        self.dtype = dtype
        self.patch_size = 2
        self.ae_scale_factor = 8
        self.max_text_tokens = 2048
        self._interrupt = False
        shift, scale = get_latent_norm()
        self.latent_shift = shift
        self.latent_scale = scale

    @property
    def device(self) -> torch.device:
        return next(self.conditional_transformer.parameters()).device

    @property
    def runtime_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else self.device)

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.text_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
        if num_text_tokens > self.max_text_tokens:
            raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.max_text_tokens}")
        return token_ids, num_text_tokens

    def _build_inputs(self, prompts: list[str], height: int, width: int) -> dict[str, torch.Tensor | int]:
        tokenized = [self._tokenize(p) for p in prompts]
        batch_size = len(prompts)
        patch = self.patch_size * self.ae_scale_factor
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by {patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, max_text_tokens, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_text_tokens, 3, dtype=torch.long)
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for batch_idx, (tokens, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[batch_idx, offset:offset + num_text] = tokens
            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[batch_idx, offset:offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset:offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset + num_text:] = image_pos
            indicator[batch_idx, offset:offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[batch_idx, offset + num_text:] = OUTPUT_IMAGE_INDICATOR
            segment_ids[batch_idx, offset:offset + total_unpadded] = 1

        device = self.runtime_device
        return {
            "token_ids": token_ids.to(device),
            "text_position_ids": text_position_ids.to(device),
            "position_ids": position_ids.to(device),
            "segment_ids": segment_ids.to(device),
            "indicator": indicator.to(device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _encode_text(self, token_ids: torch.Tensor, text_position_ids: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor | None:
        attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_position_ids[..., 0].contiguous()
        language_model = self.text_encoder.language_model
        language_model._interrupt = self._interrupt
        with torch.inference_mode():
            inputs_embeds = language_model.embed_tokens(token_ids)
            position_ids = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
            text_position_ids = position_ids[0]
            mrope_position_ids = position_ids[1:]
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            causal_mask = create_causal_mask(
                config=language_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)
            tap_layers = set(QWEN3_VL_ACTIVATION_LAYERS)
            captured = {}
            hidden_states = inputs_embeds
            for layer_idx, decoder_layer in enumerate(language_model.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                if layer_idx in tap_layers:
                    captured[layer_idx] = hidden_states.clone()
                if self._interrupt:
                    return None
        del hidden_states, inputs_embeds, position_embeddings, causal_mask, position_ids, mrope_position_ids, cache_position, text_position_ids
        first = captured[QWEN3_VL_ACTIVATION_LAYERS[0]]
        batch_size, seq_len, hidden_size = first.shape
        stacked = first.new_empty(batch_size, seq_len, hidden_size, len(QWEN3_VL_ACTIVATION_LAYERS))
        for capture_idx, layer_idx in enumerate(QWEN3_VL_ACTIVATION_LAYERS):
            hidden = captured.pop(layer_idx)
            stacked[..., capture_idx].copy_(hidden)
            del hidden
        stacked = stacked.reshape(batch_size, seq_len, hidden_size * len(QWEN3_VL_ACTIVATION_LAYERS))
        stacked.mul_(attention_mask.to(device=stacked.device, dtype=stacked.dtype).unsqueeze(-1))
        return stacked

    def _decode_image(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._unpack_vae_latents(z, grid_h, grid_w)
        vae_dtype = next(self.autoencoder.decoder.parameters()).dtype
        return self.autoencoder.decoder(z.to(vae_dtype)).float().clamp(-1.0, 1.0)

    def _decode(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        return self._decode_image(z, grid_h, grid_w).cpu().transpose(0, 1)

    def _unpack_vae_latents(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._normalize_packed_latents(z)
        batch_size = z.shape[0]
        patch = self.patch_size
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        return z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    def _normalize_packed_latents(self, z: torch.Tensor) -> torch.Tensor:
        latent_shift = self.latent_shift.to(z.device, z.dtype)
        latent_scale = self.latent_scale.to(z.device, z.dtype)
        return z * latent_scale + latent_shift

    def _pack_pid_lq_latent(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._normalize_packed_latents(z)
        batch_size = z.shape[0]
        patch = self.patch_size
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels).permute(0, 5, 3, 4, 1, 2).contiguous()
        z = z.view(batch_size, ae_channels * patch * patch, grid_h, grid_w)
        vae_mean = self.autoencoder.bn.running_mean.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
        vae_scale = torch.sqrt(self.autoencoder.bn.running_var.view(1, -1, 1, 1) + self.autoencoder.bn_eps).to(device=z.device, dtype=z.dtype)
        return z.sub(vae_mean).div(vae_scale)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: str | list[str],
        *,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 20,
        guidance_scale: float = 7.0,
        guidance_schedule=None,
        mu: float = 0.0,
        std: float = 1.75,
        seed: int | None = None,
        callback=None,
        pid_upsampler=None,
        set_progress_status=None,
    ) -> torch.Tensor | None:
        if isinstance(prompts, str):
            prompts = [prompts]
        device = self.runtime_device
        schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
        step_intervals = make_step_intervals(num_steps).to(device)
        if guidance_schedule is not None:
            gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
        else:
            gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=device)

        inputs = self._build_inputs(prompts, height=height, width=width)
        if self._interrupt:
            return None
        batch_size = len(prompts)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h = inputs["grid_h"]
        grid_w = inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = self.conditional_transformer.config.in_channels

        llm_features = self._encode_text(
            inputs["token_ids"],
            inputs["text_position_ids"],
            inputs["indicator"][:, :max_text_tokens],
        )
        if llm_features is None or self._interrupt:
            return None

        neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
        neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
        neg_indicator = inputs["indicator"][:, max_text_tokens:]
        neg_llm_features = llm_features.new_empty(batch_size, 0, llm_features.shape[-1])

        generator = torch.Generator(device=device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))
        z = torch.randn(batch_size, num_image_tokens, latent_dim, dtype=torch.float32, device=device, generator=generator)
        pos_z = torch.empty(batch_size, max_text_tokens + num_image_tokens, latent_dim, dtype=torch.float32, device=device)
        pos_z[:, :max_text_tokens].zero_()

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=num_steps)

        for step_idx, i in enumerate(tqdm(range(num_steps - 1, -1, -1), total=num_steps, desc="Denoising")):
            if self._interrupt:
                return None
            t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
            s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
            t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

            pos_z[:, max_text_tokens:].copy_(z)
            pos_out = self.conditional_transformer(
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )
            if pos_out is None:
                return None
            pos_v = pos_out[:, max_text_tokens:]

            neg_v = self.unconditional_transformer(
                llm_features=neg_llm_features,
                x=z,
                t=t,
                position_ids=neg_position_ids,
                segment_ids=neg_segment_ids,
                indicator=neg_indicator,
            )
            if neg_v is None:
                return None

            z = z + (gw_per_step[i] * pos_v + (1.0 - gw_per_step[i]) * neg_v) * (s_val - t_val)
            if callback is not None:
                preview = self._unpack_vae_latents(z[:1], grid_h, grid_w)[0].unsqueeze(1)
                callback(step_idx, preview, False)

        if self._interrupt:
            return None
        if pid_upsampler is None:
            return self._decode(z, grid_h=grid_h, grid_w=grid_w)

        def _pid_progress(_phase, current_step=None, total_steps=None):
            if callable(set_progress_status):
                if current_step is None or total_steps is None:
                    set_progress_status("PiD Spatial Upsampling in progress")
                else:
                    total_steps = int(total_steps)
                    step_no = min(int(current_step) + 1, total_steps)
                    set_progress_status(f"PiD Spatial Upsampling in progress ({step_no}/{total_steps})")

        _pid_progress(None)
        lq_image_ref = [self._decode_image(z, grid_h=grid_h, grid_w=grid_w)]
        lq_latent_ref = [self._pack_pid_lq_latent(z, grid_h, grid_w)]
        image = pid_upsampler.decode_inputs(
            lq_image_ref,
            lq_latent_ref,
            prompt=prompts,
            seed=seed,
            abort_callback=lambda: self._interrupt,
            progress_callback=_pid_progress,
        )
        if image is None:
            return None
        return image.cpu().transpose(0, 1)


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        save_quantized=False,
        **kwargs,
    ):
        model_def = model_def or {}
        if not isinstance(model_filename, (list, tuple)) or len(model_filename) < 2:
            raise ValueError("Ideogram 4 requires conditional and unconditional transformer files.")
        if text_encoder_filename is None:
            raise ValueError("Ideogram 4 requires a Qwen3-VL text encoder file.")

        self.model_type = model_type
        self.base_model_type = base_model_type
        self.model_def = model_def
        self.dtype = dtype

        text_encoder_folder = model_def.get("text_encoder_folder", "Qwen3-VL-8B-Instruct")
        tokenizer_path = os.path.dirname(fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json")))
        text_config_path = fl.locate_file(os.path.join(text_encoder_folder, "config.json"))
        vae_filename = fl.locate_file("flux2_vae.safetensors")

        self.conditional_transformer = _load_transformer(model_filename[0], dtype)
        self.unconditional_transformer = _load_transformer(model_filename[1], dtype)
        self.transformer = self.conditional_transformer
        self.transformer2 = self.unconditional_transformer
        self.model = self.conditional_transformer
        self.model2 = self.unconditional_transformer
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.conditional_transformer, model_type, model_filename[0], dtype, None)
            save_quantized_model(self.unconditional_transformer, model_type, model_filename[1], dtype, None, submodel_no=2)
        self.text_encoder = _load_text_encoder(text_encoder_filename, text_config_path, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, extra_special_tokens={})
        self.autoencoder = _load_autoencoder(vae_filename, VAE_dtype)
        self.pipeline = Ideogram4WanPipeline(
            self.conditional_transformer,
            self.unconditional_transformer,
            self.text_encoder,
            self.tokenizer,
            self.autoencoder,
            dtype=VAE_dtype,
        )

    def generate(
        self,
        seed=None,
        input_prompt="",
        sample_solver="",
        width=1024,
        height=1024,
        guide_scale=7.0,
        batch_size=1,
        model_mode=_DEFAULT_PRESET,
        pid_upsampler=None,
        set_progress_status=None,
        callback=None,
        **kwargs,
    ):
        preset_name = model_mode if model_mode in PRESETS else sample_solver if sample_solver in PRESETS else _DEFAULT_PRESET
        preset = PRESETS[preset_name]
        num_steps = preset.num_steps
        guidance_schedule = preset.guidance_schedule
        mu = preset.mu
        std = preset.std

        prompts = [input_prompt] * int(batch_size)
        return self.pipeline(
            prompts,
            height=height,
            width=width,
            num_steps=num_steps,
            guidance_scale=guide_scale,
            guidance_schedule=guidance_schedule,
            mu=mu,
            std=std,
            seed=seed,
            callback=callback,
            pid_upsampler=pid_upsampler,
            set_progress_status=set_progress_status,
        )

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = bool(value)
            self.conditional_transformer._interrupt = bool(value)
            self.unconditional_transformer._interrupt = bool(value)
            if hasattr(self.text_encoder, "language_model"):
                self.text_encoder.language_model._interrupt = bool(value)
