import json
import math
import os

import torch
from accelerate import init_empty_weights
from einops import rearrange, repeat
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2TokenizerFast

from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.text_encoder_cache import TextEncoderCache

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from models.ideogram4.qwen3_vl_transformers import Qwen3VLTextModel
from models.qwen.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

from .krea2_mmdit import SingleStreamDiT, config_from_diffusers


_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
_DEFAULT_NEGATIVE_PROMPT = ""
_TRANSFORMER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "krea2_transformer_config.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def _timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def _prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


class Krea2TextEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = Qwen3VLTextModel(config.text_config)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(self, text_encoder, tokenizer, processor, max_length=512, select_layers=_TEXT_ENCODER_SELECT_LAYERS):
        super().__init__()
        self.qwen = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def _tokenize(self, text: list[str], device):
        prefix_idx = self.prompt_template_encode_start_idx
        target_device = torch.device(device)
        prefixed_text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        # Tokenizers create PyTorch tensors via the global default device; pin that choice here so MMGP
        # offload state cannot make token tensors bounce through CPU with an unsafe async copy.
        with torch.device(target_device):
            suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(target_device)
            inputs = self.tokenizer(
                prefixed_text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(target_device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        return input_ids, mask, position_ids, prefix_idx

    @torch.inference_mode()
    def forward(self, text: list[str], device):
        self.qwen.language_model._interrupt = getattr(self, "_interrupt", False)
        if getattr(self, "_interrupt", False):
            return None, None
        input_ids, mask, position_ids, prefix_idx = self._tokenize(text, device=device)
        selected_layers = [layer_idx - 1 for layer_idx in self.select_layers]
        states = self.qwen.language_model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_mid_results_layers=selected_layers)
        if states.last_hidden_state is None:
            return None, None
        mid_results = states.mid_results
        hiddens = torch.stack(mid_results, dim=2)
        states.mid_results = None
        del mid_results, states
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        return hiddens, mask


class _TextEncodingInterrupted(Exception):
    pass


def _lora_schedules_are_static_for_modules(model, prefixes):
    scaling = getattr(model, "_loras_scaling", None)
    if not scaling:
        return True
    dynamic_adapters = {name for name, values in scaling.items() if isinstance(values, list) and any(value != values[0] for value in values[1:])}
    if not dynamic_adapters:
        return True
    shortcuts = getattr(model, "_loras_model_shortcuts", None)
    if not shortcuts:
        return True
    for module_name, loras_data in shortcuts.items():
        if module_name.startswith(prefixes) and any(adapter in loras_data for adapter in dynamic_adapters):
            return False
    return True


class Krea2Pipeline:
    def __init__(self, transformer, vae, encoder, dtype=torch.bfloat16):
        self.transformer = transformer
        self.vae = vae
        self.encoder = encoder
        self.text_encoder_cache = TextEncoderCache()
        self.dtype = dtype
        self.compression = 8
        self.channels = 16
        self._interrupt = False
        self.transformer._interrupt = False
        self.transformer.txtfusion._interrupt = False
        self.encoder._interrupt = False
        self.encoder.qwen.language_model._interrupt = False

    @property
    def runtime_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else next(self.transformer.parameters()).device)

    def _decode_latents_to_cpu_uint8(self, latents):
        latents = rearrange(latents, "b c h w -> b c 1 h w").to(self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents * latents_std) + latents_mean
        return self.vae.decode_to_cpu_uint8(latents)[:, :, 0]

    def _encode_prompts(self, prompts, device, dtype):
        self.encoder._interrupt = self._interrupt
        self.encoder.qwen.language_model._interrupt = self._interrupt

        def encode_fn(prompt_batch):
            hiddens, masks = self.encoder(prompt_batch, device=device)
            if hiddens is None:
                raise _TextEncodingInterrupted
            return [(hiddens[i], masks[i]) for i in range(len(prompt_batch))]

        cache_keys = [(self.encoder.max_length, tuple(self.encoder.select_layers), prompt) for prompt in prompts]
        try:
            encoded = self.text_encoder_cache.encode(encode_fn, prompts, device=device, cache_keys=cache_keys)
        except _TextEncodingInterrupted:
            return None, None
        hiddens = torch.stack([item[0] for item in encoded], dim=0).to(device=device, dtype=dtype, non_blocking=True)
        masks = torch.stack([item[1] for item in encoded], dim=0).to(device=device, non_blocking=True)
        return hiddens, masks

    @torch.inference_mode()
    def __call__(self, prompts, negative_prompts=None, width=1024, height=1024, steps=28, guidance=4.5, seed=0, y1=0.5, y2=1.15, mu=None, callback=None, loras_slists=None):
        patch = self.transformer.config.patch
        align = self.compression * patch
        width, height = int(width), int(height)
        if width % align != 0 or height % align != 0:
            raise ValueError(f"Krea 2 width and height must be divisible by {align}; got {width}x{height}.")
        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [_DEFAULT_NEGATIVE_PROMPT] * len(prompts) if negative_prompts is None else negative_prompts
        device = self.runtime_device
        dtype = self.dtype
        batch_size = len(prompts)
        noise = torch.empty(batch_size, self.channels, height // self.compression, width // self.compression, device=device, dtype=dtype)
        for i in range(batch_size):
            noise[i]= torch.randn(self.channels, height // self.compression, width // self.compression, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(int(seed) + i))
        txt, txtmask = self._encode_prompts(prompts, device, dtype)
        if txt is None:
            return None
        x, pos, mask = _prepare(noise, txt.shape[1], patch, txtmask)
        cfg = guidance > 0
        if cfg:
            untxt, untxtmask = self._encode_prompts(negative_prompts, device, dtype)
            if untxt is None:
                return None
            _, unpos, unmask = _prepare(noise, untxt.shape[1], patch, untxtmask)
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = _timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)
        img = x
        self.transformer._interrupt = self._interrupt
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps)
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(self.transformer, loras_slists, steps)
        context_static = _lora_schedules_are_static_for_modules(self.transformer, ("txtfusion.", "txtmlp."))
        timestep_static = _lora_schedules_are_static_for_modules(self.transformer, ("tmlp.", "tproj."))
        if context_static:
            offload.set_step_no_for_lora(self.transformer, 0)
            self.transformer._interrupt = self._interrupt
            txt_list = [txt]
            txt = None
            txt = self.transformer.prepare_context(txt_list, mask)
            if txt is None:
                return None
            if cfg:
                untxt_list = [untxt]
                untxt = None
                untxt = self.transformer.prepare_context(untxt_list, unmask)
                if untxt is None:
                    return None
        t_values = torch.tensor(ts[:-1], dtype=img.dtype, device=img.device)
        if timestep_static:
            offload.set_step_no_for_lora(self.transformer, 0)
            t_all, tvec_all = self.transformer.prepare_timestep(t_values)
            step_tensors = tuple((t_all[i : i + 1], tvec_all[i : i + 1]) for i in range(steps))
        else:
            step_tensors = []
            for step_no, tcurr in enumerate(t_values):
                offload.set_step_no_for_lora(self.transformer, step_no)
                step_tensors.append(self.transformer.prepare_timestep(tcurr[None]))
        torch.cuda.empty_cache()
        for i, (tcurr, tprev) in enumerate(tqdm(list(zip(ts[:-1], ts[1:])), total=steps)):
            offload.set_step_no_for_lora(self.transformer, i)
            self.transformer._interrupt = self._interrupt
            if self._interrupt:
                return None
            t, tvec = step_tensors[i]
            if cfg:
                step_txt = txt if context_static else self.transformer.prepare_context(txt, mask)
                step_untxt = untxt if context_static else self.transformer.prepare_context(untxt, unmask)
                if step_txt is None or step_untxt is None:
                    return None
                cond, uncond = self.transformer.forward_cfg(img=img, context=step_txt, uncond_context=step_untxt, t=t, tvec=tvec, pos=pos, uncond_pos=unpos, mask=mask, uncond_mask=unmask)
                if cond is None or uncond is None:
                    return None
                v = cond + guidance * (cond - uncond)
                del uncond
            else:
                step_txt = txt if context_static else self.transformer.prepare_context(txt, mask)
                if step_txt is None:
                    return None
                cond = self.transformer(img=img, context=step_txt, t=t, tvec=tvec, pos=pos, mask=mask)
                if cond is None:
                    return None
                v = cond
            img = img + (tprev - tcurr) * v
            del cond, v
            if callback is not None:
                preview = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // align, w=width // align)
                callback(i, preview.transpose(0, 1), False, preview_meta=None)
        if self._interrupt:
            return None
        latents = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // align, w=width // align)
        return self._decode_latents_to_cpu_uint8(latents)


def _load_transformer(model_filename, config_path, dtype):
    config = config_from_diffusers(_load_json(config_path))
    with init_empty_weights(include_buffers=True):
        transformer = SingleStreamDiT(config)
    offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=dtype)
    transformer.eval().requires_grad_(False)
    return transformer


def _load_text_encoder(text_encoder_filename, config_path, dtype):
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = Krea2TextEncoder(config)
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(text_encoder.language_model, text_encoder_filename, modelPrefix="language_model", writable_tensors=False, default_dtype=dtype)
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_vae(filename, config_path, dtype):
    config = _load_json(config_path)
    for key in ("_class_name", "_diffusers_version", "_name_or_path"):
        config.pop(key, None)
    with init_empty_weights(include_buffers=True):
        vae = AutoencoderKLQwenImage(**config)
    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae


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
        dtype = torch.bfloat16
        self.base_model_type = base_model_type
        self.model_def = model_def
        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        config_path = _TRANSFORMER_CONFIG_PATH
        transformer = _load_transformer(transformer_filename, config_path, dtype)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, transformer_filename, dtype, config_path)
        text_encoder_folder = model_def["text_encoder_folder"]
        text_encoder_config_path = fl.locate_file(os.path.join(text_encoder_folder, "config.json"))
        text_encoder = _load_text_encoder(text_encoder_filename, text_encoder_config_path, dtype)
        tokenizer_config = fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
        fl.locate_file(os.path.join(text_encoder_folder, "tokenizer.json"))
        fl.locate_file(os.path.join(text_encoder_folder, "chat_template.jinja"))
        tokenizer_path = os.path.dirname(tokenizer_config)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=512, trust_remote_code=True, extra_special_tokens={})
        processor = Qwen2TokenizerFast.from_pretrained(tokenizer_path, max_length=512, extra_special_tokens={})
        vae = _load_vae(fl.locate_file("qwen_vae.safetensors"), fl.locate_file("qwen_vae_config.json"), VAE_dtype)
        self.pipeline = Krea2Pipeline(transformer, vae, Qwen3VLConditioner(text_encoder, tokenizer, processor), dtype=dtype)
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 28,
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 4.5,
        batch_size: int = 1,
        callback=None,
        VAE_tile_size=None,
        loras_slists=None,
        **kwargs,
    ):
        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling = VAE_tile_size > 0
                tile_size = max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            if tiling:
                self.vae.enable_tiling(tile_sample_min_height=tile_size or None, tile_sample_min_width=tile_size or None)
            else:
                self.vae.disable_tiling()
        turbo = self.base_model_type == "krea2_turbo"
        if turbo:
            guide_scale = 0
            kwargs_mu = 1.15
        else:
            kwargs_mu = None
        generator_seed = seed if seed is not None and seed >= 0 else torch.seed()
        prompts = [input_prompt] * int(batch_size)
        images = self.pipeline(prompts, negative_prompts=[n_prompt or _DEFAULT_NEGATIVE_PROMPT] * len(prompts), width=width, height=height, steps=sampling_steps, guidance=guide_scale, seed=generator_seed, mu=kwargs_mu, callback=callback, loras_slists=loras_slists)
        if images is None:
            return None
        return images.transpose(0, 1)

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
            self.pipeline.encoder._interrupt = value
            self.pipeline.encoder.qwen.language_model._interrupt = value
        if hasattr(self, "transformer"):
            self.transformer._interrupt = value
            self.transformer.txtfusion._interrupt = value
        if hasattr(self, "text_encoder"):
            self.text_encoder.language_model._interrupt = value
