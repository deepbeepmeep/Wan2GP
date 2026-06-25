import datetime
import hashlib
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


def _tensor_stats(tensor):
    if tensor.numel() == 0:
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": 0,
        }
    values = tensor.float()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "finite": bool(torch.isfinite(values).all().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _mask_runs(mask):
    mask_cpu = mask.detach().bool().cpu()
    out = []
    for row in mask_cpu:
        values = row.tolist()
        if not values:
            out.append([])
            continue
        runs = []
        start = 0
        current = values[0]
        for idx, value in enumerate(values[1:], 1):
            if value != current:
                runs.append((start, idx - 1, bool(current), idx - start))
                start = idx
                current = value
        runs.append((start, len(values) - 1, bool(current), len(values) - start))
        out.append(runs)
    return out


def _tensor_sample_sha256(tensor, max_items=4096):
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return None
    flat = tensor.reshape(-1)
    if flat.numel() <= max_items * 3:
        sample = flat
    else:
        first = torch.arange(max_items, device=flat.device)
        middle_start = max((flat.numel() // 2) - (max_items // 2), 0)
        middle = torch.arange(middle_start, middle_start + max_items, device=flat.device)
        last = torch.arange(flat.numel() - max_items, flat.numel(), device=flat.device)
        sample = flat[torch.cat((first, middle, last))]
    sample = sample.detach().cpu().contiguous()
    return hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_fingerprint(tensor, max_items=4096):
    payload = _tensor_stats(tensor.detach())
    payload["sample_sha256"] = _tensor_sample_sha256(tensor, max_items=max_items)
    return payload


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

    @property
    def device(self):
        return next(self.qwen.parameters()).device

    def _tokenizer_debug_metadata(self):
        try:
            first_param = next(self.qwen.language_model.parameters())
            first_param_info = {"dtype": str(first_param.dtype), "device": str(first_param.device)}
        except StopIteration:
            first_param_info = None
        return {
            "tokenizer_class": type(self.tokenizer).__module__ + "." + type(self.tokenizer).__name__,
            "processor_class": type(self.processor).__module__ + "." + type(self.processor).__name__,
            "language_model_class": type(self.qwen.language_model).__module__ + "." + type(self.qwen.language_model).__name__,
            "language_model_first_parameter": first_param_info,
            "tokenizer_model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "processor_model_max_length": getattr(self.processor, "model_max_length", None),
            "tokenizer_padding_side": getattr(self.tokenizer, "padding_side", None),
            "tokenizer_truncation_side": getattr(self.tokenizer, "truncation_side", None),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
            "special_tokens_map": dict(getattr(self.tokenizer, "special_tokens_map", {}) or {}),
        }

    def _tokenize(self, text: list[str], return_debug=False):
        prefix_idx = self.prompt_template_encode_start_idx
        prefixed_text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.device, non_blocking=True)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()
        inputs = self.tokenizer(
            prefixed_text,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
            return_tensors="pt",
        ).to(self.device, non_blocking=True)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        debug = None
        if return_debug:
            sliced_mask = mask[:, prefix_idx:]
            debug = {
                "metadata": self._tokenizer_debug_metadata(),
                "prompts": list(text),
                "prompt_char_lengths": [len(item) for item in text],
                "max_length": self.max_length,
                "prefix_idx": prefix_idx,
                "suffix_start_idx": self.prompt_template_encode_suffix_start_idx,
                "combined_length": int(input_ids.shape[1]),
                "sliced_length": int(sliced_mask.shape[1]),
                "tokenizer_input_ids": inputs["input_ids"].detach().cpu(),
                "tokenizer_attention_mask": inputs["attention_mask"].detach().bool().cpu(),
                "suffix_input_ids": suffix_ids.detach().cpu(),
                "suffix_attention_mask": suffix_mask.detach().cpu(),
                "combined_input_ids": input_ids.detach().cpu(),
                "combined_attention_mask_before_model": mask.detach().cpu(),
                "combined_position_ids": position_ids.detach().cpu(),
                "sliced_attention_mask_before_model": sliced_mask.detach().cpu(),
                "combined_mask_true_count": int(mask.sum().item()),
                "sliced_mask_true_count": int(sliced_mask.sum().item()),
                "combined_mask_runs": _mask_runs(mask),
                "sliced_mask_runs": _mask_runs(sliced_mask),
            }
        return input_ids, mask, position_ids, prefix_idx, debug

    @torch.inference_mode()
    def tokenize_debug(self, text: list[str]):
        _, _, _, _, debug = self._tokenize(text, return_debug=True)
        return debug

    @torch.inference_mode()
    def forward(self, text: list[str], return_debug=False):
        self.qwen.language_model._interrupt = getattr(self, "_interrupt", False)
        if getattr(self, "_interrupt", False):
            if return_debug:
                return None, None, None
            return None, None
        input_ids, mask, position_ids, prefix_idx, debug = self._tokenize(text, return_debug=return_debug)
        mask_before_model = mask.detach().clone() if return_debug else None
        selected_layers = [layer_idx - 1 for layer_idx in self.select_layers]
        states = self.qwen.language_model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_mid_results_layers=selected_layers)
        if states.last_hidden_state is None:
            if return_debug:
                return None, None, debug
            return None, None
        mid_results = states.mid_results
        if return_debug:
            debug["mask_mutated_by_language_model"] = not torch.equal(mask, mask_before_model)
            debug["combined_attention_mask_after_model"] = mask.detach().cpu()
            debug["sliced_attention_mask_after_model"] = mask[:, prefix_idx:].detach().cpu()
            debug["sliced_mask_after_model_true_count"] = int(mask[:, prefix_idx:].sum().item())
            debug["selected_layers"] = tuple(self.select_layers)
            debug["selected_internal_layers"] = tuple(selected_layers)
            debug["mid_result_stats"] = [_tensor_fingerprint(item.detach()) for item in mid_results]
            debug["last_hidden_state_stats"] = _tensor_fingerprint(states.last_hidden_state.detach())
        hiddens = torch.stack(mid_results, dim=2)
        states.mid_results = None
        del mid_results, states
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        if return_debug:
            debug["hiddens_after_slice_stats"] = _tensor_fingerprint(hiddens.detach())
            debug["final_mask_true_count"] = int(mask.sum().item())
            debug["final_mask_runs"] = _mask_runs(mask)
            return hiddens, mask, debug
        return hiddens, mask


class _TextEncodingInterrupted(Exception):
    pass


def _lora_schedules_are_static(model):
    scaling = getattr(model, "_loras_scaling", None)
    if not scaling:
        return True
    for values in scaling.values():
        if isinstance(values, list) and any(value != values[0] for value in values[1:]):
            return False
    return True


def _token_row_debug(hiddens, mask):
    hiddens = hiddens.detach().cpu()
    mask = mask.detach().bool().cpu()
    flat = hiddens.float().flatten(2)
    rows = []
    for batch_idx in range(flat.shape[0]):
        row = flat[batch_idx]
        diff_first = (row - row[0]).abs().max(dim=1).values if row.shape[0] > 0 else torch.empty(0)
        diff_prev = (row[1:] - row[:-1]).abs().max(dim=1).values if row.shape[0] > 1 else torch.empty(0)
        real = mask[batch_idx]
        payload = {
            "exact_rows_equal_first": int((diff_first == 0).sum().item()) if diff_first.numel() > 0 else 0,
            "exact_consecutive_equal": int((diff_prev == 0).sum().item()) if diff_prev.numel() > 0 else 0,
            "diff_first_stats": _tensor_stats(diff_first),
            "diff_previous_stats": _tensor_stats(diff_prev),
            "real_token_count": int(real.sum().item()),
        }
        if real.any():
            real_rows = row[real]
            real_diff_first = (real_rows - real_rows[0]).abs().max(dim=1).values
            payload["real_exact_rows_equal_first"] = int((real_diff_first == 0).sum().item())
            payload["real_diff_first_stats"] = _tensor_stats(real_diff_first)
        rows.append(payload)
    return rows


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
        self._debug_dump_index = 0
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

    def _dump_text_encoding_debug(self, label, prompts, hiddens, masks, debug_dir, cache_debug=None, tokenizer_debug=None, encoder_debug=None):
        if not debug_dir:
            return
        os.makedirs(debug_dir, exist_ok=True)
        self._debug_dump_index += 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(debug_dir, f"krea2_text_encoding_{stamp}_{self._debug_dump_index:02d}_{label}.pt")
        hiddens_cpu = hiddens.detach().cpu()
        masks_cpu = masks.detach().cpu()
        mask_comparison = None
        if isinstance(tokenizer_debug, dict) and "sliced_attention_mask_before_model" in tokenizer_debug:
            expected_mask = tokenizer_debug["sliced_attention_mask_before_model"].bool()
            if expected_mask.shape == masks_cpu.shape:
                mask_comparison = {
                    "matches_tokenizer_sliced_mask": bool(torch.equal(masks_cpu.bool(), expected_mask)),
                    "different_entries": int((masks_cpu.bool() != expected_mask).sum().item()),
                    "tokenizer_true_count": int(expected_mask.sum().item()),
                    "final_true_count": int(masks_cpu.bool().sum().item()),
                }
            else:
                mask_comparison = {
                    "matches_tokenizer_sliced_mask": False,
                    "shape_mismatch": (tuple(masks_cpu.shape), tuple(expected_mask.shape)),
                }
        torch.save({
            "kind": label,
            "prompts": list(prompts),
            "max_length": self.encoder.max_length,
            "select_layers": tuple(self.encoder.select_layers),
            "hiddens": hiddens_cpu,
            "mask": masks_cpu,
            "hiddens_stats": _tensor_stats(hiddens_cpu),
            "mask_true_count": int(masks_cpu.sum().item()),
            "mask_shape": tuple(masks_cpu.shape),
            "mask_runs": _mask_runs(masks_cpu),
            "token_row_debug": _token_row_debug(hiddens_cpu, masks_cpu),
            "mask_comparison": mask_comparison,
            "cache_debug": cache_debug,
            "tokenizer_debug": tokenizer_debug,
            "encoder_debug": encoder_debug,
        }, path)
        print(f"[Krea2 debug] Saved text encoding dump: {path}")

    def _encode_prompts(self, prompts, device, dtype, debug_dir=None, debug_label="prompt"):
        self.encoder._interrupt = self._interrupt
        self.encoder.qwen.language_model._interrupt = self._interrupt
        debug_enabled = bool(debug_dir)
        encoder_debug = []

        def encode_fn(prompt_batch):
            if debug_enabled:
                hiddens, masks, debug = self.encoder(prompt_batch, return_debug=True)
                encoder_debug.append(debug)
            else:
                hiddens, masks = self.encoder(prompt_batch)
            if hiddens is None:
                raise _TextEncodingInterrupted
            return [(hiddens[i], masks[i]) for i in range(len(prompt_batch))]

        cache_keys = [(self.encoder.max_length, tuple(self.encoder.select_layers), prompt) for prompt in prompts]
        cache_debug = None
        tokenizer_debug = None
        if debug_enabled:
            entries = getattr(self.text_encoder_cache, "_entries", {})
            cache_debug = {
                "keys": [repr(key) for key in cache_keys],
                "pre_hit": [key in entries for key in cache_keys],
                "entry_count_before": len(entries),
                "size_bytes_before": int(getattr(self.text_encoder_cache, "_size_bytes", 0)),
            }
            tokenizer_debug = self.encoder.tokenize_debug(prompts)
        try:
            encoded = self.text_encoder_cache.encode(encode_fn, prompts, device=device, cache_keys=cache_keys)
        except _TextEncodingInterrupted:
            return None, None
        if debug_enabled:
            entries = getattr(self.text_encoder_cache, "_entries", {})
            cache_debug.update({
                "post_hit": [key in entries for key in cache_keys],
                "entry_count_after": len(entries),
                "size_bytes_after": int(getattr(self.text_encoder_cache, "_size_bytes", 0)),
                "fresh_encoder_batches": len(encoder_debug),
            })
        hiddens = torch.stack([item[0] for item in encoded], dim=0).to(device=device, dtype=dtype, non_blocking=True)
        masks = torch.stack([item[1] for item in encoded], dim=0).to(device=device, non_blocking=True)
        self._dump_text_encoding_debug(debug_label, prompts, hiddens, masks, debug_dir, cache_debug=cache_debug, tokenizer_debug=tokenizer_debug, encoder_debug=encoder_debug)
        return hiddens, masks

    @torch.inference_mode()
    def __call__(self, prompts, negative_prompts=None, width=1024, height=1024, steps=28, guidance=4.5, seed=0, y1=0.5, y2=1.15, mu=None, callback=None, loras_slists=None, debug_dir=None):
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
        txt, txtmask = self._encode_prompts(prompts, device, dtype, debug_dir=debug_dir, debug_label="positive")
        if txt is None:
            return None
        x, pos, mask = _prepare(noise, txt.shape[1], patch, txtmask)
        cfg = guidance > 0
        if cfg:
            untxt, untxtmask = self._encode_prompts(negative_prompts, device, dtype, debug_dir=debug_dir, debug_label="negative")
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
        context_static = _lora_schedules_are_static(self.transformer)
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
        for i, (tcurr, tprev) in enumerate(tqdm(list(zip(ts[:-1], ts[1:])), total=steps)):
            offload.set_step_no_for_lora(self.transformer, i)
            self.transformer._interrupt = self._interrupt
            if self._interrupt:
                return None
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            if cfg:
                step_txt = txt if context_static else self.transformer.prepare_context(txt, mask)
                step_untxt = untxt if context_static else self.transformer.prepare_context(untxt, unmask)
                if step_txt is None or step_untxt is None:
                    return None
                cond, uncond = self.transformer.forward_cfg(img=img, context=step_txt, uncond_context=step_untxt, t=t, pos=pos, uncond_pos=unpos, mask=mask, uncond_mask=unmask)
                if cond is None or uncond is None:
                    return None
                v = cond + guidance * (cond - uncond)
                del uncond
            else:
                step_txt = txt if context_static else self.transformer.prepare_context(txt, mask)
                if step_txt is None:
                    return None
                cond = self.transformer(img=img, context=step_txt, t=t, pos=pos, mask=mask)
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
        debug=False,
        debug_dir=None,
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
        images = self.pipeline(prompts, negative_prompts=[n_prompt or _DEFAULT_NEGATIVE_PROMPT] * len(prompts), width=width, height=height, steps=sampling_steps, guidance=guide_scale, seed=generator_seed, mu=kwargs_mu, callback=callback, loras_slists=loras_slists, debug_dir=debug_dir if debug else None)
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
