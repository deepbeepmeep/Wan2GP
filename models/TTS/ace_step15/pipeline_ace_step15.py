"""
ACE-Step 1.5 pipeline for WanGP.
"""

import copy
import math
import os
import random
import re

import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Model
from diffusers import AutoencoderOobleck

from mmgp import offload

from .models.ace_step15_hf import AceStepConditionGenerationModel

_DEFAULT_TIMBRE = [
    -1.3672e-01, -1.5820e-01,  5.8594e-01, -5.7422e-01,  3.0273e-02,
     2.7930e-01, -2.5940e-03, -2.0703e-01, -1.6113e-01, -1.4746e-01,
    -2.7710e-02, -1.8066e-01, -2.9688e-01,  1.6016e+00, -2.6719e+00,
     7.7734e-01, -1.3516e+00, -1.9434e-01, -7.1289e-02, -5.0938e+00,
     2.4316e-01,  4.7266e-01,  4.6387e-02, -6.6406e-01, -2.1973e-01,
    -6.7578e-01, -1.5723e-01,  9.5312e-01, -2.0020e-01, -1.7109e+00,
     5.8984e-01, -5.7422e-01,  5.1562e-01,  2.8320e-01,  1.4551e-01,
    -1.8750e-01, -5.9814e-02,  3.6719e-01, -1.0059e-01, -1.5723e-01,
     2.0605e-01, -4.3359e-01, -8.2812e-01,  4.5654e-02, -6.6016e-01,
     1.4844e-01,  9.4727e-02,  3.8477e-01, -1.2578e+00, -3.3203e-01,
    -8.5547e-01,  4.3359e-01,  4.2383e-01, -8.9453e-01, -5.0391e-01,
    -5.6152e-02, -2.9219e+00, -2.4658e-02,  5.0391e-01,  9.8438e-01,
     7.2754e-02, -2.1582e-01,  6.3672e-01,  1.0000e+00,
]

_AUDIO_CODE_RE = re.compile(r"<\|audio_code_(\d+)\|>")
_AUDIO_CODE_TOKEN_RE = re.compile(r"^<\|audio_code_(\d+)\|>$")
_DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
_DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
_SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


def _ace_step15_get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision):
    if vae_config == 0:
        if mixed_precision:
            device_mem_capacity = device_mem_capacity / 2
        if device_mem_capacity >= 24000:
            use_vae_config = 1
        elif device_mem_capacity >= 12000:
            use_vae_config = 2
        else:
            use_vae_config = 3
    else:
        use_vae_config = vae_config

    if use_vae_config == 1:
        return 0
    if use_vae_config == 2:
        return 256
    return 128


class ACEStep15Pipeline:
    def __init__(
        self,
        transformer_weights_path: str,
        transformer_config_path: str,
        vae_weights_path: str,
        vae_config_path: str,
        text_encoder_2_weights_path: str,
        text_encoder_2_tokenizer_dir: str,
        lm_weights_path: str,
        lm_tokenizer_dir: str,
        silence_latent_path: str | None = None,
        enable_lm: bool = True,
        device=None,
        dtype=torch.bfloat16,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.bfloat16)
        self.dtype = dtype

        if not text_encoder_2_weights_path:
            raise ValueError("Ace Step 1.5 requires a pre-text encoder weights path.")

        self.enable_lm = bool(enable_lm)
        if self.enable_lm and not lm_weights_path:
            raise ValueError("Ace Step 1.5 requires a 5Hz LM weights path.")

        self.text_encoder_2_weights_path = text_encoder_2_weights_path
        self.text_encoder_2_tokenizer_dir = text_encoder_2_tokenizer_dir
        self.lm_weights_path = lm_weights_path
        self.lm_tokenizer_dir = lm_tokenizer_dir
        self.silence_latent_path = silence_latent_path

        self._interrupt = False
        self._early_stop = False
        self.loaded = False

        self._latent_hop_length = 1920

        self._load_models(transformer_weights_path, transformer_config_path, vae_weights_path, vae_config_path)
        self._init_lm_hint_modules()
        self._load_tokenizers()
        self._load_text_encoder_2()
        if self.enable_lm:
            self._load_lm()
        else:
            self.lm_model = None
        self._load_silence_latent()

        self.loaded = True

    def _load_models(self, transformer_weights_path, transformer_config_path, vae_weights_path, vae_config_path):
        self.ace_step_transformer = offload.fast_load_transformers_model(
            transformer_weights_path,
            modelClass=AceStepConditionGenerationModel,
            defaultConfigPath=transformer_config_path,
            default_dtype=self.dtype,
            ignore_unused_weights=True,
        )
        self.ace_step_transformer.eval()
        self.model = self.ace_step_transformer

        self._patch_oobleck_weight_norm()
        self.audio_vae = offload.fast_load_transformers_model(
            vae_weights_path,
            modelClass=AutoencoderOobleck,
            defaultConfigPath=vae_config_path,
            default_dtype=self.dtype,
            ignore_unused_weights=True,
        )
        self.audio_vae.eval()
        self.audio_vae._offload_hooks = ["encode", "decode"]
        self.audio_vae.get_VAE_tile_size = _ace_step15_get_vae_tile_size
        self.vae = self.audio_vae

    @staticmethod
    def _patch_oobleck_weight_norm():
        try:
            from torch.nn.utils import parametrizations
            from diffusers.models.autoencoders import autoencoder_oobleck
            autoencoder_oobleck.weight_norm = parametrizations.weight_norm
        except Exception:
            return

    def _init_lm_hint_modules(self):
        self._lm_hint_quantizer = None
        self._lm_hint_detokenizer = None
        try:
            quantizer = self.ace_step_transformer.tokenizer.quantizer
            detokenizer = self.ace_step_transformer.detokenizer
        except AttributeError:
            return

        try:
            self._lm_hint_quantizer = copy.deepcopy(quantizer).to(device="cpu", dtype=torch.float32).eval()
            self._lm_hint_detokenizer = copy.deepcopy(detokenizer).to(device="cpu").eval()
            for p in self._lm_hint_quantizer.parameters():
                p.requires_grad_(False)
            for p in self._lm_hint_detokenizer.parameters():
                p.requires_grad_(False)
        except Exception:
            self._lm_hint_quantizer = None
            self._lm_hint_detokenizer = None

        self.audio_sample_rate = 48000
        for attr in ("sampling_rate", "sample_rate"):
            if hasattr(self.audio_vae, "config") and hasattr(self.audio_vae.config, attr):
                self.audio_sample_rate = int(getattr(self.audio_vae.config, attr))
                break

    def _load_tokenizers(self):
        self.pre_text_tokenizer = AutoTokenizer.from_pretrained(
            self.text_encoder_2_tokenizer_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.pre_text_tokenizer.pad_token_id is None:
            self.pre_text_tokenizer.pad_token = self.pre_text_tokenizer.eos_token or self.pre_text_tokenizer.unk_token
        self.pre_text_tokenizer.padding_side = "right"
        self.lm_tokenizer = None
        if not self.enable_lm:
            return
        try:
            from shared.utils.transformers_fast_tokenizer_patch import load_cached_lm_tokenizer
            self.lm_tokenizer = load_cached_lm_tokenizer(
                self.lm_tokenizer_dir,
                lambda: AutoTokenizer.from_pretrained(
                    self.lm_tokenizer_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                ),
            )
        except Exception:
            self.lm_tokenizer = AutoTokenizer.from_pretrained(
                self.lm_tokenizer_dir,
                local_files_only=True,
                trust_remote_code=True,
            )
        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token or self.lm_tokenizer.unk_token
        self.lm_tokenizer.padding_side = "left"
        self._build_audio_code_vocab()

    def _load_text_encoder_2(self):
        config_path = os.path.join(os.path.dirname(self.text_encoder_2_weights_path), "config.json")
        self.text_encoder_2 = offload.fast_load_transformers_model(
            self.text_encoder_2_weights_path,
            modelClass=Qwen3Model,
            defaultConfigPath=config_path,
            default_dtype=self.dtype,
            ignore_unused_weights=True,
        )
        self.text_encoder_2.eval()

    def _load_lm(self):
        config_path = os.path.join(os.path.dirname(self.lm_weights_path), "config.json")
        def _remap_lm_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
            # AceStep 5Hz LM weights are stored without a `model.` prefix.
            if any(key.startswith("model.") for key in state_dict.keys()):
                if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
                    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
                return state_dict, quantization_map, tied_weights_map
            remapped = {f"model.{key}": value for key, value in state_dict.items()}
            if "model.embed_tokens.weight" in remapped and "lm_head.weight" not in remapped:
                remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]
            return remapped, quantization_map, tied_weights_map

        self.lm_model = offload.fast_load_transformers_model(
            self.lm_weights_path,
            modelClass=Qwen3ForCausalLM,
            defaultConfigPath=config_path,
            default_dtype=self.dtype,
            preprocess_sd=_remap_lm_state_dict,
            ignore_unused_weights=True,
        )
        self.lm_model.eval()

    def _load_silence_latent(self):
        if not self.silence_latent_path or not os.path.isfile(self.silence_latent_path):
            self.silence_latent = None
            return
        self.silence_latent = torch.load(self.silence_latent_path, map_location="cpu")

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _should_abort(self) -> bool:
        return self._abort_requested() or self._early_stop_requested()

    def _encode_prompt(self, prompt: str, max_length: int = 256, use_embed_tokens: bool = False):
        tokens = self.pre_text_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device).bool()
        with torch.no_grad():
            if use_embed_tokens:
                hidden_states = self.text_encoder_2.embed_tokens(input_ids)
            else:
                outputs = self.text_encoder_2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                hidden_states = outputs.last_hidden_state
        return hidden_states, attention_mask

    def _build_audio_code_vocab(self):
        vocab = self.lm_tokenizer.get_vocab()
        vocab_size = len(vocab)
        audio_code_ids = []
        audio_code_map = {}
        max_code = self._get_audio_code_max()
        self._audio_code_max = max_code
        for token_text, token_id in vocab.items():
            match = _AUDIO_CODE_TOKEN_RE.match(token_text)
            if match:
                code_val = int(match.group(1))
                if max_code is not None and code_val > max_code:
                    continue
                audio_code_ids.append(token_id)
                audio_code_map[token_id] = code_val
        self._audio_code_token_ids = audio_code_ids
        self._audio_code_token_map = audio_code_map
        mask = torch.full((vocab_size,), float("-inf"))
        if audio_code_ids:
            mask[audio_code_ids] = 0
        self._audio_code_mask = mask

    def _get_audio_code_max(self):
        config = getattr(self, "ace_step_transformer", None)
        if config is None:
            return None
        config = getattr(self.ace_step_transformer, "config", None)
        if config is None:
            return None
        levels = getattr(config, "fsq_input_levels", None)
        if not levels:
            return None
        total = 1
        for level in levels:
            try:
                level_int = int(level)
            except (TypeError, ValueError):
                return None
            if level_int <= 0:
                return None
            total *= level_int
        if total <= 0:
            return None
        return total - 1

    def _parse_audio_code_string(self, code_str):
        if not code_str:
            return []
        try:
            vals = [int(x) for x in _AUDIO_CODE_RE.findall(str(code_str))]
            max_code = getattr(self, "_audio_code_max", None)
            if max_code is not None:
                vals = [v for v in vals if 0 <= v <= max_code]
            return vals
        except Exception:
            return []

    def _has_meaningful_negative_prompt(self, negative_prompt: str) -> bool:
        return bool(negative_prompt and negative_prompt.strip() and negative_prompt.strip() != "NO USER INPUT")

    def _format_lm_metadata_as_cot(self, metadata: dict) -> str:
        cot_items = {}
        for key in ("bpm", "caption", "duration", "keyscale", "language", "timesignature"):
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if key == "timesignature" and isinstance(value, str) and value.endswith("/4"):
                    value = value.split("/")[0]
                if isinstance(value, str) and value.isdigit():
                    value = int(value)
                cot_items[key] = value
        if cot_items:
            cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()
        else:
            cot_yaml = ""
        return f"<think>\n{cot_yaml}\n</think>"

    def _build_lm_prompt_with_cot(
        self,
        caption: str,
        lyrics: str,
        cot_text: str,
        is_negative_prompt: bool = False,
        negative_prompt: str = "NO USER INPUT",
    ) -> str:
        if is_negative_prompt:
            has_negative = self._has_meaningful_negative_prompt(negative_prompt)
            cot_for_prompt = "<think>\n</think>"
            caption_for_prompt = negative_prompt if has_negative else caption
        else:
            cot_for_prompt = cot_text
            caption_for_prompt = caption

        user_prompt = f"# Caption\n{caption_for_prompt}\n\n# Lyric\n{lyrics}\n"
        formatted = self.lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"# Instruction\n{_DEFAULT_LM_INSTRUCTION}\n\n"},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": cot_for_prompt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not formatted.endswith("\n"):
            formatted += "\n"
        return formatted

    def _format_meta(self, bpm, duration, keyscale, timesignature):
        duration_str = f"{duration} seconds" if isinstance(duration, (int, float)) else str(duration)
        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration_str}\n"
        )

    def _build_text_prompt(self, caption, meta, instruction=None):
        if instruction is None:
            instruction = _DEFAULT_DIT_INSTRUCTION
        return _SFT_GEN_PROMPT.format(instruction, caption, meta)

    def _build_lyrics_prompt(self, lyrics, language):
        return "# Languages\n{}\n\n# Lyric\n{}<|endoftext|>".format(language, lyrics)

    def _generate_audio_codes(
        self,
        tags,
        lyrics,
        bpm,
        duration,
        keyscale,
        timesignature,
        seed,
        min_tokens,
        max_tokens,
        temperature,
        top_p,
        top_k,
        language="",
        negative_prompt="NO USER INPUT",
        cfg_scale=None,
        callback=None,
    ):
        if cfg_scale is None:
            cfg_scale = 2.5

        metadata = {
            "bpm": bpm,
            "duration": duration,
            "keyscale": keyscale,
            "timesignature": timesignature,
        }
        if tags:
            metadata["caption"] = tags
        if language:
            metadata["language"] = language
        cot_text = self._format_lm_metadata_as_cot(metadata)

        prompt = self._build_lm_prompt_with_cot(tags, lyrics, cot_text, is_negative_prompt=False, negative_prompt=negative_prompt)
        prompt_negative = self._build_lm_prompt_with_cot(tags, lyrics, cot_text, is_negative_prompt=True, negative_prompt=negative_prompt)

        pos_ids = self.lm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        neg_ids = self.lm_tokenizer(prompt_negative, add_special_tokens=False)["input_ids"]

        pad_id = self.lm_tokenizer.pad_token_id or self.lm_tokenizer.eos_token_id or 0
        eos_token_id = self.lm_tokenizer.eos_token_id or pad_id

        max_len = max(len(pos_ids), len(neg_ids))
        pos_ids = ([pad_id] * (max_len - len(pos_ids))) + pos_ids
        neg_ids = ([pad_id] * (max_len - len(neg_ids))) + neg_ids

        input_ids = torch.tensor([pos_ids, neg_ids], device=self.device)
        attention_mask = (input_ids != pad_id).to(torch.long)

        generator = torch.Generator(device=self.device)
        if seed is not None and seed >= 0:
            generator.manual_seed(seed)

        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1]

        audio_codes = []
        max_new_tokens = max_tokens
        audio_code_mask = None
        if hasattr(self, "_audio_code_mask"):
            audio_code_mask = self._audio_code_mask.to(device=self.device, dtype=next_logits.dtype)

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_new_tokens,
                denoising_extra=f"LM codes 0/{max_new_tokens}",
                progress_unit="tokens",
            )

        for step in tqdm(range(max_new_tokens), total=max_new_tokens, desc="LM codes", leave=False):
            if self._should_abort():
                return None

            cond_logits = next_logits[0:1]
            uncond_logits = next_logits[1:2]
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            if audio_code_mask is not None:
                cfg_logits = cfg_logits + audio_code_mask

            if top_k is not None and top_k > 0:
                top_k_vals, _ = torch.topk(cfg_logits, top_k)
                min_val = top_k_vals[..., -1, None]
                cfg_logits[cfg_logits < min_val] = float("-inf")

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                cfg_logits[indices_to_remove] = float("-inf")

            if temperature is not None and temperature > 0:
                cfg_logits = cfg_logits / float(temperature)
                next_token = torch.multinomial(torch.softmax(cfg_logits, dim=-1), num_samples=1, generator=generator).squeeze(1)
            else:
                next_token = torch.argmax(cfg_logits, dim=-1)

            token = int(next_token.item())
            code_val = None
            if hasattr(self, "_audio_code_token_map"):
                code_val = self._audio_code_token_map.get(token)
            if code_val is None:
                token_text = self.lm_tokenizer.decode([token], skip_special_tokens=False)
                match = _AUDIO_CODE_RE.search(token_text)
                if match:
                    code_val = int(match.group(1))
            if code_val is not None:
                audio_codes.append(code_val)
                if len(audio_codes) >= max_tokens:
                    break

            next_input = torch.tensor([[token], [token]], device=self.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((2, 1), device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            with torch.no_grad():
                outputs = self.lm_model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1]

            if callback is not None:
                callback(
                    step_idx=int(step),
                    override_num_inference_steps=max_new_tokens,
                    denoising_extra=f"LM codes {step+1}/{max_new_tokens}",
                    progress_unit="tokens",
                )

        if len(audio_codes) == 0:
            return []
        if len(audio_codes) < min_tokens:
            pad_val = audio_codes[-1]
            audio_codes.extend([pad_val] * (min_tokens - len(audio_codes)))
        return audio_codes[:max_tokens]

    def _default_timbre_latents(self, length):
        if self.silence_latent is not None:
            return self._get_silence_latent(length, 1, self.device, self.dtype)
        base = torch.tensor(_DEFAULT_TIMBRE, device=self.device, dtype=self.dtype)
        base = base.view(1, 1, -1).repeat(1, max(1, length), 1)
        if base.shape[1] > length:
            base = base[:, :length, :]
        elif base.shape[1] < length:
            pad = length - base.shape[1]
            base = torch.nn.functional.pad(base, (0, 0, 0, pad))
        return base

    def _get_silence_latent(self, length, batch_size, device, dtype):
        if self.silence_latent is None:
            return torch.zeros((batch_size, length, 64), device=device, dtype=dtype)
        lat = self.silence_latent
        if lat.dim() == 2:
            lat = lat.unsqueeze(0)
        if lat.dim() == 3 and lat.shape[1] != length:
            if lat.shape[1] == 64:
                lat = lat.permute(0, 2, 1)
        if lat.shape[1] < length:
            pad = length - lat.shape[1]
            lat = torch.nn.functional.pad(lat, (0, 0, 0, pad))
        elif lat.shape[1] > length:
            lat = lat[:, :length, :]
        if lat.shape[0] != batch_size:
            lat = lat.repeat(batch_size, 1, 1)
        return lat.to(device=device, dtype=dtype)

    def _decode_audio_codes_to_latents(self, audio_codes, target_length, dtype):
        if audio_codes is None:
            return None
        if not torch.is_tensor(audio_codes):
            audio_codes = torch.tensor(audio_codes, device=self.device, dtype=torch.long)
        if audio_codes.dim() == 1:
            audio_codes = audio_codes.unsqueeze(0)
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)

        quantizer = self._lm_hint_quantizer or self.ace_step_transformer.tokenizer.quantizer
        detokenizer = self._lm_hint_detokenizer or self.ace_step_transformer.detokenizer

        def _resolve_module_device(module, fallback_device):
            for tensor in module.parameters():
                if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                    return tensor.device
                data = getattr(tensor, "_data", None)
                if torch.is_tensor(data) and data.device.type != "cpu":
                    return data.device
            for tensor in module.buffers():
                if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                    return tensor.device
            proj = getattr(module, "project_out", None)
            if proj is not None:
                for attr in ("qweight", "weight"):
                    t = getattr(proj, attr, None)
                    if torch.is_tensor(t) and t.device.type != "cpu":
                        return t.device
                    data = getattr(t, "_data", None) if t is not None else None
                    if torch.is_tensor(data) and data.device.type != "cpu":
                        return data.device
            return fallback_device

        if quantizer is self._lm_hint_quantizer:
            quantizer_device = _resolve_module_device(quantizer, next(quantizer.parameters(), torch.empty(0)).device)
        else:
            quantizer_device = self.device
        if detokenizer is self._lm_hint_detokenizer:
            detokenizer_device = _resolve_module_device(detokenizer, next(detokenizer.parameters(), torch.empty(0)).device)
        else:
            detokenizer_device = self.device
        if quantizer_device is None:
            quantizer_device = self.device
        if detokenizer_device is None:
            detokenizer_device = quantizer_device

        audio_codes = audio_codes.to(device=quantizer_device)
        quantized = quantizer.get_output_from_indices(audio_codes)
        if detokenizer_device != quantizer_device:
            quantized = quantized.to(device=detokenizer_device)
        if quantized.dtype != dtype:
            quantized = quantized.to(dtype)
        lm_hints_25hz = detokenizer(quantized)
        lm_hints_25hz = lm_hints_25hz[:, :target_length, :]
        lm_hints_25hz = lm_hints_25hz.to(device=self.device, dtype=dtype)
        return lm_hints_25hz

    def _is_silence(self, audio):
        return torch.all(audio.abs() < 1e-6)

    def _normalize_audio_to_stereo_48k(self, audio, sr):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        audio = audio[:2]
        if sr != self.audio_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(audio)
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio

    def _process_reference_audio(self, audio_path):
        if audio_path is None:
            return None
        audio, sr = torchaudio.load(audio_path)
        audio = self._normalize_audio_to_stereo_48k(audio, sr)
        if self._is_silence(audio):
            return None

        target_frames = int(30 * self.audio_sample_rate)
        segment_frames = int(10 * self.audio_sample_rate)
        if audio.shape[-1] < target_frames:
            repeat_times = int(math.ceil(target_frames / max(1, audio.shape[-1])))
            audio = audio.repeat(1, repeat_times)

        total_frames = audio.shape[-1]
        segment_size = total_frames // 3

        def _rand_start(base, avail):
            if avail <= 0:
                return base
            return base + random.randint(0, avail)

        front_start = _rand_start(0, max(0, segment_size - segment_frames))
        middle_start = _rand_start(segment_size, max(0, segment_size - segment_frames))
        back_start = _rand_start(2 * segment_size, max(0, (total_frames - 2 * segment_size) - segment_frames))

        front_audio = audio[:, front_start:front_start + segment_frames]
        middle_audio = audio[:, middle_start:middle_start + segment_frames]
        back_audio = audio[:, back_start:back_start + segment_frames]

        return torch.cat([front_audio, middle_audio, back_audio], dim=-1)

    def _process_src_audio(self, audio_path):
        if audio_path is None:
            return None
        audio, sr = torchaudio.load(audio_path)
        return self._normalize_audio_to_stereo_48k(audio, sr)

    @torch.no_grad()
    def _encode_waveform_to_latents(self, waveform, target_length, kwargs=None, pad_to_length=True):
        if waveform is None:
            return None
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform shape [1, 2, T], got {tuple(waveform.shape)}")

        def _normalize_latents(latents):
            if latents.dim() == 2:
                latents = latents.unsqueeze(0)
            if latents.dim() == 3:
                if latents.shape[-1] == 64:
                    pass
                elif latents.shape[1] == 64:
                    latents = latents.permute(0, 2, 1)
            return latents

        total_samples = waveform.shape[-1]
        duration_seconds = total_samples / float(self.audio_sample_rate) if self.audio_sample_rate else None
        tile_seconds = self._get_vae_temporal_tile_seconds(kwargs, duration_seconds)

        latents = None
        if tile_seconds is not None and tile_seconds > 0:
            tile_samples = int(round(tile_seconds * self.audio_sample_rate))
            if tile_samples > 0 and total_samples > tile_samples:
                overlap_samples = int(round(tile_samples * 0.25))
                if overlap_samples >= tile_samples:
                    overlap_samples = max(0, tile_samples // 4)
                step = max(1, tile_samples - overlap_samples)
                hop = int(self._latent_hop_length)
                overlap_frames = int(round(overlap_samples / max(1, hop)))

                for start in range(0, total_samples, step):
                    end = min(start + tile_samples, total_samples)
                    chunk = waveform[..., start:end].to(self.device)
                    encoded = self.audio_vae.encode(chunk)
                    chunk_latents = _normalize_latents(encoded.latent_dist.mode())
                    if latents is None:
                        latents = chunk_latents
                    else:
                        if overlap_frames > 0 and latents.shape[1] >= overlap_frames and chunk_latents.shape[1] >= overlap_frames:
                            fade = torch.linspace(
                                0.0,
                                1.0,
                                overlap_frames,
                                device=chunk_latents.device,
                                dtype=chunk_latents.dtype,
                            ).view(1, -1, 1)
                            latents[:, -overlap_frames:, :] = (
                                latents[:, -overlap_frames:, :] * (1.0 - fade)
                                + chunk_latents[:, :overlap_frames, :] * fade
                            )
                            latents = torch.cat([latents, chunk_latents[:, overlap_frames:, :]], dim=1)
                        else:
                            latents = torch.cat([latents, chunk_latents], dim=1)
                    del chunk
                if latents is None:
                    latents = torch.zeros((1, 1, 64), device=self.device, dtype=self.dtype)
            else:
                tile_seconds = None

        if tile_seconds is None:
            encoded = self.audio_vae.encode(waveform.to(self.device))
            latents = _normalize_latents(encoded.latent_dist.mode())

        if pad_to_length and target_length is not None:
            if latents.shape[1] > target_length:
                latents = latents[:, :target_length, :]
            elif latents.shape[1] < target_length:
                pad = target_length - latents.shape[1]
                latents = torch.nn.functional.pad(latents, (0, 0, 0, pad))
        return latents.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_reference_audio(self, audio_path, target_length, kwargs=None, pad_to_length=True, use_reference_processing=True):
        if audio_path is None:
            return None
        if use_reference_processing:
            waveform = self._process_reference_audio(audio_path)
        else:
            waveform = self._process_src_audio(audio_path)
        return self._encode_waveform_to_latents(waveform, target_length, kwargs, pad_to_length=pad_to_length)

    def _get_vae_temporal_tile_seconds(self, kwargs, duration_seconds):
        if kwargs is None:
            kwargs = {}
        if kwargs.get("vae_temporal_tiling", True) is False:
            return None
        tile_seconds = kwargs.get("vae_temporal_tile_seconds", None)
        if tile_seconds is not None:
            try:
                tile_seconds = float(tile_seconds)
            except (TypeError, ValueError):
                tile_seconds = None
            if tile_seconds is not None and tile_seconds <= 0:
                return None
        if tile_seconds is None:
            tile_choice = kwargs.get("VAE_tile_size")
            tile_size = None
            if isinstance(tile_choice, dict):
                tile_size = tile_choice.get("tile_sample_min_size")
                if tile_size is None:
                    tile_size = tile_choice.get("tile_latent_min_size")
            elif isinstance(tile_choice, (list, tuple)):
                if len(tile_choice) >= 2:
                    if isinstance(tile_choice[0], bool) and not tile_choice[0]:
                        tile_size = 0
                    else:
                        try:
                            tile_size = int(tile_choice[1])
                        except (TypeError, ValueError):
                            tile_size = None
                elif len(tile_choice) == 1:
                    try:
                        tile_size = int(tile_choice[0])
                    except (TypeError, ValueError):
                        tile_size = None
            elif isinstance(tile_choice, (int, float, bool)):
                tile_size = int(tile_choice)

            if tile_size is None:
                if not torch.cuda.is_available():
                    return None
                try:
                    device_index = self.device.index if getattr(self.device, "type", None) == "cuda" else 0
                    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
                except Exception:
                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_gb >= 24:
                    tile_seconds = 80.0
                elif total_gb >= 12:
                    tile_seconds = 40.0
                else:
                    tile_seconds = 20.0
            else:
                if tile_size <= 0:
                    tile_seconds = 80.0
                elif tile_size >= 256:
                    tile_seconds = 40.0
                else:
                    tile_seconds = 20.0
        if duration_seconds is not None and tile_seconds is not None and duration_seconds <= tile_seconds:
            return None
        return tile_seconds

    def _decode_latents_tiled(self, latents, tile_seconds, overlap_factor=0.25):
        if tile_seconds is None or tile_seconds <= 0:
            return None
        frames_per_sec = self.audio_sample_rate / float(self._latent_hop_length)
        tile_frames = int(round(tile_seconds * frames_per_sec))
        if tile_frames <= 0:
            return None
        overlap_frames = int(round(tile_frames * float(overlap_factor)))
        if overlap_frames < 0:
            overlap_frames = 0
        if overlap_frames >= tile_frames:
            overlap_frames = max(0, tile_frames // 4)

        batch_size, channels, total_frames = latents.shape
        if total_frames <= tile_frames:
            return None

        hop = int(self._latent_hop_length)
        total_samples = total_frames * hop
        step = max(1, tile_frames - overlap_frames)

        output = None
        for start in range(0, total_frames, step):
            end = min(start + tile_frames, total_frames)
            chunk = latents[:, :, start:end]
            with torch.no_grad():
                decoded = self.audio_vae.decode(chunk)
            chunk_audio = decoded.sample
            expected = (end - start) * hop
            if chunk_audio.shape[-1] > expected:
                chunk_audio = chunk_audio[..., :expected]
            elif chunk_audio.shape[-1] < expected:
                pad = expected - chunk_audio.shape[-1]
                chunk_audio = torch.nn.functional.pad(chunk_audio, (0, pad))
            if output is None:
                output = chunk_audio.new_zeros((batch_size, chunk_audio.shape[1], total_samples))
            start_sample = start * hop
            end_sample = start_sample + expected
            if start == 0 or overlap_frames == 0:
                output[..., start_sample:end_sample] = chunk_audio
            else:
                ov = min(overlap_frames * hop, start_sample, expected)
                if ov > 0:
                    fade = torch.linspace(0.0, 1.0, ov, device=chunk_audio.device, dtype=chunk_audio.dtype).view(1, 1, -1)
                    output[..., start_sample:start_sample + ov] = (
                        output[..., start_sample:start_sample + ov] * (1.0 - fade)
                        + chunk_audio[..., :ov] * fade
                    )
                    output[..., start_sample + ov:end_sample] = chunk_audio[..., ov:]
                else:
                    output[..., start_sample:end_sample] = chunk_audio
        return output

    def _build_t_schedule(self, shift, timesteps):
        valid_shifts = [1.0, 2.0, 3.0]
        valid_timesteps = [
            1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
            0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
            0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
            0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
        ]
        shift_timesteps = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
            3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
        }

        if timesteps is not None:
            t_list = timesteps.tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
            while len(t_list) > 0 and t_list[-1] == 0:
                t_list.pop()
            if len(t_list) > 20:
                t_list = t_list[:20]
            if len(t_list) >= 1:
                mapped = [min(valid_timesteps, key=lambda x: abs(x - t)) for t in t_list]
                return mapped

        shift_val = min(valid_shifts, key=lambda x: abs(x - float(shift)))
        return shift_timesteps[shift_val]

    def _sample_latents(
        self,
        noise,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        refer_audio,
        refer_audio_order_mask,
        audio_codes,
        src_latents,
        use_cover,
        non_cover_text_hidden_states,
        non_cover_text_attention_mask,
        audio_cover_strength,
        shift,
        timesteps,
        infer_method,
        callback=None,
    ):
        t_schedule_list = self._build_t_schedule(shift, timesteps)
        t_schedule = torch.tensor(t_schedule_list, device=self.device, dtype=noise.dtype)
        num_steps = len(t_schedule)

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=num_steps,
                denoising_extra=f"0/{num_steps} steps",
                progress_unit="steps",
            )

        batch_size = noise.shape[0]
        latent_length = noise.shape[1]
        silence_latent = self._get_silence_latent(latent_length, batch_size, noise.device, noise.dtype)
        src_latents_for_condition = src_latents if src_latents is not None else silence_latent
        if src_latents_for_condition.device != noise.device:
            src_latents_for_condition = src_latents_for_condition.to(noise.device)
        if src_latents_for_condition.dtype != noise.dtype:
            src_latents_for_condition = src_latents_for_condition.to(noise.dtype)
        if src_latents_for_condition.shape[1] > latent_length:
            src_latents_for_condition = src_latents_for_condition[:, :latent_length, :]
        elif src_latents_for_condition.shape[1] < latent_length:
            pad = latent_length - src_latents_for_condition.shape[1]
            src_latents_for_condition = torch.nn.functional.pad(src_latents_for_condition, (0, 0, 0, pad))

        chunk_masks = torch.ones_like(src_latents_for_condition)
        precomputed_lm_hints = None
        audio_codes_for_condition = audio_codes
        is_covers = torch.ones((batch_size,), device=noise.device, dtype=torch.long) if use_cover else torch.zeros((batch_size,), device=noise.device, dtype=torch.long)

        latent_attention_mask = torch.ones((batch_size, latent_length), device=noise.device, dtype=torch.bool)

        refer_audio_packed = refer_audio
        if refer_audio_packed.dim() == 3 and refer_audio_packed.shape[-1] != 64:
            refer_audio_packed = refer_audio_packed.permute(0, 2, 1)

        if refer_audio_order_mask is None:
            refer_audio_order_mask = torch.arange(batch_size, device=noise.device, dtype=torch.long)

        encoder_hidden_states, encoder_attention_mask, context_latents = self.ace_step_transformer.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents_for_condition,
            attention_mask=latent_attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents_for_condition,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints,
            audio_codes=audio_codes_for_condition,
        )

        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = None, None, None
        if audio_cover_strength < 1.0:
            non_is_covers = torch.zeros_like(is_covers, device=noise.device, dtype=is_covers.dtype)
            silence_latent_expanded = silence_latent[:, :latent_length, :].expand(batch_size, -1, -1)
            text_hidden_states_non_cover = text_hidden_states if non_cover_text_hidden_states is None else non_cover_text_hidden_states
            text_attention_mask_non_cover = text_attention_mask if non_cover_text_attention_mask is None else non_cover_text_attention_mask
            encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.ace_step_transformer.prepare_condition(
                text_hidden_states=text_hidden_states_non_cover,
                text_attention_mask=text_attention_mask_non_cover,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                hidden_states=silence_latent_expanded,
                attention_mask=latent_attention_mask,
                silence_latent=silence_latent,
                src_latents=silence_latent_expanded,
                chunk_masks=chunk_masks,
                is_covers=non_is_covers,
                precomputed_lm_hints_25Hz=None,
                audio_codes=None,
            )

        cover_steps = int(num_steps * audio_cover_strength)

        xt = noise
        for i, t in tqdm(enumerate(t_schedule), total=num_steps):
            if self._should_abort():
                return None
            t_tensor = t * torch.ones((batch_size,), device=xt.device, dtype=xt.dtype)
            if (encoder_hidden_states_non_cover is not None) and (i >= cover_steps):
                encoder_hidden_states = encoder_hidden_states_non_cover
                encoder_attention_mask = encoder_attention_mask_non_cover
                context_latents = context_latents_non_cover
            with torch.no_grad():
                vt = self.ace_step_transformer.decoder(
                    hidden_states=xt,
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    attention_mask=latent_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )[0]

            if i == num_steps - 1:
                xt = xt - vt * t_tensor.view(-1, 1, 1)
                break

            next_t = t_schedule[i + 1]
            if infer_method == "sde":
                pred_clean = xt - vt * t_tensor.view(-1, 1, 1)
                xt = next_t * torch.randn_like(pred_clean) + (1 - next_t) * pred_clean
            else:
                dt = t - next_t
                xt = xt - vt * dt

            if callback is not None:
                callback(
                    step_idx=int(i),
                    override_num_inference_steps=num_steps,
                    denoising_extra=f"{i+1}/{num_steps} steps",
                    progress_unit="steps",
                )

        return xt

    def generate(
        self,
        input_prompt: str,
        model_mode,
        audio_guide,
        *,
        alt_prompt=None,
        audio_guide2=None,
        audio_prompt_type="",
        temperature: float = 1.0,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False

        if not self.loaded:
            raise RuntimeError("ACE-Step 1.5 weights are not loaded.")

        lyrics = (input_prompt or "").strip()
        if not lyrics:
            raise ValueError("Lyrics prompt cannot be empty for ACE-Step 1.5.")

        tags = "" if alt_prompt is None else str(alt_prompt)

        duration_seconds = kwargs.get("duration_seconds")
        if duration_seconds is None:
            duration_seconds = kwargs.get("audio_duration")
        try:
            duration_seconds = float(duration_seconds) if duration_seconds is not None else 20.0
        except (TypeError, ValueError):
            duration_seconds = 20.0

        num_inference_steps = kwargs.get("num_inference_steps") or kwargs.get("sampling_steps") or 60
        try:
            num_inference_steps = int(num_inference_steps)
        except (TypeError, ValueError):
            num_inference_steps = 60

        guidance_scale = kwargs.get("guidance_scale", 7.0)
        try:
            guidance_scale = float(guidance_scale)
        except (TypeError, ValueError):
            guidance_scale = 7.0

        scheduler_type = model_mode or kwargs.get("scheduler_type", "euler")
        seed = kwargs.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None

        batch_size = kwargs.get("batch_size", 1)
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = 1

        bpm = int(kwargs.get("bpm", 120))
        timesignature = int(kwargs.get("timesignature", 2))
        keyscale = kwargs.get("keyscale", "C major")
        language = kwargs.get("language", "en")

        duration_int = int(math.ceil(duration_seconds))
        min_tokens = duration_int * 5

        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", None)

        callback = kwargs.get("callback")

        meta_cap = self._format_meta(bpm, duration_int, keyscale, timesignature)
        use_ref = "A" in (audio_prompt_type or "")
        use_timbre = "B" in (audio_prompt_type or "")
        has_src_audio = bool(use_ref and audio_guide)

        user_audio_codes = kwargs.get("audio_codes")
        if user_audio_codes is None:
            user_audio_codes = kwargs.get("audio_code_hints")
        if isinstance(user_audio_codes, str):
            parsed = self._parse_audio_code_string(user_audio_codes)
            user_audio_codes = parsed if parsed else None
        elif isinstance(user_audio_codes, (list, tuple)) and user_audio_codes:
            if isinstance(user_audio_codes[0], str):
                parsed = self._parse_audio_code_string(user_audio_codes[0])
                user_audio_codes = parsed if parsed else None

        audio_codes = None
        if user_audio_codes:
            max_code = getattr(self, "_audio_code_max", None)
            if max_code is not None:
                bad_codes = [v for v in user_audio_codes if v < 0 or v > max_code]
                if bad_codes:
                    raise ValueError(f"Audio codes out of range 0..{max_code}; example={bad_codes[0]}")
            audio_codes = user_audio_codes
        elif self.enable_lm and not has_src_audio:
            audio_codes = self._generate_audio_codes(
                tags=tags,
                lyrics=lyrics,
                bpm=bpm,
                duration=duration_int,
                keyscale=keyscale,
                timesignature=timesignature,
                seed=seed if seed is not None else 0,
                min_tokens=min_tokens,
                max_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                language=language,
                negative_prompt=kwargs.get("lm_negative_prompt", "NO USER INPUT"),
                cfg_scale=kwargs.get("lm_cfg_scale"),
                callback=callback,
            )
            if audio_codes is None:
                return None
            if len(audio_codes) == 0:
                raise RuntimeError("Audio code generation aborted or failed.")
            max_code = getattr(self, "_audio_code_max", None)
            if max_code is not None:
                bad_codes = [v for v in audio_codes if v < 0 or v > max_code]
                if bad_codes:
                    raise RuntimeError(f"LM generated out-of-range audio codes; example={bad_codes[0]}")

        if audio_codes is not None:
            audio_codes = torch.tensor(audio_codes, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(-1)

        use_cover = (audio_codes is not None) or has_src_audio

        instruction = _DEFAULT_LM_INSTRUCTION if use_cover else _DEFAULT_DIT_INSTRUCTION
        text_prompt = self._build_text_prompt(tags, meta_cap, instruction=instruction)
        lyrics_prompt = self._build_lyrics_prompt(lyrics, language)

        context, text_attention_mask = self._encode_prompt(text_prompt, max_length=256, use_embed_tokens=False)
        lyric_hidden, lyric_attention_mask = self._encode_prompt(lyrics_prompt, max_length=2048, use_embed_tokens=True)
        lyric_embed = lyric_hidden
        if batch_size > 1:
            context = context.repeat(batch_size, 1, 1)
            lyric_embed = lyric_embed.repeat(batch_size, 1, 1)
            if audio_codes is not None:
                audio_codes = audio_codes.repeat(batch_size, 1, 1)

        latent_length = int(round(duration_seconds * self.audio_sample_rate / self._latent_hop_length))
        latent_length = max(latent_length, 1)

        timbre_length = int(getattr(self.ace_step_transformer.config, "timbre_fix_frame", 750))
        default_ref = self._default_timbre_latents(timbre_length)

        src_latents = None
        if use_ref and audio_guide:
            src_latents = self._encode_reference_audio(audio_guide, latent_length, kwargs, pad_to_length=True, use_reference_processing=False)
        timbre_latents = None
        if use_timbre and audio_guide2:
            timbre_latents = self._encode_reference_audio(audio_guide2, timbre_length, kwargs, pad_to_length=True, use_reference_processing=True)

        refer_audio_latents = []
        refer_audio_order_mask = []
        if timbre_latents is not None:
            refer_audio_latents.append(timbre_latents)
            refer_audio_order_mask.append(0)
        if not refer_audio_latents:
            refer_audio_latents.append(default_ref)
            refer_audio_order_mask.append(0)

        refer_audio = torch.cat(refer_audio_latents, dim=0)
        refer_audio_order_mask = torch.tensor(refer_audio_order_mask, device=self.device, dtype=torch.long)

        if batch_size > 1:
            refer_audio = refer_audio.repeat(batch_size, 1, 1)
            refer_audio_order_mask = refer_audio_order_mask.repeat(batch_size) + torch.arange(batch_size, device=self.device).repeat_interleave(len(refer_audio_latents))

        audio_cover_strength = kwargs.get("audio_scale", 1.0)
        try:
            audio_cover_strength = float(audio_cover_strength)
        except (TypeError, ValueError):
            audio_cover_strength = 1.0
        audio_cover_strength = max(0.0, min(1.0, audio_cover_strength))
        if not use_cover:
            audio_cover_strength = 1.0

        shift = kwargs.get("shift", 1.0)
        timesteps = kwargs.get("timesteps")
        infer_method = kwargs.get("infer_method", "ode")

        silence_latent = self._get_silence_latent(latent_length, batch_size, self.device, self.dtype)

        rng = None
        if seed is not None and seed >= 0:
            rng = torch.Generator(device=self.device).manual_seed(seed)
        if rng is None:
            noise = torch.randn_like(silence_latent)
        else:
            noise = torch.randn(
                silence_latent.shape,
                device=silence_latent.device,
                dtype=silence_latent.dtype,
                generator=rng,
            )

        non_cover_text_hidden_states = None
        non_cover_text_attention_mask = None
        if use_cover and audio_cover_strength < 1.0:
            non_cover_text_prompt = self._build_text_prompt(tags, meta_cap, instruction=_DEFAULT_DIT_INSTRUCTION)
            non_cover_text_hidden_states, non_cover_text_attention_mask = self._encode_prompt(
                non_cover_text_prompt,
                max_length=256,
                use_embed_tokens=False,
            )
            if batch_size > 1:
                non_cover_text_hidden_states = non_cover_text_hidden_states.repeat(batch_size, 1, 1)
                non_cover_text_attention_mask = non_cover_text_attention_mask.repeat(batch_size, 1)

        sampled_latents = self._sample_latents(
            noise=noise,
            text_hidden_states=context,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_embed,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio=refer_audio,
            refer_audio_order_mask=refer_audio_order_mask,
            audio_codes=audio_codes,
            src_latents=src_latents,
            use_cover=use_cover,
            non_cover_text_hidden_states=non_cover_text_hidden_states,
            non_cover_text_attention_mask=non_cover_text_attention_mask,
            audio_cover_strength=audio_cover_strength,
            shift=shift,
            timesteps=timesteps,
            infer_method=infer_method,
            callback=callback,
        )

        if sampled_latents is None:
            return None

        sampled_latents = sampled_latents.permute(0, 2, 1)
        tile_seconds = self._get_vae_temporal_tile_seconds(kwargs, duration_seconds)
        tiled_audio = self._decode_latents_tiled(sampled_latents, tile_seconds)
        if tiled_audio is None:
            with torch.no_grad():
                decoded = self.audio_vae.decode(sampled_latents)
            audio = decoded.sample
        else:
            audio = tiled_audio

        target_samples = int(round(duration_seconds * self.audio_sample_rate))
        if audio.shape[-1] > target_samples:
            audio = audio[..., :target_samples]

        return {
            "x": audio,
            "audio_sampling_rate": int(self.audio_sample_rate),
        }
