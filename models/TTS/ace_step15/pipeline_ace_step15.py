"""
ACE-Step 1.5 pipeline for WanGP.
"""

import copy
import math
import os
import random
import re
from typing import Any

import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM as HfQwen3ForCausalLM, Qwen3Model

from mmgp import offload
from shared.utils.text_encoder_cache import TextEncoderCache

from .models.autoencoder_oobleck import AutoencoderOobleck
from .models.ace_step15_hf import AceStepConditionGenerationModel
from .qwen3_audio_codes import generate_audio_codes_with_engine_sampling
from .qwen3_lm_engines import Qwen3LegacyEngine, Qwen3LmEngine, Qwen3VllmEngine

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
_ACE_STEP15_MODEL_MODE_DEFAULT = 0
_ACE_STEP15_MODEL_MODE_INFER_MISSING = 1
_ACE_STEP15_MODEL_MODE_INFER_AND_REFINE = 2
_ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION = 3
_ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION = 4
_ACE_STEP15_DEFAULT_BPM = 120
_ACE_STEP15_DEFAULT_TIMESIGNATURE = 2
_ACE_STEP15_DEFAULT_KEYSCALE = "C major"
_ACE_STEP15_DURATION_MIN_SECONDS = 5
_ACE_STEP15_DURATION_MAX_SECONDS = 600
_ACE_STEP15_PHASE1_MAX_TOKENS = 512
_ACE_STEP15_ALLOWED_TIMESIGNATURES = {2, 3, 4, 6}
_ACE_STEP15_BPM_OPTIONS = [str(v) for v in range(30, 301)]
_ACE_STEP15_TIMESIGNATURE_OPTIONS = ["2", "3", "4", "6"]
_ACE_STEP15_KEYSCALE_OPTIONS = [f"{note}{accidental} {mode}" for note in ("A", "B", "C", "D", "E", "F", "G") for accidental in ("", "#", "b") for mode in ("major", "minor")]
_ACE_STEP15_VALID_LANGUAGES = (
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
)
_ACE_STEP15_VALID_LANGUAGE_SET = set(_ACE_STEP15_VALID_LANGUAGES)
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
        ignore_lm_cache_seed: bool = False,
        lm_decoder_engine: str = "legacy",
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
        self.ignore_lm_cache_seed = bool(ignore_lm_cache_seed)
        self.lm_engine = str(lm_decoder_engine or "legacy").strip().lower()
        if self.lm_engine not in {"legacy", "vllm"}:
            raise ValueError(f"Unsupported LM engine '{lm_decoder_engine}'. Expected one of: legacy, vllm.")

        self._interrupt = False
        self._early_stop = False
        self.loaded = False

        self._latent_hop_length = 1920
        self._ref_metadata_processor = None
        self._ref_metadata_processor_class = None
        self._ref_caption_postprocess_fn = None

        self._load_models(transformer_weights_path, transformer_config_path, vae_weights_path, vae_config_path)
        self._init_lm_hint_modules()
        self._load_tokenizers()
        self._load_text_encoder_2()
        self._load_silence_latent()
        self.lm_code_cache = TextEncoderCache()
        self._lm_engine_impl: Qwen3LmEngine | None = None
        self._lm_last_failure_reason = ""
        self.lm_model = None
        if self.enable_lm:
            self._load_lm()
            self._init_lm_engine()

        self.loaded = True

    def _load_models(self, transformer_weights_path, transformer_config_path, vae_weights_path, vae_config_path):
        self.ace_step_transformer = offload.fast_load_transformers_model(
            transformer_weights_path,
            modelClass=AceStepConditionGenerationModel,
            defaultConfigPath=transformer_config_path,
            default_dtype=self.dtype,
            writable_tensors=False,
            ignore_unused_weights=True,
        )
        self.ace_step_transformer.eval()
        self.model = self.ace_step_transformer

        self.audio_vae = offload.fast_load_transformers_model(
            vae_weights_path,
            modelClass=AutoencoderOobleck,
            defaultConfigPath=vae_config_path,
            default_dtype=self.dtype,
            writable_tensors=False,
            ignore_unused_weights=True,
        )
        self._fold_oobleck_decoder_weight_norm_once(self.audio_vae)
        self.audio_vae.eval()
        self.audio_vae._offload_hooks = ["encode", "decode"]
        self.audio_vae.get_VAE_tile_size = _ace_step15_get_vae_tile_size
        self.vae = self.audio_vae

    @staticmethod
    def _fold_oobleck_decoder_weight_norm_once(audio_vae):
        try:
            from torch.nn.utils import parametrize
        except Exception:
            return
        decoder = getattr(audio_vae, "decoder", None)
        if decoder is None:
            return
        folded = 0
        with torch.no_grad():
            for module in decoder.modules():
                module_parametrizations = getattr(module, "parametrizations", None)
                if module_parametrizations is None or "weight" not in module_parametrizations:
                    continue
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                folded += 1
        # if folded > 0:
        #     print(f"[ace_step15] Folded Oobleck decoder weight_norm once for {folded} layers.")

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
        print("Loading Ace Step 1.5 LM Tokenizers...")
        lm_loader = lambda: AutoTokenizer.from_pretrained(
            self.lm_tokenizer_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        cache_tag = os.path.basename(os.path.normpath(str(self.lm_tokenizer_dir or ""))) or "lm"
        try:
            from shared.utils.transformers_fast_tokenizer_patch import load_cached_lm_tokenizer
            self.lm_tokenizer = load_cached_lm_tokenizer(self.lm_tokenizer_dir, lm_loader, cache_tag=cache_tag)
            audio_token_count = self._count_audio_code_tokens(self.lm_tokenizer)
            if audio_token_count == 0:
                print("[ace_step15] LM tokenizer cache has no audio_code tokens; reloading tokenizer without stale cache.")
                rebuild_tag = f"{cache_tag}.rebuild"
                self.lm_tokenizer = load_cached_lm_tokenizer(self.lm_tokenizer_dir, lm_loader, cache_tag=rebuild_tag)
        except Exception:
            self.lm_tokenizer = lm_loader()
        if self.lm_tokenizer.pad_token_id is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token or self.lm_tokenizer.unk_token
        self.lm_tokenizer.padding_side = "left"
        self._build_audio_code_vocab()
        if len(getattr(self, "_audio_code_token_ids", [])) == 0:
            raise RuntimeError(
                f"No audio_code tokens found in LM tokenizer at '{self.lm_tokenizer_dir}'. "
                "Tokenizer cache/files are invalid for ACE-Step 1.5."
            )

    @staticmethod
    def _count_audio_code_tokens(tokenizer):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            return 0
        count = 0
        for token_text in vocab.keys():
            if _AUDIO_CODE_TOKEN_RE.match(token_text):
                count += 1
        return count

    def _load_text_encoder_2(self):
        config_path = os.path.join(os.path.dirname(self.text_encoder_2_weights_path), "config.json")
        self.text_encoder_2 = offload.fast_load_transformers_model(
            self.text_encoder_2_weights_path,
            modelClass=Qwen3Model,
            defaultConfigPath=config_path,
            default_dtype=self.dtype,
            writable_tensors=False,
            ignore_unused_weights=True,
        )
        self.text_encoder_2.eval()

    def _load_lm(self):
        config_path = os.path.join(os.path.dirname(self.lm_weights_path), "config.json")

        def _remap_lm_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
            if not next(iter(state_dict)).startswith("model."):
                state_dict = { "model." + k:v for k,v in state_dict.items() }
            embed_key = "model.embed_tokens.weight"
            if "lm_head.weight" not in state_dict and embed_key in state_dict:
                state_dict["lm_head.weight"] = state_dict[embed_key]
            return state_dict, quantization_map, tied_weights_map

        if self.lm_engine == "vllm":
            from shared.llm_engines.nanovllm.models.qwen3 import Qwen3ForCausalLM as NanoQwen3ForCausalLM
            model_class = NanoQwen3ForCausalLM
            unlimited_budget = True
        else:
            model_class = HfQwen3ForCausalLM
            unlimited_budget = False

        self.lm_model = offload.fast_load_transformers_model(
            self.lm_weights_path,
            modelClass=model_class,
            defaultConfigPath=config_path,
            default_dtype=self.dtype,
            preprocess_sd=_remap_lm_state_dict,
            writable_tensors=False,
            ignore_unused_weights=True,
        )
        if unlimited_budget: self.lm_model._budget = 0
        self.lm_model.eval()

    def _init_lm_engine(self):
        if self.lm_model is None or self.lm_tokenizer is None:
            raise RuntimeError(f"LM engine '{self.lm_engine}' requires loaded LM model and tokenizer.")
        common_kwargs = {"model": self.lm_model, "tokenizer": self.lm_tokenizer, "device": self.device}
        engine_builders = {
            "legacy": lambda: Qwen3LegacyEngine(**common_kwargs),
            "vllm": lambda: Qwen3VllmEngine(lm_weights_path=self.lm_weights_path, **common_kwargs),
        }
        build_engine = engine_builders.get(self.lm_engine, None)
        if build_engine is None:
            raise RuntimeError(f"Unsupported LM engine '{self.lm_engine}'.")
        self._lm_engine_impl = build_engine()

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

    @staticmethod
    def _parse_model_mode(model_mode) -> int:
        if model_mode is None:
            return _ACE_STEP15_MODEL_MODE_DEFAULT
        try:
            parsed_mode = int(model_mode)
        except Exception:
            return _ACE_STEP15_MODEL_MODE_DEFAULT
        if parsed_mode not in (
            _ACE_STEP15_MODEL_MODE_DEFAULT,
            _ACE_STEP15_MODEL_MODE_INFER_MISSING,
            _ACE_STEP15_MODEL_MODE_INFER_AND_REFINE,
            _ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION,
            _ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION,
        ):
            return _ACE_STEP15_MODEL_MODE_DEFAULT
        return parsed_mode

    @staticmethod
    def _parse_optional_int_custom_setting(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        try:
            parsed = float(str(value).strip())
        except Exception:
            return None
        if not parsed.is_integer():
            return None
        return int(parsed)

    @staticmethod
    def _normalize_optional_keyscale(value):
        if value is None:
            return None
        keyscale = str(value).strip()
        if len(keyscale) == 0:
            return None
        keyscale = keyscale.replace("♯", "#").replace("♭", "b")
        short_minor = re.fullmatch(r"([A-Ga-g])\s*([#b]?)\s*[mM]", keyscale)
        if short_minor:
            note = short_minor.group(1).upper()
            accidental = short_minor.group(2)
            return f"{note}{accidental} minor"
        full = re.fullmatch(r"([A-Ga-g])\s*([#b]?)\s*(major|minor|maj|min)", keyscale, flags=re.IGNORECASE)
        if not full:
            return keyscale
        note = full.group(1).upper()
        accidental = full.group(2)
        mode = full.group(3).lower()
        if mode == "maj":
            mode = "major"
        elif mode == "min":
            mode = "minor"
        return f"{note}{accidental} {mode}"

    @staticmethod
    def _format_known_metadata_lines(metadata: dict) -> str:
        lines = []
        for key in ("bpm", "keyscale", "timesignature"):
            value = metadata.get(key, None)
            if value is None:
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _build_lm_chat_prompt(self, system_instruction: str, user_content: str) -> str:
        formatted = self.lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not formatted.endswith("\n"):
            formatted += "\n"
        return formatted

    def _generate_lm_text(
        self,
        prompt: str,
        *,
        prompt_negative: str,
        max_tokens: int,
        temperature,
        top_p,
        top_k,
        cfg_scale: float,
        seed: int | None,
        callback=None,
        logits_processor=None,
        logits_processor_update_state=None,
        stop_checker=None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        if self._lm_engine_impl is None:
            raise RuntimeError(f"LM engine '{self.lm_engine}' is not initialized for text generation.")
        output = self._lm_engine_impl.generate_text(
            prompt=prompt,
            prompt_negative=prompt_negative,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            seed=seed,
            callback=callback,
            abort_fn=self._should_abort,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            stop_checker=stop_checker,
            progress_label=progress_label,
            release_vram_after=release_vram_after,
            ignore_eos=ignore_eos,
        )
        self._lm_last_failure_reason = ""
        return output

    def _ensure_reference_phase1_modules(self):
        if self._ref_metadata_processor_class is not None:
            return
        from .constrained_logits_processor import MetadataConstrainedLogitsProcessor

        self._ref_metadata_processor_class = MetadataConstrainedLogitsProcessor
        postprocess_caption = getattr(MetadataConstrainedLogitsProcessor, "postprocess_caption", None)
        if callable(postprocess_caption):
            self._ref_caption_postprocess_fn = postprocess_caption
        else:
            self._ref_caption_postprocess_fn = lambda x: str(x).strip()

    def _get_reference_phase1_processor(self):
        self._ensure_reference_phase1_modules()
        if self._ref_metadata_processor is None:
            self._ref_metadata_processor = self._ref_metadata_processor_class(
                self.lm_tokenizer,
                enabled=True,
                debug=False,
                skip_genres=True,
                max_duration=600,
            )
        return self._ref_metadata_processor

    def _build_reference_phase1_prompt(self, caption: str, lyrics: str, is_negative_prompt: bool = False, negative_prompt: str = "NO USER INPUT") -> str:
        if self.lm_tokenizer is None:
            raise RuntimeError("LM tokenizer is required for ACE-Step phase1 prompt formatting.")
        if is_negative_prompt:
            has_negative = bool(negative_prompt and str(negative_prompt).strip() and str(negative_prompt).strip() != "NO USER INPUT")
            if has_negative:
                user_prompt = f"# Caption\n{negative_prompt}\n\n# Lyric\n{lyrics}\n"
            else:
                user_prompt = f"# Lyric\n{lyrics}\n"
        else:
            user_prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
        return self.lm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"# Instruction\n{_DEFAULT_LM_INSTRUCTION}\n\n"},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_reference_lm_output(self, output_text: str):
        metadata = {}
        audio_codes = ""
        code_matches = re.findall(r"<\|audio_code_\d+\|>", str(output_text or ""))
        if code_matches:
            audio_codes = "".join(code_matches)
        reasoning_text = None
        for pattern in (r"<think>(.*?)</think>", r"<reasoning>(.*?)</reasoning>"):
            match = re.search(pattern, str(output_text or ""), re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                break
        if not reasoning_text:
            reasoning_text = str(output_text or "").split("<|audio_code_")[0].strip()
        if reasoning_text:
            lines = reasoning_text.split("\n")
            current_key = None
            current_value_lines = []

            def _save_current_field():
                nonlocal current_key, current_value_lines
                if current_key and current_value_lines:
                    value = "\n".join(current_value_lines)
                    if current_key == "bpm":
                        try:
                            metadata["bpm"] = int(str(value).strip())
                        except Exception:
                            metadata["bpm"] = str(value).strip()
                    elif current_key == "caption":
                        metadata["caption"] = self._ref_caption_postprocess_fn(value) if self._ref_caption_postprocess_fn is not None else str(value).strip()
                    elif current_key == "duration":
                        try:
                            metadata["duration"] = int(str(value).strip())
                        except Exception:
                            metadata["duration"] = str(value).strip()
                    elif current_key == "genres":
                        metadata["genres"] = str(value).strip()
                    elif current_key == "keyscale":
                        metadata["keyscale"] = str(value).strip()
                    elif current_key == "language":
                        metadata["language"] = str(value).strip()
                    elif current_key == "timesignature":
                        metadata["timesignature"] = str(value).strip()
                current_key = None
                current_value_lines = []

            for line in lines:
                if line.strip().startswith("<"):
                    continue
                if line and not line[0].isspace() and ":" in line:
                    _save_current_field()
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        current_key = parts[0].strip().lower()
                        first_value = parts[1]
                        if first_value.strip():
                            current_value_lines.append(first_value)
                elif line.startswith(" ") or line.startswith("\t"):
                    if current_key:
                        current_value_lines.append(line)
            _save_current_field()
        return metadata, audio_codes

    def _run_phase1_metadata(
        self,
        tags: str,
        lyrics: str,
        metadata: dict,
        refine_caption: bool,
        infer_language: bool,
        seed: int | None,
        cfg_scale: float,
        negative_prompt: str,
        temperature,
        top_p,
        top_k,
        callback=None,
        release_vram_after: bool = True,
    ):
        phase1_metadata = dict(metadata)
        if self._should_abort():
            return None, None
        processor = self._get_reference_phase1_processor()
        processor.reset()
        processor.enabled = True
        processor.debug = False
        processor.set_target_duration(None)

        constrained_metadata = {}
        if phase1_metadata.get("bpm", None) is not None:
            constrained_metadata["bpm"] = str(int(phase1_metadata["bpm"]))
        if phase1_metadata.get("keyscale", None) is not None:
            constrained_metadata["keyscale"] = str(phase1_metadata["keyscale"])
        if phase1_metadata.get("timesignature", None) is not None:
            constrained_metadata["timesignature"] = str(int(phase1_metadata["timesignature"]))
        if phase1_metadata.get("language", None) is not None:
            constrained_metadata["language"] = str(phase1_metadata["language"]).strip().lower()
        processor.set_user_metadata(constrained_metadata if len(constrained_metadata) > 0 else None)
        processor.set_stop_at_reasoning(True)
        processor.set_skip_genres(True)
        processor.set_skip_caption(not refine_caption)
        processor.set_skip_language(not infer_language)
        processor.set_generation_phase("cot")

        prompt = self._build_reference_phase1_prompt(tags, lyrics, is_negative_prompt=False, negative_prompt=negative_prompt)
        prompt_negative = self._build_reference_phase1_prompt(tags, lyrics, is_negative_prompt=True, negative_prompt=negative_prompt)
        metadata_seed = None if seed is None else int(seed)
        text_out = self._generate_lm_text(
            prompt=prompt,
            prompt_negative=prompt_negative,
            max_tokens=_ACE_STEP15_PHASE1_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            seed=metadata_seed,
            callback=callback,
            logits_processor=processor,
            logits_processor_update_state=processor.update_state,
            stop_checker=None,
            progress_label="LM Compute Metadata",
            release_vram_after=release_vram_after,
        )
        if text_out is None:
            return None, None
        if self._should_abort():
            return None, None

        phase1_text = text_out.get("text", "") if isinstance(text_out, dict) else ""
        parsed_metadata, _ = self._parse_reference_lm_output(phase1_text)

        bpm_value = self._parse_optional_int_custom_setting(parsed_metadata.get("bpm", None))
        if bpm_value is not None:
            if bpm_value < 30 or bpm_value > 300:
                raise RuntimeError(f"LM phase1 produced unsupported bpm '{bpm_value}'.")
            phase1_metadata["bpm"] = bpm_value

        keyscale_value = self._normalize_optional_keyscale(parsed_metadata.get("keyscale", None))
        if keyscale_value is not None:
            if keyscale_value not in _ACE_STEP15_KEYSCALE_OPTIONS:
                raise RuntimeError(f"LM phase1 produced unsupported keyscale '{keyscale_value}'.")
            phase1_metadata["keyscale"] = keyscale_value

        timesig_raw = parsed_metadata.get("timesignature", None)
        if timesig_raw is not None:
            timesig_text = str(timesig_raw).strip()
            if "/" in timesig_text:
                timesig_text = timesig_text.split("/", 1)[0].strip()
            timesig_value = self._parse_optional_int_custom_setting(timesig_text)
            if timesig_value is None or timesig_value not in _ACE_STEP15_ALLOWED_TIMESIGNATURES:
                raise RuntimeError(f"LM phase1 produced unsupported timesignature '{timesig_raw}'.")
            phase1_metadata["timesignature"] = timesig_value

        duration_value = self._parse_optional_int_custom_setting(parsed_metadata.get("duration", None))
        if duration_value is not None:
            phase1_metadata["duration"] = int(duration_value)

        language_raw = parsed_metadata.get("language", None)
        if language_raw is not None:
            language_value = str(language_raw).strip().lower()
            if len(language_value) > 0:
                if language_value not in _ACE_STEP15_VALID_LANGUAGE_SET:
                    raise RuntimeError(f"LM phase1 produced unsupported language '{language_raw}'.")
                phase1_metadata["language"] = language_value

        refined_caption = None
        if refine_caption:
            refined_caption = str(parsed_metadata.get("caption", "")).strip()
            if "<" in refined_caption or ">" in refined_caption:
                raise RuntimeError(f"LM phase1 produced an invalid refined caption '{refined_caption}'.")
            if len(refined_caption) == 0:
                raise RuntimeError("LM phase1 returned an empty refined caption.")

        return phase1_metadata, refined_caption

    @staticmethod
    def _normalize_duration_cache_value(duration):
        try:
            return round(float(duration), 4)
        except Exception:
            return str(duration)

    def _build_lm_cache_key(
        self,
        tags,
        lyrics,
        bpm,
        duration,
        keyscale,
        timesignature,
        min_tokens,
        max_tokens,
        temperature,
        top_p,
        top_k,
        language,
        negative_prompt,
        cfg_scale,
        seed_value,
    ):
        return (
            "ace_step15_lm_codes_v1",
            os.path.abspath(self.lm_weights_path) if self.lm_weights_path else "",
            os.path.abspath(self.lm_tokenizer_dir) if self.lm_tokenizer_dir else "",
            _DEFAULT_LM_INSTRUCTION,
            self.lm_engine,
            str(tags or ""),
            str(lyrics or ""),
            int(bpm),
            self._normalize_duration_cache_value(duration),
            str(keyscale),
            int(timesignature),
            str(language or ""),
            str(negative_prompt or ""),
            float(cfg_scale),
            float(temperature) if temperature is not None else None,
            float(top_p) if top_p is not None else None,
            int(top_k) if top_k is not None else None,
            int(min_tokens),
            int(max_tokens),
            int(seed_value) if seed_value is not None else None,
        )

    def _reserve_lm_runtime_for_max_duration(self, tags: str, lyrics: str, negative_prompt: str, lm_cfg_scale):
        if self._lm_engine_impl is None or self.lm_tokenizer is None:
            return
        reserve_fn = getattr(self._lm_engine_impl, "reserve_runtime", None)
        if not callable(reserve_fn):
            return
        cfg_phase1 = 1.0 if lm_cfg_scale is None else float(lm_cfg_scale)
        cfg_phase2 = 2.5 if lm_cfg_scale is None else float(lm_cfg_scale)
        reserve_cfg = max(1.0, cfg_phase1, cfg_phase2)
        max_duration_seconds = int(_ACE_STEP15_DURATION_MAX_SECONDS)
        phase1_max_tokens = int(_ACE_STEP15_PHASE1_MAX_TOKENS)
        phase2_max_tokens = max(1, max_duration_seconds * 5)
        reserve_max_tokens = max(phase1_max_tokens, phase2_max_tokens)

        phase1_prompt = self._build_reference_phase1_prompt(tags, lyrics, is_negative_prompt=False, negative_prompt=negative_prompt)
        phase1_prompt_negative = self._build_reference_phase1_prompt(tags, lyrics, is_negative_prompt=True, negative_prompt=negative_prompt)
        reserve_metadata = {
            "bpm": _ACE_STEP15_DEFAULT_BPM,
            "duration": max_duration_seconds,
            "keyscale": _ACE_STEP15_DEFAULT_KEYSCALE,
            "timesignature": _ACE_STEP15_DEFAULT_TIMESIGNATURE,
            "caption": tags,
        }
        reserve_cot = self._format_lm_metadata_as_cot(reserve_metadata)
        phase2_prompt = self._build_lm_prompt_with_cot(tags, lyrics, reserve_cot, is_negative_prompt=False, negative_prompt=negative_prompt)
        phase2_prompt_negative = self._build_lm_prompt_with_cot(tags, lyrics, reserve_cot, is_negative_prompt=True, negative_prompt=negative_prompt)

        prompt_len = 0
        for text in (phase1_prompt, phase1_prompt_negative, phase2_prompt, phase2_prompt_negative):
            try:
                prompt_len = max(prompt_len, len(self.lm_tokenizer.encode(text)))
            except Exception:
                pass
        if prompt_len <= 0:
            prompt_len = 256
        reserve_fn(prompt_len=prompt_len, max_tokens=reserve_max_tokens, cfg_scale=reserve_cfg)

    def _generate_audio_codes_uncached(
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
        offloadobj=None,
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

        if self._lm_engine_impl is None:
            raise RuntimeError(f"LM engine '{self.lm_engine}' is not initialized.")
        audio_codes, failure_reason = generate_audio_codes_with_engine_sampling(
            engine=self._lm_engine_impl,
            tokenizer=self.lm_tokenizer,
            prompt=prompt,
            prompt_negative=prompt_negative,
            audio_code_mask=getattr(self, "_audio_code_mask", None),
            audio_code_token_map=getattr(self, "_audio_code_token_map", {}),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            seed=seed,
            callback=callback,
            abort_fn=self._should_abort,
            release_vram_after=True,
        )
        self._lm_last_failure_reason = failure_reason
        return audio_codes

    def _release_lm_runtime(self):
        if self._lm_engine_impl is None:
            return
        release_fn = getattr(self._lm_engine_impl, "release_runtime_allocations", None)
        if not callable(release_fn):
            return
        try:
            release_fn()
        except Exception:
            pass

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
        offloadobj=None,
    ):
        if cfg_scale is None:
            cfg_scale = 2.5

        cache_seed = seed if seed is not None and seed >= 0 else None
        use_cache = self.lm_code_cache is not None and cache_seed is not None
        if not use_cache:
            return self._generate_audio_codes_uncached(
                tags=tags,
                lyrics=lyrics,
                bpm=bpm,
                duration=duration,
                keyscale=keyscale,
                timesignature=timesignature,
                seed=seed,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                language=language,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                callback=callback,
                offloadobj=offloadobj,
            )

        effective_seed = cache_seed
        cache_key = self._build_lm_cache_key(
            tags=tags,
            lyrics=lyrics,
            bpm=bpm,
            duration=duration,
            keyscale=keyscale,
            timesignature=timesignature,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            language=language,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            seed_value=cache_seed,
        )

        def encode_fn(_prompts):
            codes = self._generate_audio_codes_uncached(
                tags=tags,
                lyrics=lyrics,
                bpm=bpm,
                duration=duration,
                keyscale=keyscale,
                timesignature=timesignature,
                seed=effective_seed,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                language=language,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                callback=callback,
                offloadobj=offloadobj,
            )
            if codes is None:
                return [None]
            return [torch.tensor(codes, dtype=torch.int32)]

        cached = self.lm_code_cache.encode(
            encode_fn,
            "lm_codes",
            device="cpu",
            cache_keys=cache_key,
        )[0]
        if cached is None:
            if hasattr(self.lm_code_cache, "_entries"):
                self.lm_code_cache._entries.pop(cache_key, None)
            return None
        # Never keep empty LM-code cache entries: force uncached regeneration so
        # diagnostics from the LM engine are visible instead of replaying [].
        if torch.is_tensor(cached) and cached.numel() == 0:
            if hasattr(self.lm_code_cache, "_entries"):
                self.lm_code_cache._entries.pop(cache_key, None)
            return self._generate_audio_codes_uncached(
                tags=tags,
                lyrics=lyrics,
                bpm=bpm,
                duration=duration,
                keyscale=keyscale,
                timesignature=timesignature,
                seed=effective_seed,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                language=language,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                callback=callback,
                offloadobj=offloadobj,
            )
        if not torch.is_tensor(cached):
            cached_list = list(cached)
            if len(cached_list) == 0:
                if hasattr(self.lm_code_cache, "_entries"):
                    self.lm_code_cache._entries.pop(cache_key, None)
                return self._generate_audio_codes_uncached(
                    tags=tags,
                    lyrics=lyrics,
                    bpm=bpm,
                    duration=duration,
                    keyscale=keyscale,
                    timesignature=timesignature,
                    seed=effective_seed,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    language=language,
                    negative_prompt=negative_prompt,
                    cfg_scale=cfg_scale,
                    callback=callback,
                    offloadobj=offloadobj,
                )
        if callback is not None:
            callback(
                step_idx=0,
                override_num_inference_steps=1,
                denoising_extra="Compute Audio Codes cached",
                progress_unit="tokens",
            )
        if torch.is_tensor(cached):
            self._release_lm_runtime()
            return cached.tolist()
        self._release_lm_runtime()
        return list(cached)

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
        with tqdm(enumerate(t_schedule), total=num_steps) as pbar:
            for i, t in pbar:
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
                else:
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
        duration_seconds=None,
        audio_duration=None,
        num_inference_steps=None,
        sampling_steps=None,
        guidance_scale=7.0,
        scheduler_type="euler",
        seed=None,
        batch_size=1,
        custom_settings=None,
        language="en",
        top_p=0.9,
        top_k=None,
        callback=None,
        audio_codes=None,
        audio_code_hints=None,
        lm_negative_prompt="NO USER INPUT",
        lm_cfg_scale=None,
        offloadobj=None,
        audio_scale=1.0,
        shift=1.0,
        timesteps=None,
        infer_method="ode",
        vae_temporal_tiling=True,
        vae_temporal_tile_seconds=None,
        VAE_tile_size=None,
        set_header_text=None,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False

        if not self.loaded:
            raise RuntimeError("ACE-Step 1.5 weights are not loaded.")

        lyrics = (input_prompt or "").strip()
        if not lyrics:
            raise ValueError("Lyrics prompt cannot be empty for ACE-Step 1.5.")
        instrumental_only = lyrics.lower() == "[instrumental]"

        tags = "" if alt_prompt is None else str(alt_prompt)

        if duration_seconds is None:
            duration_seconds = audio_duration
        if duration_seconds is None or duration_seconds <= 0:
            duration_seconds = 20.0

        if num_inference_steps is None:
            num_inference_steps = sampling_steps
        if num_inference_steps is None or num_inference_steps <= 0:
            num_inference_steps = 60

        if guidance_scale is None:
            guidance_scale = 7.0

        _ = scheduler_type
        model_mode_value = self._parse_model_mode(model_mode)
        if seed is not None and seed < 0:
            seed = None

        if batch_size is None or batch_size <= 0:
            batch_size = 1

        if not isinstance(custom_settings, dict):
            custom_settings = {}
        set_progress_status = kwargs.get("set_progress_status", None)

        def _update_progress_status(status_text):
            if callable(set_progress_status):
                try:
                    set_progress_status(status_text)
                except Exception:
                    pass

        def _phase_callback(
            step_idx=-1,
            latent=None,
            force_refresh=True,
            read_state=False,
            override_num_inference_steps=-1,
            pass_no=-1,
            preview_meta=None,
            denoising_extra="",
            progress_unit=None,
        ):
            if callback is None:
                return
            callback(
                step_idx=step_idx,
                latent=latent,
                force_refresh=force_refresh,
                read_state=True,
                override_num_inference_steps=override_num_inference_steps,
                pass_no=pass_no,
                preview_meta=preview_meta,
                denoising_extra=denoising_extra,
                progress_unit=progress_unit,
            )

        user_bpm = self._parse_optional_int_custom_setting(custom_settings.get("bpm", None))
        user_timesignature = self._parse_optional_int_custom_setting(custom_settings.get("timesignature", None))
        if user_timesignature is not None and user_timesignature not in _ACE_STEP15_ALLOWED_TIMESIGNATURES:
            user_timesignature = None
        user_keyscale = self._normalize_optional_keyscale(custom_settings.get("keyscale", None))
        if isinstance(user_keyscale, str):
            user_keyscale = user_keyscale.strip()
            if len(user_keyscale) == 0:
                user_keyscale = None
        user_language = custom_settings.get("language", None)
        if user_language is not None:
            user_language = str(user_language).strip().lower()
            if len(user_language) == 0:
                user_language = None
        if user_language is None and instrumental_only:
            user_language = "unknown"
        use_ref = "A" in (audio_prompt_type or "")
        has_src_audio = bool(use_ref and audio_guide)
        user_audio_codes = audio_codes if audio_codes is not None else audio_code_hints
        if isinstance(user_audio_codes, str):
            parsed = self._parse_audio_code_string(user_audio_codes)
            user_audio_codes = parsed if parsed else None
        elif isinstance(user_audio_codes, (list, tuple)) and user_audio_codes and isinstance(user_audio_codes[0], str):
            parsed = self._parse_audio_code_string(user_audio_codes[0])
            user_audio_codes = parsed if parsed else None
        keep_lm_loaded_for_phase2 = bool(
            self.enable_lm
            and (not has_src_audio)
            and (not user_audio_codes)
            and bool(getattr(self._lm_engine_impl, "keep_loaded_for_phase2", False))
        )

        phase1_metadata = {
            "bpm": user_bpm,
            "keyscale": user_keyscale,
            "timesignature": user_timesignature,
            "language": user_language,
        }
        lm_runtime_retained = keep_lm_loaded_for_phase2
        try:
            if self.enable_lm:
                try:
                    self._reserve_lm_runtime_for_max_duration(tags, lyrics, lm_negative_prompt, lm_cfg_scale)
                except Exception:
                    pass
            tags_for_generation = tags
            computed_phase1_metadata = {}
            refined_caption = None
            if model_mode_value == _ACE_STEP15_MODEL_MODE_DEFAULT:
                bpm = phase1_metadata["bpm"] if phase1_metadata["bpm"] is not None else _ACE_STEP15_DEFAULT_BPM
                timesignature = phase1_metadata["timesignature"] if phase1_metadata["timesignature"] is not None else _ACE_STEP15_DEFAULT_TIMESIGNATURE
                if timesignature not in _ACE_STEP15_ALLOWED_TIMESIGNATURES:
                    timesignature = _ACE_STEP15_DEFAULT_TIMESIGNATURE
                keyscale = phase1_metadata["keyscale"] if phase1_metadata["keyscale"] is not None else _ACE_STEP15_DEFAULT_KEYSCALE
                keyscale = str(keyscale).strip()
                if len(keyscale) == 0:
                    keyscale = _ACE_STEP15_DEFAULT_KEYSCALE
                language = phase1_metadata["language"] if phase1_metadata["language"] is not None else "en"
            else:
                if not self.enable_lm:
                    raise RuntimeError("ACE-Step 1.5 model mode 1/2/3/4 requires an LM definition (text_encoder_URLs).")
                missing_fields = [field_name for field_name in ("bpm", "keyscale", "timesignature", "language") if phase1_metadata.get(field_name) is None]
                infer_language = phase1_metadata.get("language", None) is None
                infer_duration = model_mode_value in (_ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION, _ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION)
                refine_caption = model_mode_value in (_ACE_STEP15_MODEL_MODE_INFER_AND_REFINE, _ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION)
                run_phase1 = model_mode_value in (
                    _ACE_STEP15_MODEL_MODE_INFER_AND_REFINE,
                    _ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION,
                    _ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION,
                ) or (
                    model_mode_value == _ACE_STEP15_MODEL_MODE_INFER_MISSING and len(missing_fields) > 0
                )
                if run_phase1:
                    phase1_seed = seed if seed is not None else 0
                    _update_progress_status("LM Compute Metadata")
                    phase1_metadata, refined_caption = self._run_phase1_metadata(
                        tags=tags,
                        lyrics=lyrics,
                        metadata=phase1_metadata,
                        refine_caption=refine_caption,
                        infer_language=infer_language,
                        seed=phase1_seed,
                        cfg_scale=1.0 if lm_cfg_scale is None else float(lm_cfg_scale),
                        negative_prompt=lm_negative_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        callback=_phase_callback,
                        release_vram_after=not keep_lm_loaded_for_phase2,
                    )
                    for field_name in missing_fields:
                        value = phase1_metadata.get(field_name, None) if phase1_metadata is not None else None
                        if value is not None:
                            computed_phase1_metadata[field_name] = value
                if phase1_metadata is None:
                    return None
                if infer_duration:
                    duration_value = self._parse_optional_int_custom_setting(phase1_metadata.get("duration", None))
                    if duration_value is None:
                        raise RuntimeError("LM phase1 failed to resolve required metadata 'duration'.")
                    duration_value = max(_ACE_STEP15_DURATION_MIN_SECONDS, min(_ACE_STEP15_DURATION_MAX_SECONDS, int(duration_value)))
                    duration_seconds = float(duration_value)
                    computed_phase1_metadata["duration"] = int(duration_value)
                if len(computed_phase1_metadata) > 0:
                    print(f"[ace_step15][phase1] computed metadata: {computed_phase1_metadata}")
                if refined_caption is not None:
                    tags_for_generation = refined_caption
                    if refined_caption.strip() != tags.strip():
                        print(f"[ace_step15][phase1] refined caption: {refined_caption}")
                if set_header_text is not None:
                    header_parts = []
                    if len(computed_phase1_metadata) > 0:
                        computed_bits = []
                        for field_name in ("bpm", "keyscale", "timesignature", "language", "duration"):
                            if field_name in computed_phase1_metadata:
                                computed_bits.append(f"{field_name}={computed_phase1_metadata[field_name]}")
                        if len(computed_bits) > 0:
                            header_parts.append("LM computed metadata: " + "<BR>- ".join(computed_bits))
                    if refined_caption is not None and refined_caption.strip() != tags.strip():
                        header_parts.append(f"<BR>- LM refined caption: {refined_caption}")
                    if len(header_parts) > 0:
                        try:
                            set_header_text("".join(header_parts))
                        except Exception:
                            pass
                for field_name in ("bpm", "keyscale", "timesignature", "language"):
                    if phase1_metadata.get(field_name, None) is None:
                        raise RuntimeError(f"LM phase1 failed to resolve required metadata '{field_name}'.")
                bpm = int(phase1_metadata["bpm"])
                timesignature = int(phase1_metadata["timesignature"])
                if timesignature not in _ACE_STEP15_ALLOWED_TIMESIGNATURES:
                    raise RuntimeError("LM phase1 produced an unsupported timesignature.")
                keyscale = str(phase1_metadata["keyscale"]).strip()
                if len(keyscale) == 0:
                    raise RuntimeError("LM phase1 produced an empty keyscale.")
                language = str(phase1_metadata["language"]).strip().lower()
                if len(language) == 0:
                    raise RuntimeError("LM phase1 produced an empty language.")
                if language not in _ACE_STEP15_VALID_LANGUAGE_SET:
                    raise RuntimeError(f"LM phase1 produced unsupported language '{language}'.")

            duration_int = int(math.ceil(duration_seconds))
            min_tokens = duration_int * 5

            meta_cap = self._format_meta(bpm, duration_int, keyscale, timesignature)
            use_timbre = "B" in (audio_prompt_type or "")

            audio_codes = None
            if user_audio_codes:
                max_code = getattr(self, "_audio_code_max", None)
                if max_code is not None:
                    bad_codes = [v for v in user_audio_codes if v < 0 or v > max_code]
                    if bad_codes:
                        raise ValueError(f"Audio codes out of range 0..{max_code}; example={bad_codes[0]}")
                audio_codes = user_audio_codes
            elif self.enable_lm and not has_src_audio:
                _update_progress_status("LM Compute Audio Codes")
                audio_codes = self._generate_audio_codes(
                    tags=tags_for_generation,
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
                    negative_prompt=lm_negative_prompt,
                    cfg_scale=lm_cfg_scale,
                    callback=_phase_callback,
                    offloadobj=offloadobj,
                )
                if audio_codes is None:
                    return None
                if self._should_abort():
                    return None
                if len(audio_codes) == 0:
                    if self._should_abort():
                        return None
                    failure_reason = str(getattr(self, "_lm_last_failure_reason", "") or "")
                    if self._lm_engine_impl is not None:
                        get_reason = getattr(self._lm_engine_impl, "get_last_failure_reason", None)
                        if callable(get_reason):
                            try:
                                engine_reason = str(get_reason() or "")
                                if len(engine_reason) > 0:
                                    failure_reason = engine_reason
                            except Exception:
                                pass
                    extra = f" last_failure={failure_reason}" if failure_reason else ""
                    raise RuntimeError(
                        "Audio code generation returned 0 usable codes "
                        f"(engine={self.lm_engine}, tokenizer_dir='{self.lm_tokenizer_dir}', "
                        f"audio_code_vocab={len(getattr(self, '_audio_code_token_ids', []))}).{extra}"
                    )
                max_code = getattr(self, "_audio_code_max", None)
                if audio_codes is not None and max_code is not None:
                    bad_codes = [v for v in audio_codes if v < 0 or v > max_code]
                    if bad_codes:
                        raise RuntimeError(f"LM generated out-of-range audio codes; example={bad_codes[0]}")
        finally:
            if lm_runtime_retained:
                self._release_lm_runtime()

        _update_progress_status("")
        if audio_codes is not None:
            audio_codes = torch.tensor(audio_codes, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(-1)

        use_cover = (audio_codes is not None) or has_src_audio

        instruction = _DEFAULT_LM_INSTRUCTION if use_cover else _DEFAULT_DIT_INSTRUCTION
        text_prompt = self._build_text_prompt(tags_for_generation, meta_cap, instruction=instruction)
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
        vae_options = {"vae_temporal_tiling": vae_temporal_tiling, "vae_temporal_tile_seconds": vae_temporal_tile_seconds, "VAE_tile_size": VAE_tile_size}
        if use_ref and audio_guide:
            src_latents = self._encode_reference_audio(audio_guide, latent_length, vae_options, pad_to_length=True, use_reference_processing=False)
        timbre_latents = None
        if use_timbre and audio_guide2:
            timbre_latents = self._encode_reference_audio(audio_guide2, timbre_length, vae_options, pad_to_length=True, use_reference_processing=True)

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

        audio_cover_strength = audio_scale
        if audio_cover_strength is None:
            audio_cover_strength = 1.0
        audio_cover_strength = max(0.0, min(1.0, audio_cover_strength))
        if not use_cover:
            audio_cover_strength = 1.0

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
            non_cover_text_prompt = self._build_text_prompt(tags_for_generation, meta_cap, instruction=_DEFAULT_DIT_INSTRUCTION)
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
        tile_seconds = self._get_vae_temporal_tile_seconds(vae_options, duration_seconds)
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

        final_custom_settings = dict(custom_settings) if isinstance(custom_settings, dict) else {}
        final_custom_settings["bpm"] = int(bpm)
        final_custom_settings["keyscale"] = str(keyscale)
        final_custom_settings["timesignature"] = int(timesignature)
        final_custom_settings["language"] = str(language)
        overridden_inputs = {
            "custom_settings": final_custom_settings,
        }
        if tags_for_generation.strip() != tags.strip():
            overridden_inputs["alt_prompt"] = tags_for_generation
            original_caption = str(tags).strip()
            if len(original_caption) > 0:
                overridden_inputs["extra_info"] = {"Original Caption": original_caption}
        if model_mode_value in (_ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION, _ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION):
            overridden_inputs["duration_seconds"] = int(duration_int)
            overridden_inputs["duration"] = int(duration_int)
        if model_mode_value in (
            _ACE_STEP15_MODEL_MODE_INFER_AND_REFINE,
            _ACE_STEP15_MODEL_MODE_INFER_REFINE_AND_DURATION,
            _ACE_STEP15_MODEL_MODE_INFER_MISSING_AND_DURATION,
        ):
            overridden_inputs["model_mode"] = _ACE_STEP15_MODEL_MODE_INFER_MISSING

        return {
            "x": audio,
            "audio_sampling_rate": int(self.audio_sample_rate),
            "overridden_inputs": overridden_inputs,
        }
