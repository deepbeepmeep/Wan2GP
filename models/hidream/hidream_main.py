import os
import json
import types

import torch
from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen2VLImageProcessorFast, Qwen2VLProcessor
from transformers.processing_utils import ProcessorMixin

from .pipeline import DEFAULT_TIMESTEPS, NOISE_SCALE, generate_image, resample_timesteps
from .qwen3_vl_configuration import register_qwen3_vl_config
from .qwen3_vl_transformers import Qwen3VLForConditionalGeneration


HIDREAM_QUANTO_BF16_EXCLUDE = [
    "model.language_model.layers.*.mlp.down_proj.weight",
    "model.language_model.layers.*.self_attn.o_proj.weight",
]


class HiDreamQwen3VLProcessor(Qwen2VLProcessor):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = tokenizer.image_token_id if getattr(tokenizer, "image_token_id", None) else tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.video_token_id if getattr(tokenizer, "video_token_id", None) else tokenizer.convert_tokens_to_ids(self.video_token)
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)


def add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def load_processor(processor_path):
    tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(processor_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_path = os.path.join(processor_path, "chat_template.json")
    if chat_template is None and os.path.isfile(chat_template_path):
        with open(chat_template_path, "r", encoding="utf-8") as reader:
            chat_template = json.load(reader).get("chat_template")
    return HiDreamQwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)


def _as_pil(image):
    return convert_tensor_to_image(image) if torch.is_tensor(image) else image


def _quantized_transformer_filename(model_filename, dtype):
    model_filename = os.path.basename(model_filename)
    if dtype == torch.bfloat16:
        model_filename = model_filename.replace("fp16", "bf16").replace("FP16", "bf16")
    elif dtype == torch.float16:
        model_filename = model_filename.replace("bf16", "fp16").replace("BF16", "fp16")

    for rep in ["mfp16", "fp16", "mbf16", "bf16"]:
        if "_" + rep in model_filename:
            return model_filename.replace("_" + rep, "_quanto_" + rep + "_int8")

    pos = model_filename.rfind(".")
    return model_filename[:pos] + "_quanto_int8" + model_filename[pos:] if pos >= 0 else model_filename + "_quanto_int8"


def save_quantized_transformer(model, model_filename, dtype, config_file):
    if "quanto" in model_filename:
        return None
    quantized_filename = _quantized_transformer_filename(model_filename, dtype)
    existing_path = fl.locate_file(quantized_filename, error_if_none=False)
    if existing_path is not None:
        print(f"There isn't any model to quantize as quantized model '{quantized_filename}' already exists")
        return existing_path

    quantized_path = fl.get_download_location(quantized_filename)
    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)
    offload.save_model(model, quantized_path, do_quantize=True, config_file_path=config_file, quantize_exclude=HIDREAM_QUANTO_BF16_EXCLUDE)
    print(f"New quantized file '{quantized_filename}' had been created.")
    return quantized_path


def _attach_lora_preprocessor(transformer):
    def preprocess_loras(self, model_type, sd):
        if not sd:
            return sd

        qwen3_model_prefixes = (
            "visual.",
            "language_model.",
            "t_embedder1.",
            "t_embedder2.",
            "x_embedder.",
            "final_layer2.",
        )
        wrapper_prefixes = ("diffusion_model.", "transformer.")
        new_sd = {}
        for key, value in sd.items():
            for wrapper_prefix in wrapper_prefixes:
                if key.startswith(wrapper_prefix):
                    inner_key = key[len(wrapper_prefix):]
                    if inner_key.startswith(qwen3_model_prefixes):
                        key = wrapper_prefix + "model." + inner_key
                    break
            else:
                if key.startswith(qwen3_model_prefixes):
                    key = "model." + key
            new_sd[key] = value
        return new_sd

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        save_quantized=False,
        **kwargs,
    ):
        model_def = model_def or {}
        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if transformer_filename is None:
            raise ValueError("No transformer filename provided for HiDream O1.")

        self.model_type = model_type
        self.base_model_type = base_model_type
        self.model_def = model_def
        self.dtype = dtype
        self._abort = False

        processor_folder = model_def.get("processor_folder", base_model_type)
        processor_path = os.path.dirname(fl.locate_file(os.path.join(processor_folder, "tokenizer_config.json")))
        config_path = fl.locate_file(os.path.join(processor_folder, "config.json"))

        register_qwen3_vl_config()
        self.processor = load_processor(processor_path)
        self.tokenizer = get_tokenizer(self.processor)
        add_special_tokens(self.tokenizer)

        source = model_def.get("source", None)
        load_filename = fl.locate_file(source) if source is not None else transformer_filename
        self.transformer = offload.fast_load_transformers_model(
            load_filename,
            writable_tensors=False,
            modelClass=Qwen3VLForConditionalGeneration,
            defaultConfigPath=config_path,
            default_dtype=dtype,
            ignore_unused_weights=True,
            do_quantize=quantizeTransformer and not save_quantized,
        )
        self.transformer.eval().requires_grad_(False)
        self.model = self.transformer
        _attach_lora_preprocessor(self.transformer)
        self._set_interrupt(False)

        if source is not None:
            from wgp import save_model

            save_model(self.transformer, model_type, dtype, config_path)

        if save_quantized:
            save_quantized_transformer(self.transformer, transformer_filename, dtype, config_path)

    def generate(
        self,
        input_prompt="",
        image_start=None,
        input_frames=None,
        input_ref_images=None,
        batch_size=1,
        height=1024,
        width=1024,
        shift=None,
        sampling_steps=50,
        guide_scale=5.0,
        seed=None,
        callback=None,
        joint_pass=True,
        original_input_ref_images=None,
        custom_settings=None,
        **kwargs
    ):
        self._set_interrupt(False)
        is_dev = self.base_model_type == "hidream_o1_dev"
        custom_settings = custom_settings or {}
        sampling_steps = int(sampling_steps)

        if seed is None or int(seed) < 0:
            seed = int(torch.seed() % (2**31 - 1))
        else:
            seed = int(seed)

        if is_dev:
            scheduler_name = "flash"
            timesteps_list = resample_timesteps(DEFAULT_TIMESTEPS, sampling_steps)
            guide_scale = 0.0
            shift = 1.0 if shift is None else shift
            noise_scale_start = float(custom_settings.get("noise_scale_start", 7.5))
            noise_scale_end = float(custom_settings.get("noise_scale_end", 7.5))
            noise_clip_std = float(custom_settings.get("noise_clip_std", 2.5))
        else:
            scheduler_name = "default"
            timesteps_list = None
            shift = 3.0 if shift is None else shift
            noise_scale_start = float(custom_settings.get("noise_scale_start", NOISE_SCALE))
            noise_scale_end = float(custom_settings.get("noise_scale_end", NOISE_SCALE))
            noise_clip_std = float(custom_settings.get("noise_clip_std", 0.0))

        ref_images = []
        if image_start is not None:
            ref_images.append(_as_pil(image_start))
        if input_frames is not None:
            ref_images.append(_as_pil(input_frames))
        image_ref_source = original_input_ref_images if original_input_ref_images else input_ref_images
        if image_ref_source is not None:
            ref_images.extend(_as_pil(img) for img in image_ref_source)

        batch_size = max(1, int(batch_size))
        with torch.inference_mode():
            try:
                images = generate_image(
                    model=self.transformer,
                    processor=self.processor,
                    prompt=input_prompt,
                    ref_images=ref_images,
                    height=height,
                    width=width,
                    num_inference_steps=sampling_steps,
                    guidance_scale=guide_scale,
                    shift=shift,
                    timesteps_list=timesteps_list,
                    scheduler_name=scheduler_name,
                    seed=seed,
                    noise_scale_start=noise_scale_start,
                    noise_scale_end=noise_scale_end,
                    noise_clip_std=noise_clip_std,
                    keep_original_aspect=False,
                    batch_size=batch_size,
                    joint_pass=joint_pass,
                    callback=callback,
                    abort_callback=lambda: self._interrupt,
                )
            finally:
                if hasattr(self.transformer, "clear_runtime_caches"):
                    self.transformer.clear_runtime_caches()
            if images is None:
                return None
            if not isinstance(images, list):
                images = [images]
            images = [convert_image_to_tensor(image) for image in images]

        return torch.stack(images, dim=1)

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    def _set_interrupt(self, value):
        self._abort = bool(value)
        for module in (
            getattr(self, "transformer", None),
            getattr(getattr(self, "transformer", None), "model", None),
            getattr(getattr(self, "transformer", None), "visual", None),
            getattr(getattr(self, "transformer", None), "language_model", None),
        ):
            if module is not None:
                setattr(module, "_interrupt", self._abort)

    @property
    def _interrupt(self):
        return self._abort

    @_interrupt.setter
    def _interrupt(self, value):
        self._set_interrupt(value)
