import os

import torch
from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
from transformers import AutoProcessor, PreTrainedTokenizerBase

from .pipeline import DEFAULT_TIMESTEPS, NOISE_SCALE, generate_image
from .qwen3_vl_transformers import Qwen3VLForConditionalGeneration


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
    offload.save_model(model, quantized_path, do_quantize=True, config_file_path=config_file)
    print(f"New quantized file '{quantized_filename}' had been created.")
    return quantized_path


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

        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
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
        self._set_interrupt(False)

        if source is not None:
            from wgp import save_model

            save_model(self.transformer, model_type, dtype, config_path)

        if save_quantized:
            save_quantized_transformer(self.transformer, transformer_filename, dtype, config_path)

    def generate(
        self,
        input_prompt="",
        alt_prompt="",
        image_start=None,
        image_end=None,
        input_frames=None,
        input_frames2=None,
        input_ref_images=None,
        input_ref_masks=None,
        input_masks=None,
        input_masks2=None,
        input_video=None,
        input_faces=None,
        input_custom=None,
        denoising_strength=1.0,
        masking_strength=1.0,
        prefix_frames_count=0,
        frame_num=1,
        batch_size=1,
        height=1024,
        width=1024,
        fit_into_canvas=None,
        shift=None,
        sample_solver="default",
        sampling_steps=50,
        guide_scale=5.0,
        guide2_scale=5.0,
        guide3_scale=5.0,
        switch_threshold=0,
        switch2_threshold=0,
        guide_phases=1,
        model_switch_phase=1,
        embedded_guidance_scale=0.0,
        n_prompt=None,
        seed=None,
        callback=None,
        enable_RIFLEx=False,
        VAE_tile_size=None,
        joint_pass=True,
        perturbation_switch=0,
        perturbation_layers=None,
        perturbation_start=0.0,
        perturbation_end=1.0,
        apg_switch=0,
        cfg_star_switch=0,
        cfg_zero_step=-1,
        alt_guide_scale=1.0,
        audio_cfg_scale=4.0,
        input_waveform=None,
        input_waveform_sample_rate=0,
        audio_guide=None,
        audio_guide2=None,
        audio_prompt_type="",
        audio_proj=None,
        audio_scale=None,
        audio_context_lens=None,
        context_scale=None,
        control_scale_alt=1.0,
        alt_scale=0.0,
        motion_amplitude=1.0,
        model_mode=0,
        causal_block_size=5,
        causal_attention=True,
        fps=1,
        overlapped_latents=None,
        return_latent_slice=False,
        overlap_noise=0,
        overlap_size=0,
        color_correction_strength=0,
        conditioning_latents_size=0,
        input_video_is_hdr=False,
        lora_dir=None,
        keep_frames_parsed=None,
        model_filename=None,
        model_type=None,
        loras_slists=None,
        NAG_scale=1.0,
        NAG_tau=3.5,
        NAG_alpha=0.5,
        speakers_bboxes=None,
        image_mode=1,
        video_prompt_type="",
        window_no=1,
        offloadobj=None,
        set_header_text=None,
        pre_video_frame=None,
        prefix_video=None,
        original_input_ref_images=None,
        image_refs_relative_size=50,
        outpainting_dims=None,
        face_arc_embeds=None,
        custom_settings=None,
        temperature=0.8,
        window_start_frame_no=0,
        input_video_strength=1.0,
        self_refiner_setting=0,
        self_refiner_plan="",
        self_refiner_f_uncertainty=0.0,
        self_refiner_certain_percentage=0.999,
        duration_seconds=0,
        pause_seconds=0,
        top_p=0.9,
        top_k=50,
        set_progress_status=None,
        loras_selected=None,
        frames_relative_positions_list=None,
        frames_to_inject=None,
        **kwargs
    ):
        self._set_interrupt(False)
        is_dev = self.base_model_type == "hidream_o1_dev"
        custom_settings = custom_settings or {}

        if seed is None or int(seed) < 0:
            seed = int(torch.seed() % (2**31 - 1))
        else:
            seed = int(seed)

        if is_dev:
            scheduler_name = "flash"
            timesteps_list = DEFAULT_TIMESTEPS
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
