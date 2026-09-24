import os
from functools import lru_cache
from urllib.parse import urlsplit

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from shared.utils.gguf_mapping import has_standard_gguf_tensor_names, remap_state_dict_triplet
from .enhancer import GENERATION, EDITING


ENCODER = "Qwen3-VL-8B-Instruct"
PROJECT = "qwen_image_21"
REPO = "DeepBeepMeep/Qwen_image_2"
ENCODER_REPO = "DeepBeepMeep/Ideogram4"
VISION_FILE = ENCODER + "_vision_bf16.safetensors"
VIGGLE_TURBO_LORA_FILENAME = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
PROCESSOR_FILES = ["config_legacy.json", "tokenizer_legacy.json", "tokenizer_config_legacy.json", "preprocessor_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "added_tokens.json", "special_tokens_map.json", "video_preprocessor_config.json"]


@lru_cache(maxsize=1)
def encoder_gguf_name_map():
    import json
    from accelerate import init_empty_weights
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    with open(fl.locate_file(os.path.join(ENCODER, "config_legacy.json")), encoding="utf-8") as reader:
        text_config = json.load(reader)["text_config"]
    # Qwen3-VL's text layers use Qwen3's GGUF names and tensor layout.
    config = Qwen3Config(**{key: text_config[key] for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim")})
    with torch.device("cpu"), init_empty_weights():
        model = Qwen3ForCausalLM(config)
    return {name: target.replace("model.", "model.language_model.", 1)
            for name, target in get_gguf_hf_weights_map(model).items()}


def encoder_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
    """Accept GGUF, shared encoder-only and conditional-generation layouts."""
    if has_standard_gguf_tensor_names(state_dict):
        state_dict, quantization_map, tied_weights_map = remap_state_dict_triplet(
            state_dict, quantization_map, tied_weights_map, encoder_gguf_name_map())

    def model_name(key):
        key = key.removeprefix("model.")
        if key.partition(".")[0] in ("embed_tokens", "layers", "norm"):
            key = "language_model." + key
        return "model." + key

    def remap(mapping):
        if mapping is None:
            return None
        return {model_name(key): ([model_name(item) for item in value if item != "lm_head.weight"] if isinstance(value, list) else value)
                for key, value in mapping.items() if key != "lm_head.weight" and key != "lm_head"}
    return remap(state_dict), remap(quantization_map), remap(tied_weights_map)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["qwen_image_21_7B"]

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen": (1110, "Qwen")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "qwen21"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "image_outputs": True,
            "viggle_turbo_lora_filename": VIGGLE_TURBO_LORA_FILENAME,
            "resolutions_categories": ["<=4096p"],
            "custom_settings": [{
                "id": "qwen21_kv_cache",
                "name": "KV Cache",
                "label": "KV Cache",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled (Slower but lower VRAM/RAM)", "Disabled"), ("Enabled (Faster but requires more VRAM/RAM)", "Enabled")],
                "info": "Disabled recomputes conditioning each denoising step without retaining per-layer K/V. Enabled caches conditioning K/V on the GPU for faster denoising. The CPU text-embedding cache remains active in both modes.",
            }, {
                "id": "rgba",
                "name": "RGBA",
                "label": "RGBA",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled", "Disabled"), ("Enabled", "Enabled")],
                "info": "Disabled saves RGB in your selected image format. Enabled preserves the generated alpha channel and saves RGBA PNG. Request transparency in the prompt when needed.",
            }],
            "specialities": [
                {"name": "text rendering", "aliases": ["text writing", "lettering"], "description": "Moderate text-rendering quality, best with short text. Spelling and layout require proofreading; not recommended for dense infographics."},
            ],
            "prompt_enhancer_def": {"selection": ["T", "TI", "T1", "TI1"], "labels": {"T": "A New Image Description using existing Text Prompt", "TI": "An Image Edit using existing Text Prompt and {image_inputs}", "T1": "An Editing Instruction using existing Text Prompt", "TI1": "A New Composition using existing Text Prompt and {image_inputs}"}, "default": ""},
            "text_prompt_enhancer_instructions": GENERATION,
            "image_prompt_enhancer_instructions": EDITING,
            "text_prompt_enhancer_instructions1": EDITING,
            "image_prompt_enhancer_instructions1": EDITING,
            "text_prompt_enhancer_max_tokens": 768,
            "image_prompt_enhancer_max_tokens": 768,
            "text_prompt_enhancer_max_tokens1": 768,
            "image_prompt_enhancer_max_tokens1": 768,
            "vae_block_size": 32,
            "sample_solvers": [("Default", "default"), ("Viggle Turbo v0.2", "viggle_v02")],
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "NAG": False,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": ["qwen21"],
            "inpaint_support": True,
            "inpaint_video_prompt_type": "VAG",
            "inpaint_with_image_ref": True,
            "image_ref_inpaint": True,
            "mask_strength_always_enabled": True,
            "inpaint_color": "FF0000",
            "guide_inpaint_color": "FF0000",
            "video_guide_outpainting": [1, 2],
            "outpainting_quantize_margins": 32,
            "guide_preprocessing": {"selection": ["", "PV", "DV", "SV", "CV", "V"], "labels": {"V": "Control Image"}},
            "mask_preprocessing": {"selection": ["", "A"], "visible": True},
            "model_modes": {"choices": [("Masked Denoising", 0), ("LanPaint (2 steps)", 2), ("LanPaint (5 steps)", 3), ("LanPaint (10 steps)", 4), ("LanPaint (15 steps)", 5)], "default": 0, "label": "Inpainting Method", "image_modes": [2]},
            "text_encoder_folder": ENCODER,
            "text_encoder_URLs": [build_hf_url(ENCODER_REPO, ENCODER, ENCODER + suffix) for suffix in ("_bf16.safetensors", "_int8_convrot.safetensors")],
            "image_ref_choices": {"choices": [("None", ""), ("First Image Is the Main Subject or Landscape", "KI"), ("Reference Images Are People or Objects", "I")], "letters_filter": "KI"},
            "infos": "**Qwen Image 2.1 (7B)** generates images from text and edits images using up to **10 reference images**. Select reference images in the order you will describe them. Use a control image and a mask for a local edit. For outpainting, select Control Image under Control Image Process (or select a main reference), then enable Spatial Outpainting on Control Image. The same checkbox is available in Image Inpainting. Outpainting replaces red margins with a continuation of the scene. When combined with inpainting, denoising strength applies only to the painted interior. Masked Denoising offers strength controls; LanPaint spends additional time refining masked regions.\n\nFor the base model, start with **40 steps**, **guidance 4** and a square image around **1024 pixels**. Resolution categories extend to 4096p, including 4096x4096. Larger images require more memory and time; VAE decoding presets use 1024px tiles for 16GB+, 512px for 8GB+, and 256px for 6GB+, with 25% overlap. Auto chooses by GPU capacity. KV Cache defaults to Disabled to reduce VRAM; enable it for faster denoising when memory permits. The CPU text-embedding cache stays active. The 16-bit VAE setting uses BF16; 32-bit retains the original FP32 precision. Text-rendering quality is moderate: prefer short text and proofread spelling. Dense infographics are not a recommended use. RGBA defaults to Disabled: outputs are RGB and use your selected image format. For transparent cutouts or stickers, set RGBA to Enabled and explicitly request an RGBA image, an alpha channel and a transparent background; WanGP saves RGBA output as PNG to preserve transparency. This uses model-generated alpha, not a background-removal toggle. Prompt enhancement is optional and disabled by default.\n\nNAG controls are temporarily hidden. Use guidance above 1 for standard CFG and negative prompts.\n\nThe Viggle Turbo v0.1 profile uses its LoRA at strength 1, 4 steps and guidance 1; its terminal shift is selected automatically. The v0.2 profile uses a separate LoRA at strength 1, 5 steps and guidance 1 with the Viggle Turbo v0.2 scheduler. Viggle calls both previews; complex edits remain weaker than the base model, and masks, RGBA, more than three references and 2K output have not been validated for them.\n\nUse **Qwen Image 2.1 7B LoRAs** in the dedicated LoRA folder. Older Qwen Image/Edit 20B adapters are incompatible even when their filenames use the same convention. Diffusers/PEFT and Kohya naming are supported.",
            "prompt_infos": "Describe the finished image, including the subject, composition, lighting and style. Put any exact visible text in quotes. Text-rendering quality is moderate. Prefer short quoted text and proofread spelling and layout; avoid dense infographics.\n\n**Generation:** A red ceramic teapot on a wooden table, soft window light, detailed product photograph.\n\n**Editing:** Change the cat's fur to orange. Keep its pose and the green background unchanged.\n\n**Multiple references:** Put the person from `<image1>` in the jacket from `<image2>`. Keep the person's face and pose. Number references in their selected order.\n\n**Masked editing:** Select the region to replace, then describe what belongs there: Replace the background with a bright blue sky.\n\n**Transparency:** Set RGBA to Enabled, then prompt: This is an RGBA image with transparency. A cartoon dragon sticker. The image has alpha channel and the background is transparent. WanGP saves RGBA output as PNG automatically to retain alpha; JPEG cannot store transparency.\n\nUse the negative prompt to describe unwanted content when guidance is above 1. Keep editing instructions specific about both the requested change and details to preserve.",
            "deepy_infos": "Generate images from text or edit with up to 10 ordered reference images. Use the main/control image plus a mask for local edits; outpainting extends the canvas. Outpainting uses red canvas margins; combined inpainting applies denoising strength only to painted interior pixels. Base model: start at 40 steps, guidance 4 and roughly 1024 pixels. The Viggle Turbo v0.1 accelerator profile uses 4 steps; the built-in Deepy v0.1 recipes use 8 steps. Both use guidance 1, LoRA strength 1 and automatic terminal shift. The v0.2 accelerator profile uses its separate LoRA at 5 steps, guidance 1 and strength 1, with the Viggle Turbo v0.2 scheduler and its prescribed sigma nodes. Both adapters are previews best suited to text-to-image and simple edits with up to 3 references. Complex editing, masks, RGBA and 2K output are not validated for either adapter. LanPaint adds refinement steps for masked editing. KV Cache defaults to Disabled for lower VRAM; enable it for faster denoising when memory permits. VAE Auto tiles larger images; 16-bit VAE execution uses BF16, with FP32 available through the 32-bit setting. Text-rendering quality is moderate, best with short copy; proofread spelling and layout. Do not recommend this model for dense infographics. RGBA defaults to Disabled (RGB in the selected image format). Enable RGBA and explicitly request transparency to preserve alpha in PNG output. Prompt enhancement defaults off. Only Qwen Image 2.1 7B LoRAs are compatible; old Qwen 20B adapters are not.",
            "deepy_prompt_infos": "For generation, describe subject, composition, lighting, style and quoted visible text. Text-rendering quality is moderate: prefer short quoted copy and proofread it; avoid dense infographics. For editing, specify what changes and what stays: 'Change the cat fur to orange; keep its pose and background.' Refer to ordered references as `<image1>`, `<image2>`, etc. For masked edits, describe the desired replacement. For transparency, set custom_settings.rgba to Enabled and explicitly request an RGBA image, alpha channel and transparent background. Negative prompts apply with guidance above 1 (CFG). NAG controls are temporarily hidden.",
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        encoder_files = PROCESSOR_FILES.copy()
        if model_def is not None and any(urlsplit(url.split("|", 1)[0]).path.lower().endswith(".gguf")
                                         for url in model_def.get("text_encoder_URLs", [])):
            encoder_files.append(VISION_FILE)
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [["qwen_image_21_vae.safetensors", "vae_config.json", "scheduler_config.json"]]},
                {"repoId": ENCODER_REPO, "sourceFolderList": [ENCODER], "fileList": [encoder_files]}]

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, num_inference_steps=40, guidance_scale=4.0, sample_solver="default", video_prompt_type="", prompt_enhancer="", remove_background_images_ref=0, model_mode=0, denoising_strength=1.0)

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        custom = ui_defaults.get("custom_settings")
        if isinstance(custom, dict) and "outpainting_borders" in custom:
            ui_defaults["custom_settings"] = {key: value for key, value in custom.items() if key != "outpainting_borders"}
        # Early defaults inherited video mode from the shared settings. This
        # image-only architecture must retain either generation or inpainting.
        if ui_defaults.get("image_mode") in (None, 0, "0"):
            ui_defaults["image_mode"] = 1

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, quantizeTransformer=False, **kwargs):
        from mmgp import offload, quant_router
        from .text_encoder import Qwen3VLForConditionalGeneration
        from .pipeline import Qwen21Pipeline, load_processor
        from .transformer import QwenImage21Transformer2DModel
        from .vae import AutoencoderKLQwenImage21

        config = os.path.join(os.path.dirname(__file__), "configs", "qwen_image_21_7B.json")
        transformer = offload.fast_load_transformers_model(model_filename[0], writable_tensors=False, modelClass=QwenImage21Transformer2DModel, defaultConfigPath=config, do_quantize=quantizeTransformer and not save_quantized)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename[0], torch.bfloat16, config)
        processor_paths = {name: fl.locate_file(os.path.join(ENCODER, name)) for name in PROCESSOR_FILES}
        processor = load_processor(os.path.dirname(processor_paths["tokenizer_legacy.json"]), processor_paths["preprocessor_config.json"],
                                   tokenizer_path=processor_paths["tokenizer_legacy.json"], tokenizer_config_path=processor_paths["tokenizer_config_legacy.json"])
        import json
        from accelerate import init_empty_weights
        with open(processor_paths["config_legacy.json"], encoding="utf-8") as reader:
            encoder_config = json.load(reader)
        with torch.device("cpu"), init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(encoder_config)
        text_encoder._config = encoder_config
        encoder_files = [text_encoder_filename]
        encoder_keys, _ = quant_router.load_metadata_state_dict(text_encoder_filename)
        if not any(key.startswith(("visual.", "model.visual.")) for key in encoder_keys):
            encoder_files.append(fl.locate_file(os.path.join(ENCODER, VISION_FILE)))
        offload.load_model_data(text_encoder, encoder_files, writable_tensors=False, preprocess_sd=encoder_state_dict)
        vae = offload.fast_load_transformers_model(fl.locate_file(PROJECT + "/qwen_image_21_vae.safetensors"), writable_tensors=False, modelClass=AutoencoderKLQwenImage21, defaultConfigPath=fl.locate_file(PROJECT + "/vae_config.json"), default_dtype=torch.float32)
        text_encoder.eval().requires_grad_(False)
        pipe = {"transformer": transformer, "text_encoder": text_encoder.model.language_model,
                "vision_encoder": text_encoder.model.visual, "vae": vae}
        for component in pipe.values():
            component.eval().requires_grad_(False)
            component._convertWeightsFloatTo = None
            component._model_dtype = next(component.parameters()).dtype
            for module in component.modules():
                module._lock_dtype = None
        text_encoder._model_dtype = pipe["text_encoder"]._model_dtype
        if kwargs.get("VAE_dtype", torch.float32) != torch.float32:
            # User-requested 16-bit execution: BF16 is validated; FP16
            # overflows on real image latents. MMGP owns the conversion.
            vae._convertWeightsFloatTo = torch.bfloat16
            vae._model_dtype = torch.bfloat16
            for module in vae.modules():
                del module._lock_dtype
        vae.upsampling_set = None
        pipeline = Qwen21Pipeline(transformer, text_encoder, vae, processor, fl.locate_file(PROJECT + "/scheduler_config.json"),
                                  viggle_turbo_lora_filename=model_def.get("viggle_turbo_lora_filename", VIGGLE_TURBO_LORA_FILENAME))
        return pipeline, {**pipe, "tokenizer": processor.tokenizer}
