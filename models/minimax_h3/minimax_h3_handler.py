"""WanGP family handler for MiniMax H3."""

import os

import torch

from shared.utils.hf import build_hf_url

from .minimax_h3_main import AUDIO_VAE_FILE, TEXT_ENCODER_FOLDER, VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE
from .prompt_enhancer import (FL2VA_IMAGE_SYSTEM_PROMPT, FL2VA_PROMPT_INFOS, FL2VA_TEXT_SYSTEM_PROMPT,
                              REF2VA_IMAGE_SYSTEM_PROMPT, REF2VA_PROMPT_INFOS, REF2VA_TEXT_SYSTEM_PROMPT)


REPO_ID = "DeepBeepMeep/MiniMax-H3"
TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
TEXT_ENCODER_GGUF_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
TEXT_ENCODER_GGUF_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
TEXT_ENCODER_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TEXT_ENCODER_VARIANTS = {
    "gguf_q2_k": [TEXT_ENCODER_GGUF_Q2],
    "gguf_q4_k_m": [TEXT_ENCODER_GGUF_Q4],
    "nvfp4_awq": [TEXT_ENCODER_NVFP4],
}
FL2VA_ARCHITECTURE = "minimax_h3_fl2va"
FL2VA_PRUNED_ARCHITECTURE = "minimax_h3_fl2va_pruned"
REF2VA_ARCHITECTURE = "minimax_h3_ref2va"
REF2VA_PRUNED_ARCHITECTURE = "minimax_h3_ref2va_pruned"
FIRST_BLOCK_CACHE_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.14)
LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS = {1.5: 0.06, 1.75: 0.08, 2.0: 0.10, 2.25: 0.12, 2.5: 0.14}
FIRST_BLOCK_CACHE_STRENGTHS = [
    ("Low (0.06)", 0.06),
    ("Balanced (0.08, upstream default)", 0.08),
    ("High (0.10)", 0.10),
    ("Very High (0.12)", 0.12),
    ("Maximum (0.14)", 0.14),
]

FL2VA_INFOS = """## FL2VA — First/Last Frame to Video and Audio

FL2VA generates a new video with native 32 kHz stereo audio from a text prompt and optional boundary frames.

- **No frame:** text-to-video-and-audio generation.
- **First frame only:** use the image as the opening frame.
- **Last frame only:** use the image as the ending frame.
- **First and last frames:** constrain both ends of the generated clip.

Each boundary accepts one image. These images are positions on the output timeline, not general character, identity, or style references; use Ref2VA for those.

H3's native frame rate is 24 FPS. WanGP can use another output frame rate by adjusting H3's temporal positions. MiniMax documents an official output range of 4–15 seconds. WanGP can generate longer videos with sliding windows, but each individual window longer than 15 seconds is outside that documented range.
"""

REF2VA_INFOS = """## Ref2VA — Reference to Video and Audio

Ref2VA generates a new video with native 32 kHz stereo audio from text plus multimodal references. Images can guide identity, appearance, or scene content; video can guide content, appearance, and motion; audio can guide or reuse sound and voice. A reference video is contextual material, not a guaranteed frame-exact continuation constraint.

### Reference limits

- **Images:** up to 9.
- **Videos in WanGP:** up to 2 clips; each source must be at least 2 seconds, inputs longer than 15 seconds are truncated, and the prepared clips may total at most 15 seconds. The H3 model itself documents support for 3 video clips.
- **Audio in WanGP:** up to 2 inputs; each clip must be 2–15 seconds, with at most 15 seconds of reference audio in total. The H3 model itself documents support for 3 audio clips.
- **Audio requires matching visual references:** the combined number of reference images and videos must be at least the number of reference audio clips.
- **Video soundtracks:** selecting reference-video soundtracks uses one audio-reference slot per selected video. A soundtrack shares its video's uploaded file, so it does not add another file to the mixed-input count.
- **Mixed references:** at most 12 files across images, videos, and audio.

### Choosing a video input

- **Reference Video:** reuse subjects, appearance, or motion without changing the selected output resolution.
- **Depth Control:** reuse the scene's depth and layout.
- **Generic Control:** provide the video unchanged, use its aspect ratio to set the output dimensions and use the text prompt to tell the model what to do with it.

Reference videos adapt to your chosen output size. Control videos instead define the output size and are converted into the selected guide. Both guide the result creatively rather than reproducing every frame exactly.

### Reference-image size

The reference-image selector can preserve the selected output dimensions or use the first reference image to define them. In **Advanced Mode**, **Rescale Internaly Image Ref (% in relation to Output Video) to change Output Composition** controls each reference image's internal pixel budget while preserving its aspect ratio. **100%** matches the output video's pixel budget. Higher values can preserve more reference detail and improve fidelity, but generation is slower; lower values are faster but can lose fine details. This setting does not change the output resolution after it has been selected or derived.

H3's native frame rate is 24 FPS. WanGP can use another output frame rate by resampling reference videos and adjusting H3's temporal positions. MiniMax documents an official output range of 4–15 seconds. Ref2VA uses one model pass per generation; sliding-window continuation is not available for this variant.

See the [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) for the upstream specifications and prompting guidance.
"""

H3_RUNTIME_INFOS = """
### Speed and memory choices

Enable **Advanced Mode** to access these options:

- **Spectrum:** in **Steps Skipping**, select **Spectrum Feature Forecasting**. Spectrum accelerates generation by forecasting selected transformer steps, at the cost of possible changes to motion or fine details. Keep the default 25% start for five full warmup steps in a 20-step generation; increasing it starts later and skips fewer steps.
- **First Block Cache:** in **Steps Skipping**, select **First Block Cache**. It runs the first transformer block to decide whether the remaining blocks can reuse their previous result. The balanced strength uses the upstream 0.08 threshold; higher strengths skip more work but can change motion or fine details. The displayed strength is not an exact speed multiplier.
- **Sol-Attn:** in **Advanced Mode > Misc. > Override Attention Mode**, select **sol**. It uses sparse attention on large visual sequences to reduce attention cost and speed up generation, with possible small quality differences. It requires BF16, Triton 3.6 or newer, and a CUDA NVIDIA GPU using SM89, SM90, SM100, or SM120 (such as RTX 40/50-series, H100/H200, or B100/B200); the dropdown reports whether it is available on the current system.
- **Text Encoder:** at the bottom of **Misc.**, use the **Text Encoder** configuration to reduce system RAM. **Qwen3-VL BF16** uses the most memory; **Quanto INT8** is a balanced lower-memory choice; **NVFP4 AWQ**, **GGUF Q4_K_M**, and especially **GGUF Q2_K** reduce it further. More aggressive quantization can slightly affect prompt interpretation.
- **Priority:** beside the Text Encoder configuration, choose which memory limit matters most. **Lower VRAM** uses all code optimizations and reduces greatly VRAM consumption while **Lower RAM** uses only VRAM optimizations that doesnt consume extra RAM.
"""

PRUNED_INFOS = """
### Pruned 20B checkpoint

The Pruned checkpoint replaces the full AdaLN timestep projection matrices with precomputed low-rank modulation curves. It accepts the same inputs and settings as its 33B counterpart while reducing checkpoint size and weight-transfer cost.
"""


class family_handler:
    @staticmethod
    def query_supported_types():
        return [FL2VA_ARCHITECTURE, FL2VA_PRUNED_ARCHITECTURE,
                REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {
            FL2VA_PRUNED_ARCHITECTURE: FL2VA_ARCHITECTURE,
            REF2VA_ARCHITECTURE: FL2VA_ARCHITECTURE,
            REF2VA_PRUNED_ARCHITECTURE: FL2VA_ARCHITECTURE,
        }, {}

    @staticmethod
    def query_model_family():
        return "minimax_h3"

    @staticmethod
    def query_family_infos():
        return {"minimax_h3": (70, "MiniMax H3")}

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("minimax_h3")

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument("--lora-dir-minimax-h3", type=str, default=None,
                            help=f"Path to MiniMax H3 LoRAs (default: {os.path.join(lora_root, 'minimax_h3')})")

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_minimax_h3", None) or os.path.join(lora_root, "minimax_h3")

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        if cache_type == "first_block":
            skip_steps_cache.threshold = float(skip_steps_cache.multiplier)
        elif cache_type != "spectrum":
            raise ValueError(f"MiniMax H3 does not support step-skipping type {cache_type!r}")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        reference_mode = base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        pruned = base_model_type in (FL2VA_PRUNED_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        text_encoder_variant = model_def.get("text_encoder_variant")
        text_encoder_files = [TEXT_ENCODER_BF16, TEXT_ENCODER_INT8] if text_encoder_variant is None else TEXT_ENCODER_VARIANTS[text_encoder_variant]
        result = {
            "dtype": "bf16",
            "fps": 24,
            "frames_minimum": 107,
            "frames_steps": 17,
            "frames_offset": 5,
            "block_size": 32,
            "vae_block_size": 32,
            "guidance_max_phases": 0,
            "lora_multiplier_phases": 1,
            "inference_steps": True,
            "flow_shift": True,
            "spectrum_cache": True,
            "first_block_cache": True,
            "skip_steps_multiplier_choices": FIRST_BLOCK_CACHE_STRENGTHS,
            "skip_steps_multiplier_label": "First Block Cache Threshold",
            "first_block_cache_thresholds": FIRST_BLOCK_CACHE_THRESHOLDS,
            "sol_attention": True,
            "sample_solvers": [("Euler", "euler")],
            "no_negative_prompt": True,
            "skip_prompt_template": True,
            "returns_audio": True,
            "multimedia_generation": True,
            "control_video_trim_disabled": True,
            "infos": (REF2VA_INFOS if reference_mode else FL2VA_INFOS) + H3_RUNTIME_INFOS + (PRUNED_INFOS if pruned else ""),
            "prompt_infos": REF2VA_PROMPT_INFOS if reference_mode else FL2VA_PROMPT_INFOS,
            "prompt_enhancer_button_label": "Write H3 Prompt",
            "prompt_enhancer_def": {
                "selection": ["T", "TI"],
                "labels": {
                    "TV": "Write an H3 Reference Prompt from Text" if reference_mode else "Write an H3 Prompt from Text",
                    "TIV": "Write an H3 Reference Prompt from Text + First Reference Image" if reference_mode else "Write an H3 Prompt from Text + Start Image",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": REF2VA_TEXT_SYSTEM_PROMPT if reference_mode else FL2VA_TEXT_SYSTEM_PROMPT,
            "video_prompt_enhancer_instructions": REF2VA_IMAGE_SYSTEM_PROMPT if reference_mode else FL2VA_IMAGE_SYSTEM_PROMPT,
            "text_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
            "video_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
            "profiles_dir": ["minimax_h3"],
            "finetune_custom_urls": ["video_vae_file", "audio_vae_file"],
            "qkv_splitting": True,
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, filename) for filename in text_encoder_files],
            "system_configs": {
                "_name": "Text Encoder",
                "bf16": {"name": "Qwen3-VL BF16", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16)]},
                "int8": {"name": "Qwen3-VL Quanto INT8", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8)]},
                "nvfp4_awq": {"name": "Qwen3-VL NVFP4 AWQ", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_NVFP4)]},
                "gguf_q4_k_m": {"name": "Qwen3-VL GGUF Q4_K_M", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q4)]},
                "gguf_q2_k": {"name": "Qwen3-VL GGUF Q2_K", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q2)]},
            },
            "system_configs2": {
                "_name": "Video VAE",
                "_default_label": "Original VAE",
                "fp8mix": {"name": "FP8 Mixed Precision", "video_vae_file": VIDEO_VAE_FP8MIX_FILE},
            },
            "system_configs3": {
                "_name": "DiT Denoising Priority",
                "_default_label": "Lower VRAM",
                "lower_ram": {"name": "Lower RAM", "qkv_splitting": False},
            },
        }
        if reference_mode:
            result.update({
                "sliding_window": False,
                "frames_maximum": 737,
                "image_prompt_types_allowed": "T",
                "image_ref_choices": {
                    "choices": [("Generate without Reference Images", ""),
                                ("Use Reference Images", "I"),
                                ("First Reference Image is the Main Subject / Landscape, defines Output Dimensions, and may be followed by other Reference Images", "KI")],
                    "letters_filter": "KI",
                    "default": "",
                    "label": "Reference Images",
                },
                "reference_image_enabled": True,
                "return_image_refs_tensor": False,
                "no_processing_on_last_images_refs": 12,
                "no_background_removal": True,
                "any_image_refs_relative_size": True,
                "image_refs_relative_size": {"min": 50, "max": 400, "step": 1},
                "guide_custom_choices": {
                    "choices": [("Generate without a Reference or Control Video", ""), ("Use One Reference Video", "V-"),
                                ("Use Two Reference Videos", "V+-"),
                                # ("Transfer Human Pose From Control Video", "PV"),
                                ("Transfer Depth Map From Control Video", "DV"),
                                # ("Transfer Edges Map From Control Video", "EV"),
                                ("Provide Generic Control Video", "V")],
                    "letters_filter": "PDEV+-",
                    "default": "",
                    "label": "Reference / Control Video",
                },
                "preprocess_video_guide2": True,
                "reference_video_max_frames": 15 * 24,
                "reference_video_max_size": (768, 1344),
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "video_guide_label": "Reference / Control Video 1",
                "video_guide2_label": "Reference Video 2",
                "audio_guide_label": "Audio Reference 1",
                "audio_guide2_label": "Audio Reference 2",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "AB", "K"],
                    "labels": {
                        "": "Generate without an Audio Reference",
                        "A": "Use One Audio Reference",
                        "AB": "Use Two Audio References",
                        "K": "Use Reference-Video Soundtrack(s)",
                    },
                    "letters_filter": "ABK",
                    "label": "Audio References",
                    "show_label": True,
                    "default": "",
                },
                "audio_guide_window_slicing": True,
                "video_length_not_limited_by_audio": True,
            })
        else:
            result.update({
                "sliding_window": True,
                "video_continuation": True,
                "sliding_window_defaults": {"window_min": 124, "window_max": 481, "window_step": 17, "window_default": 362,
                                            "overlap_min": 1, "overlap_max": 1, "overlap_step": 0, "overlap_default": 1},
                "image_prompt_types_allowed": "TSEVL",
                "end_frames_always_enabled": True,
            })
        return result

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            return None

        inputs["sliding_window_size"] = inputs["video_length"]
        inputs["sliding_window_overlap"] = 0
        inputs["sliding_window_discard_last_frames"] = 0
        inputs["sliding_window_trim_first_frames"] = 0
        inputs["sliding_window_overlap_noise"] = 0
        inputs["sliding_window_color_correction_strength"] = 0
        inputs["sub_parallel_window_size"] = 0
        inputs["sub_parallel_window_overlap"] = 0

        video_prompt_type = inputs["video_prompt_type"]
        audio_prompt_type = inputs["audio_prompt_type"]
        image_count = len(inputs["image_refs"] or [])
        videos = [inputs["video_guide"]] if "V" in video_prompt_type else []
        if "+" in video_prompt_type:
            videos.append(inputs["video_guide2"])
        audios = [inputs["audio_guide"]] if "A" in audio_prompt_type else []
        if "B" in audio_prompt_type:
            audios.append(inputs["audio_guide2"])

        if image_count > 9:
            return "MiniMax H3 Ref2VA accepts at most 9 reference images"
        if len(videos) > 2:
            return "WanGP accepts at most 2 MiniMax H3 reference videos"

        from shared.utils.utils import get_video_info

        video_durations = []
        for index, video in enumerate(videos, 1):
            try:
                fps, _, _, frames = get_video_info(video)
                duration = frames / fps
            except Exception as error:
                return f"Unable to read Reference Video {index}: {error}"
            if duration < 2:
                return f"Reference Video {index} must be at least 2 seconds long (found {duration:.2f}s)"
            video_durations.append(min(duration, 15))
        if sum(video_durations) > 15:
            return f"Reference videos must total at most 15 seconds (found {sum(video_durations):.2f}s)"

        soundtrack_mode = "K" in audio_prompt_type
        if soundtrack_mode:
            if not videos:
                return "Using reference-video soundtracks requires at least one Reference Video"
            from shared.utils.audio_video import extract_audio_tracks

            for index, video in enumerate(videos, 1):
                try:
                    if extract_audio_tracks(video, query_only=True) == 0:
                        return f"Reference Video {index} has no audio track"
                except Exception as error:
                    return f"Unable to inspect the soundtrack of Reference Video {index}: {error}"
            audio_durations = video_durations
            audio_count = len(videos)
        else:
            import librosa

            audio_durations = []
            for index, audio in enumerate(audios, 1):
                try:
                    duration = float(librosa.get_duration(path=os.fspath(audio)))
                except Exception as error:
                    return f"Unable to read Audio Reference {index}: {error}"
                if not 2 <= duration <= 15:
                    return f"Audio Reference {index} must be between 2 and 15 seconds long (found {duration:.2f}s)"
                audio_durations.append(duration)
            audio_count = len(audios)

        if audio_count > 2:
            return "WanGP accepts at most 2 MiniMax H3 audio references"
        if sum(audio_durations) > 15:
            return f"Audio references must total at most 15 seconds (found {sum(audio_durations):.2f}s)"
        visual_count = image_count + len(videos)
        if audio_count > visual_count:
            return f"MiniMax H3 requires at least as many reference images and videos as audio references (found {visual_count} visual and {audio_count} audio)"
        file_count = image_count + len(videos) + (0 if soundtrack_mode else audio_count)
        if file_count > 12:
            return f"MiniMax H3 accepts at most 12 reference files (found {file_count})"
        return None

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        source_folders = []
        file_lists = []
        vae_files = []
        video_vae_file = model_def.get("video_vae_file", VIDEO_VAE_FILE)
        if video_vae_file in (VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE):
            vae_files.append(video_vae_file)
        if "audio_vae_file" not in model_def:
            vae_files.append(AUDIO_VAE_FILE)
        if vae_files:
            source_folders.append("")
            file_lists.append(vae_files)
        source_folders.append(TEXT_ENCODER_FOLDER)
        file_lists.append(["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json"])
        return [{
            "repoId": REPO_ID,
            "sourceFolderList": source_folders,
            "fileList": file_lists,
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False,
                   text_encoder_quantization=None, dtype=torch.bfloat16, VAE_dtype=torch.float32,
                   mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None,
                   text_encoder_filename=None, **kwargs):
        from .minimax_h3_main import model_factory

        pipeline = model_factory(model_filename, text_encoder_filename, dtype=dtype, VAE_dtype=VAE_dtype,
                                 reference_mode=base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE),
                                 save_quantized=save_quantized, model_type=model_type,
                                 qkv_splitting=model_def["qkv_splitting"],
                                 video_vae_filename=model_def.get("video_vae_file", VIDEO_VAE_FILE),
                                 audio_vae_filename=model_def.get("audio_vae_file", AUDIO_VAE_FILE))
        return pipeline, {
            "transformer": pipeline.transformer,
            "text_encoder": pipeline.text_encoder.language_model,
            "vision_encoder": pipeline.text_encoder.visual,
            "vae": pipeline.video_decoder,
            "video_encoder": pipeline.video_encoder,
            "audio_vae": pipeline.audio_vae,
        }

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if settings_version < 2.70:
            cache_value = float(ui_defaults.get("skip_steps_multiplier", 0.08))
            ui_defaults["skip_steps_multiplier"] = LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS.get(cache_value, cache_value)
        if settings_version < 2.69:
            encoder, priority, _, finetune = (str(ui_defaults.get("config", "")).split(",") + [""] * 4)[:4]
            ui_defaults["config"] = ",".join((encoder, "", priority, finetune)).rstrip(",")
        if base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            return
        if settings_version < 2.67:
            ui_defaults["image_refs_relative_size"] = 100
            ui_defaults["video_prompt_type"] = ui_defaults.get("video_prompt_type", "").replace("G", "")
        if settings_version < 2.68 and "V" in ui_defaults.get("video_prompt_type", "") and "-" not in ui_defaults["video_prompt_type"]:
            ui_defaults["video_prompt_type"] += "-"

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        reference_mode = base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        ui_defaults.update({
            "video_length": 124,
            "sliding_window_size": 124 if reference_mode else 362,
            "sliding_window_overlap": 0 if reference_mode else 1,
            "num_inference_steps": 20,
            "guidance_scale": 1.0,
            "flow_shift": 12.0,
            "sample_solver": "euler",
            "skip_steps_start_step_perc": 25,
            "skip_steps_multiplier": 0.08,
            "audio_prompt_type": "",
            "video_prompt_type": "",
            "image_mode": 0,
        })
        if reference_mode:
            ui_defaults["image_refs_relative_size"] = 100
