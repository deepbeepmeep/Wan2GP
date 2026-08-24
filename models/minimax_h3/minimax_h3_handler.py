"""WanGP family handler for MiniMax H3."""

import os

import gradio as gr
import torch

from shared.utils.hf import build_hf_url
from shared.utils.frame_scheduler import normalize_overlap

from .constants import H3_PHASE_2_NOISE_LEVEL_START_DEFAULT
from .minimax_h3_main import AUDIO_VAE_FILE, LATENT_UPSCALER_FILE, LATENT_UPSCALER_FOLDER, TEXT_ENCODER_FOLDER, VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE
from .prompt_enhancer import (FL2VA_IMAGE_SYSTEM_PROMPT, FL2VA_PROMPT_INFOS, FL2VA_TEXT_SYSTEM_PROMPT,
                              REF2VA_IMAGE_SYSTEM_PROMPT, REF2VA_PROMPT_INFOS, REF2VA_TEXT_SYSTEM_PROMPT)


REPO_ID = "DeepBeepMeep/MiniMax-H3"
TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
TEXT_ENCODER_GGUF_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
TEXT_ENCODER_GGUF_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
TEXT_ENCODER_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TURBO_LORA_FILE = "minimax_h3_lightx2v_fl2v_turbo_4step_alpha16_v0.1.safetensors"
TURBO_LORA_KEY = "minimax_h3_lora_turbo"
REF_TURBO_LORA_FILE = "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors"
REF_TURBO_LORA_KEY = "minimax_h3_ref_lora_turbo"
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

FL2VA creates a video with stereo sound from your text prompt. You can optionally provide a start image, an end image, a control video, injected frames, or a soundtrack.

### Start and end images

- **No image:** generate the video and audio from the text prompt.
- **Start image only:** begin the video with that image.
- **End image only:** finish the video with that image.
- **Start and end images:** guide both ends of the video.

Start and end images are placed at those exact points in the video. For general character, object, or style references, use Ref2VA instead.

### Control Video / Frames Injection

- **Generate without using a Control Video:** generate normally from the prompt and any start/end images.
- **Use Control Video:** use an uploaded video to guide the result. Lower **Denoising Strength** values keep the result closer to the control video; `1.0` gives the model full freedom. At `1.0` with **Whole Frame** selected, the control video does not affect the result, so WanGP skips that work. Choose **Masked Area** or **Non Masked Area** to limit editing to part of the frame. **Masking Strength** controls how strongly the rest of the frame stays close to the control video. Use a lower masking strength (<0.75) to facilitate continuity with masked areas.
- **Inject Frames:** add images at specific points in the generated video. Add the images under **Reference Images**, then enter one position per image in the same order. Position `1` means the first frame; `L` means the last frame of a sliding-window segment.

### Audio Source

- **Generate Video and Audio from Text Prompt:** let H3 create both.
- **Generate Video based on Soundtrack and Text Prompt:** upload a soundtrack and H3 will create the video around it. If the soundtrack covers the whole video, WanGP uses the original audio in the final file. If it ends early, H3 generates the remaining sound.
- **Generate Video based on Control Video + its Audio Track and Text Prompt:** select **Use Control Video**. H3 uses both its frames and its existing audio track; the visual changes follow the denoising and mask settings.
- **Generate Audio based on Control Video and Text Prompt:** select **Use Control Video**. WanGP keeps the control video's frames unchanged and asks H3 to create a new soundtrack.

### Longer videos

Sliding windows can continue a video beyond one generation. Choose any overlap amount and WanGP will round it to the nearest H3-compatible value (1, 18, 35, 52...). It automatically reuses the overlapping video and audio to make the join smoother.

H3 is designed for 24 FPS, although WanGP can generate at another frame rate. MiniMax documents an official duration of 4–15 seconds per generation window; longer videos are possible through sliding windows.
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

### Background removal

Background removal is disabled by default. After selecting reference images, use **Automatic Removal of Background behind People or Objects in Reference Images** when you want WanGP to isolate people or objects before sending the images to H3. Keep backgrounds when the surrounding scene is part of the reference you want H3 to follow.

### Start/end images and longer videos

Ref2VA also supports optional start and end images. These are shown to the prompt before the general reference images: with a start image and one reference image, the start image is `<Picture 1>` and the reference image is `<Picture 2>`.

For longer videos, choose any sliding-window overlap amount and WanGP will round it to the nearest H3-compatible value (1, 18, 35, 52...). It automatically carries the overlapping video and audio into the next window while keeping the selected references available.

H3 is designed for 24 FPS, although WanGP can generate at another frame rate. MiniMax documents an official duration of 4–15 seconds per generation window; longer videos are possible through sliding windows.

See the [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) for the upstream specifications and prompting guidance.
"""

H3_RUNTIME_INFOS = """
### How to use one phase, two phases, and tiling

Enable **Advanced Mode**, open **General**, and choose an option under **Phases**:

- **One Phase (default):** generates directly at the selected resolution. Use it for standard resolutions or when whole-frame consistency matters more than high-resolution speed.
- **Two Phases:** use it for faster high-resolution generation and improved fine detail. Most of the work is performed at a lower resolution before H3 enhances the result at the selected output resolution. This mode does not reduce the peak VRAM required by the final enhancement.
- **Two Phases with Tiling:** use it when regular two-phase generation runs out of VRAM. It processes the final enhancement as four overlapping areas, reducing peak VRAM at the cost of extra processing time and a possible risk of visible seams or local inconsistencies.

Start with the default **Phase 2 Noise Level Start**. Lower it to keep the result closer to the first phase and favor smoother tile blending; raise it to encourage stronger new details, with a greater risk of seams or changes between tiles.

WanGP manages the required phase-two Turbo LoRA automatically. Other selected Turbo LoRAs are disabled during phase two to avoid conflicts, while non-Turbo LoRAs retain their selected phase-two multiplier.

### Speed and memory choices

Enable **Advanced Mode** to access these options:

- **Spectrum:** in **Steps Skipping**, select **Spectrum Feature Forecasting**. Spectrum captures an accelerated local-only trajectory, retains its actual-step anchors in system RAM, then performs a transformer-free smoothing replay with independent video and audio prediction. Keep the default 25% start for five full warmup steps in a 20-step generation; increasing it starts later and skips fewer steps. Short Euler schedules can bootstrap after their first actual step, while RES Multistep preserves a three-step actual tail.
- **First Block Cache:** in **Steps Skipping**, select **First Block Cache**. It runs the first transformer block to decide whether the remaining blocks can reuse their previous result. The balanced strength uses the upstream 0.08 threshold; higher strengths skip more work but can change motion or fine details. The displayed strength is not an exact speed multiplier.
- **Ralston 2S:** in **Sampler Solver / Scheduler**, select **Ralston 2S** to use the anchored deterministic second-order Runge-Kutta sampler. It evaluates H3 at the start and two-thirds point of every interval, anchors the second prediction to the interval start, then combines both predictions with Ralston's `1/4, 3/4` weights. This can reduce numerical integration error and may improve fine-detail retention, motion stability, and audio/video coherence. Perceptual improvements are prompt-dependent and are not guaranteed. Its second prediction depends on the first, so they cannot run in parallel: Ralston performs two full transformer predictions per step and sampling is approximately **2x slower** than Euler or RES Multistep at the same step count. Spectrum Feature Forecasting is unsupported with Ralston 2S.
- **Sol-Attn:** in **Advanced Mode > Misc. > Override Attention Mode**, select **sol**. The **Start Tau** slider then appears below the attention selector and shows that End Tau is fixed at `0.8`. H3 defaults to `1.3`; this value is used on the first denoising step and decreases linearly to `0.8` on the final step. Use `1.0` for the Sol-Attn paper starting value, increase it to route more attention blocks through the approximate path for greater speed, or lower it for denser attention and higher fidelity. It uses sparse attention only on large visual sequences and requires BF16, Triton 3.6 or newer, and a CUDA NVIDIA GPU using SM86, SM89, SM90, SM100, SM120, or SM121 (such as RTX 30/40/50-series, H100/H200, B100/B200, or DGX Spark); the dropdown reports whether it is available on the current system.
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
            "guidance_max_phases": 2,
            "visible_phases": 0,
            "lora_multiplier_phases": 2,
            "phase_2_spatial_tiling": True,
            "switch_threshold": {
                "label": "Phase 2 Noise Level Start",
                "type": "number",
                "min": 0.7,
                "max": 1.0,
                "step": 0.0001,
            },
            "inference_steps": True,
            "flow_shift": True,
            "spectrum_cache": True,
            "first_block_cache": True,
            "skip_steps_multiplier_choices": FIRST_BLOCK_CACHE_STRENGTHS,
            "skip_steps_multiplier_label": "First Block Cache Threshold",
            "first_block_cache_thresholds": FIRST_BLOCK_CACHE_THRESHOLDS,
            "sol_attention": True,
            "attention_sparsity": {
                "label": "Start Tau (higher = more sparse/faster; lower = more faithful; End Tau = 0.8)",
                "start": 0.0,
                "end": 4.0,
                "inc": 0.05,
            },
            "sample_solvers": [("Euler", "euler"), ("RES Multistep", "res_multistep"), ("Ralston 2S (~2x slower)", "ralston_2s")],
            "no_negative_prompt": True,
            "returns_audio": True,
            "multimedia_generation": True,
            "image_end_frame_position": True,
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
            TURBO_LORA_KEY: build_hf_url(REPO_ID, "loras", TURBO_LORA_FILE),
            REF_TURBO_LORA_KEY: build_hf_url(REPO_ID, "loras", REF_TURBO_LORA_FILE),
            "qkv_splitting": True,
            "keep_frames_video_guide_not_supported": True,
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
                "sliding_window": True,
                "video_continuation": True,
                "sliding_window_defaults": {"window_min": 124, "window_max": 481, "window_step": 17, "window_default": 362,
                                            "overlap_min": 1, "overlap_max": 120, "overlap_step": 17, "overlap_offset": 1, "overlap_default": 18},
                "frames_selection_maximum": 737,
                "image_prompt_types_allowed": "TSEVL",
                "end_frames_always_enabled": True,
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
                "fit_into_canvas_image_refs": 0,
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
                                            "overlap_min": 1, "overlap_max": 120, "overlap_step": 17, "overlap_offset": 1, "overlap_default": 18},
                "image_prompt_types_allowed": "TSEVL",
                "end_frames_always_enabled": True,
                "audio_guide_window_slicing": True,
                "guide_custom_choices": {
                    "choices": [("Generate without using a Control Video", ""),
                                ("Use Control Video", "GV"),
                                ("Inject Frames", "KFI")],
                    "letters_filter": "GVKFI",
                    "default": "",
                    "label": "Control Video / Frames Injection",
                },
                "video_guide_label": "Control Video",
                "mask_preprocessing": {"selection": ["", "A", "NA"]},
                "custom_frames_injection": True,
                "one_image_ref_only": True,
                "no_background_removal": True,
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "audio_guide_label": "Source Audio / Soundtrack",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "K", "2"],
                    "labels": {
                        "": "Generate Video and Audio from Text Prompt",
                        "A": "Generate Video based on Soundtrack and Text Prompt",
                        "K": "Generate Video based on Control Video + its Audio Track and Text Prompt",
                        "2": "Generate Audio based on Control Video and Text Prompt",
                    },
                    "letters_filter": "AK2",
                    "label": "Audio Source",
                    "show_label": True,
                    "default": "",
                },
                "video_length_not_limited_by_audio": True,
                "output_audio_is_input_audio": True,
            })
        return result

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        overlap, error = normalize_overlap(int(inputs["sliding_window_overlap"] or 0), 17, 1)
        if error:
            return error
        inputs["sliding_window_overlap"] = overlap
        if "~" in (inputs["video_prompt_type"] or ""):
            from .pipeline import H3_PHASE_2_TILE_COUNT, _spatial_tiles

            width, height = map(int, inputs["resolution"].split("x"))
            rows, columns = _spatial_tiles(height), _spatial_tiles(width)
            gr.Info(f"MiniMax H3 phase 2 tiling: {H3_PHASE_2_TILE_COUNT} tiles of {columns[0][1]}x{rows[0][1]} pixels (2x2 grid) for a {width}x{height} output.")
        if base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            video_prompt_type = inputs["video_prompt_type"]
            audio_prompt_type = inputs["audio_prompt_type"]
            if "F" in inputs["video_prompt_type"]:
                position_count = len((inputs["frames_positions"] or "").replace(",", " ").split())
                image_count = len(inputs["image_refs"] or [])
                if position_count != image_count:
                    return f"MiniMax H3 frame injection requires one position per Reference Image (found {position_count} positions and {image_count} images)"
            if "2" in audio_prompt_type:
                if "A" in audio_prompt_type or "K" in audio_prompt_type:
                    return "MiniMax H3 audio generation from Control Video cannot also use a source soundtrack"
                if "G" not in video_prompt_type or "V" not in video_prompt_type or inputs["video_guide"] is None:
                    return "MiniMax H3 audio generation from Control Video requires Use Control Video and a Control Video file"
            if "K" in audio_prompt_type:
                if "G" not in video_prompt_type or "V" not in video_prompt_type or inputs["video_guide"] is None:
                    return "MiniMax H3 Control Video soundtrack mode requires Use Control Video and a Control Video file"
                from shared.utils.audio_video import extract_audio_tracks

                try:
                    if extract_audio_tracks(inputs["video_guide"], query_only=True) == 0:
                        return "MiniMax H3 Control Video has no audio track"
                except Exception as error:
                    return f"Unable to inspect the MiniMax H3 Control Video soundtrack: {error}"
            return None

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
        source_folders.append(LATENT_UPSCALER_FOLDER)
        file_lists.append([LATENT_UPSCALER_FILE])
        return [{
            "repoId": REPO_ID,
            "sourceFolderList": source_folders,
            "fileList": file_lists,
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False,
                   text_encoder_quantization=None, dtype=torch.bfloat16, VAE_dtype=torch.float32,
                   mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None,
                   text_encoder_filename=None, shared_h3_pipeline=None, shared_h3_offloadobj=None,
                   disable_pinning=False, **kwargs):
        from .minimax_h3_main import model_factory

        pipeline = model_factory(model_filename, text_encoder_filename, dtype=dtype, VAE_dtype=VAE_dtype,
                                 reference_mode=base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE),
                                 save_quantized=save_quantized, model_type=model_type,
                                 qkv_splitting=model_def["qkv_splitting"],
                                 video_vae_filename=model_def.get("video_vae_file", VIDEO_VAE_FILE),
                                 audio_vae_filename=model_def.get("audio_vae_file", AUDIO_VAE_FILE), shared_h3_pipeline=shared_h3_pipeline)
        pipe = {"transformer": pipeline.transformer}
        if shared_h3_pipeline is None:
            pipe.update({
                "text_encoder": pipeline.text_encoder.language_model,
                "vision_encoder": pipeline.text_encoder.visual,
                "vae": pipeline.video_decoder,
                "video_encoder": pipeline.video_encoder,
                "audio_vae": pipeline.audio_vae,
                "latent_upscaler": pipeline.latent_upscaler,
            })
        else:
            class BorrowingPipe(dict):
                pass

            borrowed_names = ("text_encoder", "vision_encoder", "vae", "video_encoder", "audio_vae", "latent_upscaler")
            pipe = BorrowingPipe(pipe)
            pipe.update({name: shared_h3_offloadobj.models[name] for name in borrowed_names})
            pipe._mmgp_ignore_models = borrowed_names
        return pipeline, {"pipe": pipe, "pinnedMemory": False} if disable_pinning else pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if settings_version < 2.75:
            ui_defaults["switch_threshold"] = H3_PHASE_2_NOISE_LEVEL_START_DEFAULT
        if settings_version < 2.74:
            ui_defaults["attention_sparsity"] = 1.3
        if settings_version < 2.73 and "sliding_window_overlap" in ui_defaults:
            overlap = max(1, int(ui_defaults["sliding_window_overlap"] or 18))
            ui_defaults["sliding_window_overlap"] = normalize_overlap(overlap, 17, 1)[0]
        if settings_version < 2.73 and base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            ui_defaults["sliding_window_size"] = 362
        if settings_version < 2.71:
            ui_defaults["denoising_strength"] = 1.0
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
            "sliding_window_size": 362,
            "sliding_window_overlap": 18,
            "num_inference_steps": 20,
            "guidance_phases": 1,
            "switch_threshold": H3_PHASE_2_NOISE_LEVEL_START_DEFAULT,
            "guidance_scale": 1.0,
            "flow_shift": 12.0,
            "sample_solver": "euler",
            "attention_sparsity": 1.3,
            "skip_steps_start_step_perc": 25,
            "skip_steps_multiplier": 0.08,
            "denoising_strength": 1.0,
            "audio_prompt_type": "",
            "video_prompt_type": "",
            "image_mode": 0,
        })
        if reference_mode:
            ui_defaults.update({"image_refs_relative_size": 100, "remove_background_images_ref": 0})
