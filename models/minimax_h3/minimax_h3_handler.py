"""WanGP family handler for MiniMax H3."""

import os

import torch

from shared.utils.hf import build_hf_url

from .minimax_h3_main import AUDIO_VAE_FILE, TEXT_ENCODER_FOLDER, VIDEO_VAE_FILE
from .prompt_enhancer import (FL2VA_IMAGE_SYSTEM_PROMPT, FL2VA_PROMPT_INFOS, FL2VA_TEXT_SYSTEM_PROMPT,
                              REF2VA_IMAGE_SYSTEM_PROMPT, REF2VA_PROMPT_INFOS, REF2VA_TEXT_SYSTEM_PROMPT)


REPO_ID = "DeepBeepMeep/MiniMax-H3"
TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
FL2VA_ARCHITECTURE = "minimax_h3_fl2va"
FL2VA_PRUNED_ARCHITECTURE = "minimax_h3_fl2va_pruned"
REF2VA_ARCHITECTURE = "minimax_h3_ref2va"
REF2VA_PRUNED_ARCHITECTURE = "minimax_h3_ref2va_pruned"

FL2VA_INFOS = """## FL2VA — First/Last Frame to Video and Audio

FL2VA generates a new video with native 32 kHz stereo audio from a text prompt and optional boundary frames.

- **No frame:** text-to-video-and-audio generation.
- **First frame only:** use the image as the opening frame.
- **Last frame only:** use the image as the ending frame.
- **First and last frames:** constrain both ends of the generated clip.

Each boundary accepts one image. These images are positions on the output timeline, not general character, identity, or style references; use Ref2VA for those.

H3 outputs video at 24 FPS. MiniMax documents an official output range of 4–15 seconds. WanGP can generate longer videos with sliding windows, but each individual window longer than 15 seconds is outside that documented range.
"""

REF2VA_INFOS = """## Ref2VA — Reference to Video and Audio

Ref2VA generates a new video with native 32 kHz stereo audio from text plus multimodal references. Images can guide identity, appearance, or scene content; video can guide content, appearance, and motion; audio can guide or reuse sound and voice. A reference video is contextual material, not a guaranteed frame-exact continuation constraint.

### Reference limits

- **Images:** up to 9.
- **Videos in WanGP:** up to 2 clips; each clip must be 2–15 seconds, with at most 15 seconds of reference video in total. The H3 model itself documents support for 3 video clips.
- **Audio in WanGP:** up to 2 inputs; each clip must be 2–15 seconds, with at most 15 seconds of reference audio in total. The H3 model itself documents support for 3 audio clips.
- **Audio requires matching visual references:** the combined number of reference images and videos must be at least the number of reference audio clips.
- **Video soundtracks:** selecting reference-video soundtracks uses one audio-reference slot per selected video. A soundtrack shares its video's uploaded file, so it does not add another file to the mixed-input count.
- **Mixed references:** at most 12 files across images, videos, and audio.

H3 outputs video at 24 FPS. MiniMax documents an official output range of 4–15 seconds. Ref2VA uses one model pass per generation; sliding-window continuation is not available for this variant.

See the [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) for the upstream specifications and prompting guidance.
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
    def query_model_def(base_model_type, model_def):
        reference_mode = base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        pruned = base_model_type in (FL2VA_PRUNED_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
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
            "sample_solvers": [("Euler", "euler")],
            "no_negative_prompt": True,
            "skip_prompt_template": True,
            "returns_audio": True,
            "multimedia_generation": True,
            "control_video_trim_disabled": True,
            "infos": (REF2VA_INFOS if reference_mode else FL2VA_INFOS) + (PRUNED_INFOS if pruned else ""),
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
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16),
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8),
            ],
        }
        if reference_mode:
            result.update({
                "sliding_window": False,
                "image_prompt_types_allowed": "T",
                "image_ref_choices": {
                    "choices": [("Generate without Reference Images", ""), ("Use Reference Images", "I"),
                                # ("Set Output Dimensions from a Background Image, then Use Reference Images", "KI")
                                ],
                    "letters_filter": "KI",
                    "default": "",
                    "label": "Reference Images",
                },
                "reference_image_enabled": True,
                "return_image_refs_tensor": False,
                "no_processing_on_last_images_refs": 12,
                "no_background_removal": True,
                "guide_custom_choices": {
                    "choices": [("Generate without a Reference Video", ""), ("Use One Reference Video", "VG"),
                                ("Use Two Reference Videos", "V+G")],
                    "letters_filter": "V+G",
                    "default": "",
                    "label": "Reference Videos",
                },
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "video_guide_label": "Reference Video 1",
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
            if not 2 <= duration <= 15:
                return f"Reference Video {index} must be between 2 and 15 seconds long (found {duration:.2f}s)"
            video_durations.append(duration)
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
        if file_count == 0:
            return "MiniMax H3 Ref2VA requires at least one reference image, video, or audio"
        return None

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        source_folders = []
        file_lists = []
        vae_files = []
        if "video_vae_file" not in model_def:
            vae_files.append(VIDEO_VAE_FILE)
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
                                 compressed_modulation=base_model_type in (FL2VA_PRUNED_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE),
                                 reference_mode=base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE),
                                 save_quantized=save_quantized, model_type=model_type,
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
            "audio_prompt_type": "",
            "video_prompt_type": "",
            "image_mode": 0,
        })
