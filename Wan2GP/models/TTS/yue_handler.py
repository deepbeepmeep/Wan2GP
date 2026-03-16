import os


YUE_STAGE1_COT_REPO = "m-a-p/YuE-s1-7B-anneal-en-cot"
YUE_STAGE1_ICL_REPO = "m-a-p/YuE-s1-7B-anneal-en-icl"
YUE_STAGE2_REPO = "m-a-p/YuE-s2-1B-general"
YUE_STAGE1_FILES = [
    "config.json",
]
YUE_STAGE2_FILES = [
    "config.json",
]


def _get_yue_model_def(model_def):
    use_audio_prompt = bool(model_def.get("yue_audio_prompt", False))
    yue_def = {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": True,
        "image_prompt_types_allowed": "",
        "profiles_dir": ["yue"],
        "alt_prompt": {
            "label": "Genres / Tags",
            "placeholder": "pop, dreamy, warm vocal, female, nostalgic",
            "lines": 2,
        },
        "yue_max_new_tokens": 3000,
        "yue_run_n_segments": 2,
        "yue_stage2_batch_size": 4,
        "yue_segment_duration": 6,
        "yue_prompt_start_time": 0.0,
        "yue_prompt_end_time": 30.0,
    }
    if use_audio_prompt:
        yue_def.update(
            {
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "audio_guide_label": "Vocal prompt",
                "audio_guide2_label": "Instrumental prompt",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "AB"],
                    "labels": {
                        "": "Lyrics only",
                        "A": "Mixed audio prompt",
                        "AB": "Vocal + Instrumental prompts",
                    },
                    "letters_filter": "AB",
                    "default": "",
                },
            }
        )
    return yue_def


def _get_yue_download_def(model_def):
    use_audio_prompt = bool(model_def.get("yue_audio_prompt", False))
    stage1_repo = YUE_STAGE1_ICL_REPO if use_audio_prompt else YUE_STAGE1_COT_REPO
    stage1_folder = os.path.basename(stage1_repo)
    stage2_folder = os.path.basename(YUE_STAGE2_REPO)
    xcodec_root = "xcodec_mini_infer"
    xcodec_source_folders = [
        "final_ckpt",
        "decoders",
        "models",
        "modules",
        "quantization",
        "RepCodec",
        "descriptaudiocodec",
        "vocos",
        "semantic_ckpts/hf_1_325000",
    ]
    xcodec_files = [
        ["config.yaml", "ckpt_00360000.pth"],
        ["config.yaml", "decoder_131000.pth", "decoder_151000.pth"],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    return [
        {
            "repoId": stage1_repo,
            "sourceFolderList": [""],
            "targetFolderList": [stage1_folder],
            "fileList": [YUE_STAGE1_FILES],
        },
        {
            "repoId": YUE_STAGE2_REPO,
            "sourceFolderList": [""],
            "targetFolderList": [stage2_folder],
            "fileList": [YUE_STAGE2_FILES],
        },
        {
            "repoId": stage1_repo,
            "sourceFolderList": [""],
            "targetFolderList": ["mm_tokenizer_v0.2_hf"],
            "fileList": [["tokenizer.model"]],
        },
        {
            "repoId": "m-a-p/xcodec_mini_infer",
            "sourceFolderList": [""],
            "targetFolderList": [xcodec_root],
            "fileList": [["vocoder.py", "post_process_audio.py"]],
        },
        {
            "repoId": "m-a-p/xcodec_mini_infer",
            "sourceFolderList": xcodec_source_folders,
            "targetFolderList": [xcodec_root] * len(xcodec_source_folders),
            "fileList": xcodec_files,
        },
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["yue"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (200, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-tts",
            type=str,
            default=None,
            help=f"Path to a directory that contains TTS settings (default: {os.path.join(lora_root, 'tts')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_tts", None) or os.path.join(lora_root, "tts")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_yue_model_def(model_def)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_yue_download_def(model_def or {})

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None,
        VAE_dtype=None,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        profile=0,
        **kwargs,
    ):
        from .yue.pipeline import YuePipeline

        if isinstance(model_filename, list):
            stage1_weights = model_filename[0] if len(model_filename) > 0 else ""
            stage2_weights = model_filename[1] if len(model_filename) > 1 else ""
        else:
            stage1_weights = model_filename or ""
            stage2_weights = ""

        pipeline = YuePipeline(
            stage1_weights_path=stage1_weights,
            stage2_weights_path=stage2_weights,
            use_audio_prompt=bool(model_def.get("yue_audio_prompt", False)),
            max_new_tokens=model_def.get("yue_max_new_tokens", 200),
            run_n_segments=model_def.get("yue_run_n_segments", 1),
            stage2_batch_size=model_def.get("yue_stage2_batch_size", 10),
            segment_duration=model_def.get("yue_segment_duration", 6),
            prompt_start_time=model_def.get("yue_prompt_start_time", 0.0),
            prompt_end_time=model_def.get("yue_prompt_end_time", 30.0),
        )

        pipe = {
            "transformer": pipeline.model_stage1,
            "transformer2": pipeline.model_stage2,
            "codec_model": pipeline.codec_model,
            "vocoder_vocal": pipeline.vocoder_vocal,
            "vocoder_inst": pipeline.vocoder_inst,
        }
        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""

        defaults = {
            "audio_prompt_type": "",
        }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "alt_prompt": "pop, dreamy, warm vocal, female, nostalgic",
                "repeat_generation": 1,
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "temperature": 1.0,
                "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Lyrics prompt cannot be empty for Yue."
        alt_prompt = inputs.get("alt_prompt", "")
        if alt_prompt is None or len(str(alt_prompt).strip()) == 0:
            return "Genres prompt cannot be empty for Yue."
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        if model_def.get("yue_audio_prompt", False):
            if "A" in audio_prompt_type:
                if inputs.get("audio_guide") is None:
                    return "You must provide a vocal or mixed audio prompt for Yue ICL."
                if "B" in audio_prompt_type and inputs.get("audio_guide2") is None:
                    return "You must provide an instrumental prompt for Yue ICL."
                start_time = float(
                    inputs.get(
                        "yue_prompt_start_time",
                        model_def.get("yue_prompt_start_time", 0.0),
                    )
                )
                end_time = float(
                    inputs.get(
                        "yue_prompt_end_time",
                        model_def.get("yue_prompt_end_time", 30.0),
                    )
                )
                if start_time >= end_time:
                    return "Audio prompt start time must be less than end time."
                if end_time - start_time > 30:
                    return "Audio prompt duration should not exceed 30 seconds."
            elif inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
                return "Select an audio prompt type for Yue ICL or clear audio prompts."
        else:
            if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
                return "Yue base model does not support audio prompts. Please use Yue ICL."
        return None
