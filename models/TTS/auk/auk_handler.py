import os

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS
from .prompt_enhancer import SYSTEM_PROMPT


REPO_ID = 'DeepBeepMeep/TTS'
MAX_DURATION = 300  # Qwen2.5-Omni's bundled audio feature extractor chunk_length.
ENCODER_FOLDER = 'Qwen2.5-Omni-3B'
ENCODER_FILES = ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'merges.txt', 'vocab.json', 'chat_template.json', 'LICENSE', 'Notice']


class family_handler:
    @staticmethod
    def query_supported_types():
        return ['auk']

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return 'tts'

    @staticmethod
    def query_family_infos():
        return {'tts': (2200, 'TTS')}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument('--lora-dir-auk', type=str, default=None, help='Path to AuK LoRAs')

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, 'lora_dir_auk', None) or os.path.join(lora_root, 'auk')

    @staticmethod
    def query_model_def(base_model_type, model_def):
        flash = model_def.get('auk_flash', False)
        return {
            'group': 'tts', 'audio_only': True, 'image_outputs': False, 'sliding_window': False,
            'image_prompt_types_allowed': '', 'no_negative_prompt': True, 'guidance_max_phases': 0 if flash else 1, 'embedded_guidance': False,
            'inference_steps': not flash, 'temperature': False, 'supports_early_stop': True,
            'profiles_dir': ['auk'], 'compile': False,
            'text_prompt_enhancer_instructions': SYSTEM_PROMPT, 'text_prompt_enhancer_max_tokens': 1024,
            'prompt_enhancer_button_label': 'Refine',
            'prompt_enhancer_transcription_use_duration': True,
            'prompt_enhancer_def': {'selection': ['T', 'TW'], 'labels': {'T': 'AuK instruction from text', 'TW': 'AuK instruction from text + source transcription'}, 'default': '', 'letters_filter': 'TW'},
            'text_encoder_folder': ENCODER_FOLDER,
            'text_encoder_URLs': [build_hf_url(REPO_ID, ENCODER_FOLDER, f'Qwen2.5-Omni-3B_{suffix}.safetensors') for suffix in ('bf16', 'int8_convrot')],
            'any_audio_prompt': True, 'audio_prompt_choices': True, 'audio_guide_label': 'Source / reference audio',
            'audio_prompt_type_sources': {'selection': ['', 'A'], 'labels': {'': 'Instruction TTS', 'A': 'Voice cloning / edit source audio'}, 'default': '', 'letters_filter': 'A'},
            'duration_slider': {'label': 'Target duration (seconds)', 'name': 'Target Duration', 'min': 0.1, 'max': MAX_DURATION, 'increment': 0.1, 'default': 5},
            'infos': INFOS, 'prompt_infos': PROMPT_INFOS,
            'deepy_infos': DEEPY_INFOS, 'deepy_prompt_infos': DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return {'repoId': REPO_ID, 'sourceFolderList': ['auk', ENCODER_FOLDER], 'fileList': [['vae_bf16.safetensors'], ENCODER_FILES]}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, dtype=None, text_encoder_filename=None, save_quantized=False, **kwargs):
        from .pipeline import AuKPipeline

        config_path = os.path.join(os.path.dirname(__file__), 'auk.json')
        tokenizer_paths = [fl.locate_file(os.path.join(ENCODER_FOLDER, name)) for name in ENCODER_FILES]
        pipeline = AuKPipeline(model_filename[0], text_encoder_filename, fl.locate_file('auk/vae_bf16.safetensors'), os.path.dirname(tokenizer_paths[0]), config_path, flash=model_def.get('auk_flash', False), dtype=dtype or torch.bfloat16)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(pipeline.transformer, model_type, model_filename[0], dtype or torch.bfloat16, config_path)
        return pipeline, {'pipe': {'transformer': pipeline.transformer, 'text_encoder': pipeline.text_encoder, 'vae': pipeline.vae}, 'coTenantsMap': {}}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({'prompt': 'Say "Welcome back. It is wonderful to hear from you." in a warm, calm female voice.', 'audio_prompt_type': '', 'duration_seconds': 5, 'num_inference_steps': 4 if model_def.get('auk_flash', False) else 32, 'guidance_scale': 0 if model_def.get('auk_flash', False) else 2, 'negative_prompt': '', 'repeat_generation': 1, 'prompt_enhancer': '', 'video_length': 0, 'multi_prompts_gen_type': 'FG'})

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if not 0 < inputs['duration_seconds'] <= MAX_DURATION:
            return f'AuK target duration must be greater than zero and at most {MAX_DURATION} seconds.'
        if 'A' in inputs['audio_prompt_type']:
            if not inputs['audio_guide']:
                return 'Provide a source/reference audio clip for cloning or editing.'
            import gradio as gr
            import librosa
            source_duration = librosa.get_duration(path=inputs['audio_guide'])
            if source_duration > inputs['duration_seconds']:
                gr.Info(f'AuK will process only the first {inputs["duration_seconds"]:g} seconds of the source audio. The maximum per run is {MAX_DURATION} seconds.')
        return None
