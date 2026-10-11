# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
import torch
from models.help_shared import specialities

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

REPO = 'DeepBeepMeep/Wan2.2'
ENCODER_FOLDER = 'UMT5-XXL-Transformers'
ENCODER_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
VIDEO_VAE = 'Wan2.1_VAE_bf16.safetensors'
AUDIO_VAE = 'prism_audio_vae_bf16.safetensors'

INFOS = '''Prism Alpha generates a video and a synchronized 48 kHz mono soundtrack from a starting image and a prompt. A starting image is required. Describe motion, camera work and sound together, or use Audio Prompt for a separate soundtrack description. Audio Prompt is text, not a recording or a voice-cloning input.

The upstream reference setting is 1280×720, 205 frames, 24 fps, 50 steps, guidance 5 and flow shift 7. Full Video Attention is the default. Sparse Video Attention is experimental: a 480p comparison produced damaged final frames in Sparse mode while Full remained coherent. Smaller images and shorter clips run faster but can lose quality; 480p is an experimental test size. Higher resolutions and longer clips require substantially more memory and time. The checkpoint supports multiple resolutions; 2K on a consumer GPU is not guaranteed. Frame counts follow 4n+1, starting at 49. This is a preview model: audio quality can vary substantially with the prompt, seed and starting image. No end-image, audio-reference or video-editing workflow is exposed. Prompt enhancement is optional and disabled by default.

For faster generation, use the shared Wan2.2 **Lightning i2v** accelerator profiles. They select paired high/low video adapters and two or three guidance phases. Keep the profile's adapters, phase multipliers, switch thresholds and flow shift together. Prism enables four audio updates per video step with guidance 5 for these adapters. Full Video Attention is recommended. These are Wan2.2 video accelerators adapted to Prism's audio workflow; audio quality and motion can differ from the 50-step base model. Select I2V profiles, not T2V or Fun Inp profiles. To return to base sampling, clear the LoRAs and restore Euler, 50 steps, guidance 5 in both phases, expert switch 900 and flow shift 7.

Sliding-window continuation is unavailable: the one-frame continuation experiment produced a motion pause and deteriorating picture quality. Generate each clip in a single pass, with 49–289 frames in steps of four.
'''
PROMPT_INFOS = '''Describe the scene, subject and actions in temporal order, followed by camera movement, lighting and audible events. Keep the description consistent with the starting image. Use `<speech>exact words</speech>` for dialogue, `<sfx>sound description</sfx>` for effects, `<music>music description</music>` for music and `<text>visible wording</text>` for written content. These are prompt cues, not guaranteed transcription, typography or exact lip sync.

Example: A woman at a bakery counter smiles, raises a fresh loaf and says in a warm voice <speech>Fresh out of the oven.</speech> The camera slowly approaches. <sfx>Quiet bakery ambience and a tray settling on wood.</sfx> No background music.

Audio Prompt optionally describes the soundtrack separately; leave it empty to reuse the main prompt. Negative Prompt describes unwanted content. Avoid contradicting exact speech or visible text in the negative prompt.
'''
ENHANCER = '''Refine the user's Prism image-to-video and audio prompt. Preserve language, intent, exclusions and every exact supplied spoken or visible string. Describe actions in order, camera motion, lighting and sound consistent with the starting image. Use <speech>exact dialogue</speech>, <sfx>sound effects</sfx>, <music>music</music>, and <text>visible writing</text> only where requested. If the request needs written content but leaves wording open, author all intended visible copy and enclose each complete string in <text>...</text>. Do not add unrelated speech, music or writing. Return only the enhanced prompt, without explanation or Markdown fences.'''


class family_handler:
    @staticmethod
    def query_supported_types(): return ['prism']

    @staticmethod
    def query_family_maps(): return {}, {}

    @staticmethod
    def query_model_family(): return 'prism'

    @staticmethod
    def query_family_infos(): return {}

    @staticmethod
    def get_lora_dir(base_model_type): return 'wan'

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors('wan')

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            'specialities': specialities('audio-video generation'),
            'device_explicit': True, 'dtype': 'bf16', 'size': 'large', 'returns_audio': True, 'multimedia_generation': True,
            'fps': 24, 'frames_minimum': 49, 'frames_steps': 4, 'frames_selection_maximum': 289, 'sliding_window': False, 'video_continuation': False,
            'image_prompt_types_allowed': 'S', 'i2v_class': True, 'capability_overrides': {'text_to_video': False}, 'lora_multiplier_phases': 3,
            'guidance_max_phases': 3, 'embedded_guidance': False, 'inference_steps': True, 'flow_shift': True, 'profiles_dir': ['wan_2_2'],
            'tiny_vae_architecture': 't2v', 'compile': False, 'vae_block_size': 16,
            'resolution_budget': '2048x2048',
            'sample_solvers': [('Euler', 'euler'), ('LightX2V With Audio Substeps', 'lightx2v')],
            'alt_prompt': {'label': 'Audio Prompt (Optional)', 'name': 'Audio Prompt', 'lines': 3},
            'model_modes': {'label': 'Video Attention', 'choices': [('Full', 0), ('Sparse (Experimental)', 1)], 'default': 0},
            'text_encoder_folder': ENCODER_FOLDER,
            'text_encoder_URLs': [build_hf_url(REPO, ENCODER_FOLDER, 'UMT5-XXL_int8_convrot.safetensors')],
            'infos': INFOS, 'prompt_infos': PROMPT_INFOS,
            'deepy_infos': 'Prism Alpha preview: starting image + prompt → video with generated 48 kHz mono audio. image_start is required. alt_prompt optionally supplies separate audio text. Upstream reference: 1280x720, 205 frames, 24 fps, 50 steps, guidance 5, flow_shift 7, sample_solver=euler. For acceleration, select a shared Wan2.2 Lightning I2V profile and keep its adapter pair, phase multipliers, thresholds and flow shift. Prism enables four audio substeps per video step with CFG 5. Two- and three-phase video guidance/expert switching follow the profile. To return to base, clear LoRAs and restore the reference settings. Full Video Attention (model_mode=0) is the default; Sparse (1) is experimental and damaged final frames in a 480p comparison. Generate each clip in a single pass: 49..289 frames in steps of 4. Sliding-window continuation is unavailable because the one-frame experiment produced a motion pause and deteriorating picture quality. Smaller sizes such as 480p are experimental and can lose quality. Audio quality varies with prompt, seed and image. Higher resolutions/lengths cost much more memory. No supplied audio, voice cloning, video editing or ending image. Enhancement is opt-in.',
            'deepy_prompt_infos': 'Describe actions in order, camera and sound consistent with image_start. Inline cues: <speech>exact words</speech>, <sfx>sound</sfx>, <music>music</music>, <text>visible writing</text>. Example: The woman smiles and says <speech>Welcome back.</speech> <sfx>Birdsong outside.</sfx> No background music. alt_prompt overrides the audio description only; empty reuses prompt. Exact speech, typography and lip sync are not guaranteed.',
            'video_prompt_enhancer_instructions': ENHANCER, 'text_prompt_enhancer_instructions': ENHANCER,
            'text_prompt_enhancer_max_tokens': 1024, 'video_prompt_enhancer_max_tokens': 1024,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {'repoId': REPO, 'sourceFolderList': ['', ENCODER_FOLDER], 'fileList': [[AUDIO_VAE], ENCODER_FILES]},
            {'repoId': 'DeepBeepMeep/Wan2.1', 'sourceFolderList': [''], 'fileList': [[VIDEO_VAE]]},
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, **kwargs):
        from .pipeline import PrismPipeline
        tokenizer_files = [fl.locate_file(os.path.join(ENCODER_FOLDER, name)) for name in ENCODER_FILES]
        pipeline = PrismPipeline(model_filename, text_encoder_filename, os.path.dirname(tokenizer_files[0]),
                                 fl.locate_file(VIDEO_VAE), fl.locate_file(AUDIO_VAE))
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(pipeline.model, model_type, model_filename[0], torch.bfloat16, None)
            save_quantized_model(pipeline.model2, model_type, model_filename[1], torch.bfloat16, None, submodel_no=2)
        return pipeline, {'transformer': pipeline.model, 'transformer2': pipeline.model2, 'text_encoder': pipeline.text_encoder, 'vae': pipeline.vae.model, 'audio_vae': pipeline.audio_vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(num_inference_steps=50, guidance_scale=5.0, guidance2_scale=5.0, guidance_phases=2, switch_threshold=900, model_switch_phase=1, flow_shift=7.0, video_length=205, resolution='1280x720', image_prompt_type='S', prompt_enhancer='', alt_prompt='', model_mode=0, sample_solver='euler')

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        selected = [os.path.basename(path.replace('\\', '/')).lower() for path in inputs['activated_loras']]
        accelerators = [name for name in selected if 'i2v' in name and ('lightx2v' in name or 'seko' in name)]
        if accelerators:
            if not all(any(expert in name for name in accelerators) for expert in ('high', 'low')):
                return 'Select both high- and low-noise adapters from a Wan2.2 Lightning I2V profile.'
            inputs['sample_solver'] = 'lightx2v'
