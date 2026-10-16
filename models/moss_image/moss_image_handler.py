# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from shared.utils.hf import build_hf_url
from .prompts import CLASSIC, STRUCTURED

REPO = 'DeepBeepMeep/Moss'
TEXT_FOLDER = 'Qwen3.5-9B'


class family_handler:
    @staticmethod
    def query_supported_types():
        return ['moss_image']

    @staticmethod
    def query_model_family():
        return 'moss_image'

    @staticmethod
    def query_family_infos():
        return {'moss_image': (1161, 'MOSS Image')}

    @staticmethod
    def query_family_maps():
        return {}, {'moss_image': ['moss_image']}

    @staticmethod
    def get_lora_dir(base_model_type):
        return 'moss_image'

    @staticmethod
    def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
        from .prompt_helper import render_prompt_helper
        return render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id)

    @staticmethod
    def get_prompt_helper_css():
        from .prompt_helper import get_prompt_helper_css
        return get_prompt_helper_css()

    @staticmethod
    def get_prompt_helper_javascript():
        from .prompt_helper import get_prompt_helper_javascript
        return get_prompt_helper_javascript()

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            'image_outputs': True, 'device_explicit': True,
            'group': 'moss_image', 'profiles_dir': ['moss_image'], 'inference_steps': True,
            'guidance_max_phases': 1, 'embedded_guidance': False, 'no_negative_prompt': False,
            'no_background_removal': True, 'vae_block_size': 16, 'resolution_budget': '2048x2048',
            'tiny_vae_architecture': 'flux2_klein_4b',
            'skip_prompt_template': True, 'preserve_empty_prompt_lines': True,
            'text_encoder_folder': TEXT_FOLDER,
            'text_encoder_URLs': [build_hf_url(REPO, TEXT_FOLDER, f'qwen3_5_9b_text_{fmt}.safetensors') for fmt in ('bf16', 'int8_convrot')],
            'infos': 'Generate images from natural-language prompts, tag sequences or experimental JSON layouts. Base and SFT use the same 5B architecture; SFT is the curated-data post-trained release. Both start at 50 steps, CFG 3 and shift 5, with layer 22 skipped in the negative branch. Try CFG 3â€“6 for short prompts. Upstream recommends about 1â€“2 megapixels and describes extrapolation from 512Ã—512 to about 4 megapixels. Dimensions must be multiples of 16. Batch Size generates images together; larger batches use more VRAM. Layout edits change the prompt for a new generation; these checkpoints do not accept reference images or pixel masks. Prompt enhancement is optional and off by default.',
            'prompt_infos': 'Use Prompt Helper to draw, move and resize items and edit their exact visible text. It recovers common JSON punctuation mistakes and reports repairs before saving; unrepairable input stays available in the text editor. Save preserves comments and extra fields. The layout enhancer uses the selected output resolution. Re-enhance after changing resolution, or use Fit Boxes to Canvas to move existing boxes inside while preserving their size where possible. Describe the subject, composition, lighting, style and exact visible words. English has the most training coverage; Chinese and other languages are supported. Quote lettering and use explicit line breaks when needed, e.g. `A cream travel poster with blue mountains, title "EXPLORE", subtitle "Find your next trail".` Tags can follow quality, score, meta, year, character, series, artist and general content. JSON layouts use `summary.caption` and `items`, with `label`, `caption`, optional `text` and `bbox:[y_min,x_min,y_max,x_max]`. Set `bbox_format:"yxyx"` and `bbox_coord_space:"abs_centralized"`. Coordinates are pixel corners measured from the canvas center, X right and Y down: a 1024Ã—1024 canvas runs from -512 to 512 on both axes. Keep boxes inside the requested dimensions. Text and placement are guidance, not exact guarantees. Whole comment lines beginning with # are ignored during encoding.',
            'deepy_infos': 'Text-to-image only: `prompt` as natural language, tags or JSON layout; no image_refs, audio inputs or masks. Both releases default to num_inference_steps=50, guidance_scale=3, flow_shift=5, resolution="1024x1024", batch_size=1, video_prompt_type="", prompt_enhancer="". batch_size>1 denoises and decodes all images together with seeds seed, seed+1, etc.; larger batches use more VRAM. Negative-branch layer 22 skipping is built in. Dimensions must be multiples of 16; budget about 4 megapixels, recommended 1â€“2 megapixels. Base and SFT share architecture and adapters; neither is a step-distilled release.',
            'deepy_prompt_infos': 'Optional prompt_enhancer="T1" produces a layout using the selected resolution and centered pixel bounds; "T1K" adds Think. "T" produces a classic caption. Re-enhance layouts after changing resolution. The UI Prompt Helper edits centered-pixel boxes, reports punctuation repairs and preserves comments/extra fields on save. Use multi_prompts_gen_type="FG" for multiline captions/JSON. Quote exact text, specify line breaks and describe full scene/style. English recommended; preserve requested language. Optional layout: {summary:{meta:["poster"],canvas:"single",caption:"scene"},items:[{label:"text",caption:"appearance",text:"EXACT WORDS",bbox:[ymin,xmin,ymax,xmax]}],bbox_coord_space:"abs_centralized",bbox_format:"yxyx"}. Serialize as valid JSON. Integer pixel corners, origin canvas center, X right/Y down; bounds X +/-width/2, Y +/-height/2. Every visible element belongs to one item. No reference editing. Comments whose first nonspace character is # are ignored in encoding.',
            'text_prompt_enhancer_instructions': CLASSIC, 'text_prompt_enhancer_instructions1': STRUCTURED,
            'text_prompt_enhancer_max_tokens1': 4096,
            'prompt_enhancer_def': {'selection': ['T', 'T1'], 'labels': {'T': 'A Detailed Image Caption', 'T1': 'An Image Layout with Bounding Boxes'}, 'default': ''},
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{'repoId': REPO, 'sourceFolderList': [TEXT_FOLDER, 'moss_image'],
                 'fileList': [['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'], ['flux2_small_decoder_bf16.safetensors']]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, dtype=torch.bfloat16, **kwargs):
        from .pipeline import load_pipeline
        pipeline = load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized, dtype)
        return pipeline, {'transformer': pipeline.transformer, 'text_encoder': pipeline.text_encoder, 'vae': pipeline.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({'image_mode': 1, 'resolution': '1024x1024', 'batch_size': 1, 'num_inference_steps': 50,
                            'guidance_scale': 3, 'flow_shift': 5, 'prompt_enhancer': '', 'video_prompt_type': '', 'multi_prompts_gen_type': 'FG'})

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors('flux', sub_family='flux2')
