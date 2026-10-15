# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from shared.utils.hf import build_hf_url
from .prompts import CLASSIC, STRUCTURED

REPO = 'DeepBeepMeep/Flux3'
TEXT_FOLDER = 'Qwen3-VL-4B-Instruct'


class family_handler:
    @staticmethod
    def query_supported_types():
        return ['flux3_klein', 'flux3_klein_base']

    @staticmethod
    def query_model_family():
        return 'flux3'

    @staticmethod
    def query_family_infos():
        return {'flux3': (1102, 'Flux 3')}

    @staticmethod
    def query_family_maps():
        return {}, {'flux3_klein': ['flux3_klein'], 'flux3_klein_base': ['flux3_klein_base']}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

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
        distilled = base_model_type == 'flux3_klein'
        settings = '16 steps, no CFG' if distilled else '50 steps, guidance_scale=3.5'
        return {
            'image_outputs': True, 'device_explicit': True, 'cuda_graphs': ['transformer'],
            'skip_prompt_template': True, 'preserve_empty_prompt_lines': True,
            'group': 'flux3', 'profiles_dir': [base_model_type], 'inference_steps': True,
            'guidance_max_phases': 0 if distilled else 1, 'embedded_guidance': False,
            'no_negative_prompt': distilled, 'no_background_removal': True,
            'vae_block_size': 16, 'resolution_budget': '4096x4096', 'max_width': 4096, 'max_height': 4096,
            'tiny_vae_architecture': 'flux2_klein_4b',
            'text_encoder_folder': TEXT_FOLDER,
            'text_encoder_URLs': [build_hf_url(REPO, TEXT_FOLDER, f'qwen3vl_4b_{fmt}.safetensors') for fmt in ('bf16', 'int8_convrot')],
            'image_ref_choices': {
                'choices': [('None', ''),
                            ('Main Image Ref - First Reference Sets Output Dimensions (up to 10 Images)', 'KI'),
                            ('Ordered Reference Images (up to 10)', 'I')],
                'letters_filter': 'KI', 'default': '',
            },
            'fit_into_canvas_image_refs': 0,
            **({'accelerated': 'native'} if distilled else {}),
            'infos': f'Generate from a Text Prompt, or combine the prompt with up to 10 ordered Reference Images for edits and compositions. Main Image Ref uses the first reference to set output dimensions through the selected resolution and canvas fitting mode; later images remain additional references. Ordered Reference Images keeps the selected output dimensions. Start with {settings}. References anchor content but do not lock composition. Describe the complete desired image, including what remains unchanged. Bounding boxes in the prompt guide placement; they are not pixel masks, and edits may affect surrounding content. Prompt enhancement is optional and off by default. The upstream model supports sides from 64 to 4096 pixels, aligned to 16 pixels; start at 1024×1024. Large images and many references need more memory.',
            'prompt_infos': 'Use Prompt Helper to draw, move and resize layout boxes. For reference edits, select a Content Source and use the Source view to mark its region, then place its Output Box; removing a reference element sets a null target. The editor saves the caption and element list together, preserving comments and extra fields. Write a complete scene caption: framing, subject, background, lighting and style. For an edit, fold the change into a full description of the reference scene, e.g. "A side view of the same red bicycle against the white brick wall, now fitted with a wicker basket, soft afternoon light." Quote exact visible lettering. Identify references as `<ref_image_0>`, `<ref_image_1>`, etc. For layout, tag elements with `<id>` in the caption and append a JSON array: `[{"id":"title","bbox":[80,100,220,900],"desc":"Navy title reading \\"NIGHT GARDEN\\""}]`. Coordinates are `[top,left,bottom,right]` on a 0–1000 grid, with the origin at top left. Reference edits use `from`, `src_bbox` and `tgt_bbox` instead of `bbox`; a null target removes an element. Review enhancer-generated coordinates: some LLMs may swap the vertical and horizontal axes. Keep caption and JSON together in one prompt. Text accuracy, exact placement and preservation are not guaranteed.',
            'deepy_infos': f'Text-to-image or editing: `prompt` plus optional ordered `image_refs` (maximum 10). Use `video_prompt_type="KI"` for a main image reference: `image_refs[0]` sets output dimensions through `resolution` and `fit_canvas`; later images remain additional references. Use `video_prompt_type="I"` to keep the requested `resolution` with references, or empty for text-only generation. Defaults: {settings}. `resolution` dimensions must be multiples of 16, each 64–4096. Bounding boxes are prompt guidance, not hard masks. No native inpainting mask workflow is declared. `prompt_enhancer=""` by default.',
            'deepy_prompt_infos': 'Describe the entire finished scene, including retained reference details and requested changes; short edit commands can reframe it. Reference IDs are zero-based `<ref_image_0>`, etc. Quote exact visible strings. Optional layout: caption with `<id>` tags followed by JSON elements {id,desc,bbox:[top,left,bottom,right]}, integer 0–1000 corners. Editing elements use {id,desc,from,src_bbox,tgt_bbox}; from is ref_image_0, ref_image_1, etc. or target; null tgt_bbox removes. Use multi_prompts_gen_type="FG" for a multiline caption and JSON. Review enhancer boxes for Y/X swaps; boxes guide placement, not exact pixels.',
            'prompt_enhancer_def': {'selection': ['T', 'TI', 'T1', 'TI1'], 'labels': {'T': 'A Complete Scene Caption', 'TI': 'A Complete Scene Caption using {image_inputs}', 'T1': 'A Scene Caption with Bounding Boxes', 'TI1': 'A Scene Edit with Bounding Boxes using {image_inputs}'}, 'default': ''},
            'text_prompt_enhancer_instructions': CLASSIC, 'image_prompt_enhancer_instructions': CLASSIC,
            'text_prompt_enhancer_instructions1': STRUCTURED, 'image_prompt_enhancer_instructions1': STRUCTURED,
            'text_prompt_enhancer_max_tokens1': 4096, 'image_prompt_enhancer_max_tokens1': 4096,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{'repoId': REPO, 'sourceFolderList': [TEXT_FOLDER, 'flux3'],
                 'fileList': [['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'], ['flux2-vae.safetensors']]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, dtype=torch.bfloat16, **kwargs):
        from .pipeline import load_pipeline
        pipeline = load_pipeline(model_filename, text_encoder_filename, base_model_type, model_def, model_type, save_quantized, dtype)
        return pipeline, {'transformer': pipeline.transformer, 'text_encoder': pipeline.text_encoder, 'vae': pipeline.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        distilled = base_model_type == 'flux3_klein'
        ui_defaults.update({'image_mode': 1, 'resolution': '1024x1024', 'batch_size': 1, 'num_inference_steps': 16 if distilled else 50,
                            'guidance_scale': 1 if distilled else 3.5, 'prompt_enhancer': '', 'video_prompt_type': '', 'remove_background_images_ref': 0,
                            'multi_prompts_gen_type': 'FG'})

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if len(inputs.get('image_refs') or []) > 10:
            return 'Flux 3 Klein supports at most 10 Reference Images.'
        width, height = map(int, inputs['resolution'].split('x'))
        if any(side < 64 or side > 4096 or side % 16 for side in (width, height)):
            return 'Flux 3 dimensions must be multiples of 16, between 64 and 4096 pixels.'

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors('flux', sub_family='flux2')
