import logging
from typing import Union, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""
T2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

T2T_TEXT_PROMPT= """You are an expert speechwriter who crafts compelling, audience-appropriate speeches that effectively communicate the speaker's message while maintaining authenticity and impact.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

IT2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have the following information:
1. The user provides a general text input about its scenes expectations 
2. The user provides a caption of an image of a subject that relates to the scene
When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene evolves by following the user text input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have been provided with a caption of an image of a subject that relates to the scene to film.
Focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene should naturally evolves.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

IT2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
You have the following information:
1. The user provides a general text input about the expected photography 
2. The user provides a caption of an image of a subject he wants to be represented in the photography
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. 
You have been provided with a caption of an image of a subject to be represented in the photography.
Focus on detailed, descriptions of actions that are happening in the photography.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

def tensor_to_pil(tensor):
    # Ensure tensor is in range [-1, 1]
    assert tensor.min() >= -1 and tensor.max() <= 1

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # Convert to numpy array and then to uint8 range [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # Convert to PIL Image
    return Image.fromarray(numpy_image)


def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    images: Optional[List] = None,
    video_prompt= True,
    text_prompt = False,
    max_new_tokens: int = 256,
    prompt_enhancer_instructions = None,
) -> List[str]:
    prompts = [prompt] if isinstance(prompt, str) else prompt

    if images is None:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions=  T2T_TEXT_PROMPT if text_prompt else (T2V_CINEMATIC_PROMPT if video_prompt else T2I_VISUAL_PROMPT)
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            prompt_enhancer_instructions,
        )
    else:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions=  IT2V_CINEMATIC_PROMPT if video_prompt else IT2I_VISUAL_PROMPT

        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            images,
            max_new_tokens,
            prompt_enhancer_instructions,
        )

    return prompts


def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]


def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}"},
        ]
        for p in prompts
    ]

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    out_prompts = []
    for text in texts:
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        out_prompts.append(_generate_and_decode_prompts(prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens)[0])

    return out_prompts

def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    image_captions = _generate_image_captions(
        image_caption_model, image_caption_processor, first_frames
    )
    if len(image_captions) == 1 and len(image_captions) < len(prompts):
        image_captions *= len(prompts)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}\nimage_caption: {c}"},
        ]
        for p, c in zip(prompts, image_captions)
    ]

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    out_prompts = []
    for text in texts:
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        out_prompts.append(_generate_and_decode_prompts(prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens)[0])

    return out_prompts


def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
) -> List[str]:
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to("cuda") #.to(image_caption_model.device)

    with torch.inference_mode():
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    return image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _generate_and_decode_prompts(
    prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens: int
) -> List[str]:
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
            **model_inputs,  max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    return decoded_prompts
