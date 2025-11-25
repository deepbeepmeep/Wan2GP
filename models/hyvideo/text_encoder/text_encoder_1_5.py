# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple
from shared.utils import files_locator as fl 
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from transformers.utils import ModelOutput


def use_default(value, default):
    """Utility: return value if not None, else default."""
    return value if value is not None else default

# Prompt templates for different models and tasks


__all__ = [
    "C_SCALE", "PROMPT_TEMPLATE",
    "MODEL_BASE",
]

# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE = {
    "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON, "crop_start": -1}, # auto-calculate crop_start
    "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, "crop_start": -1}, # auto-calculate crop_start
}


MODEL_BASE = os.getenv("MODEL_BASE", "")
TEXT_ENCODER_PATH = {}
TOKENIZER_PATH = {}

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if text_encoder_path is None:
        if text_encoder_type not in TEXT_ENCODER_PATH:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]

    from mmgp import offload
    # text_encoder = offload.fast_load_transformers_model(text_encoder_path, forcedConfigPath=  os.path.join(os.path.dirname(text_encoder_path), "config.json"))
    text_encoder = offload.fast_load_transformers_model(text_encoder_path,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration,  defaultConfigPath= fl.locate_file(os.path.join("Qwen2.5-VL-7B-Instruct", "config.json")) )

    # text_encoder = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
    
    if hasattr(text_encoder, 'language_model'):
        text_encoder = text_encoder.language_model
    text_encoder.final_layer_norm = text_encoder.norm
    
    # from_pretrained will ensure that the model is in eval mode.
    # if text_encoder_precision is not None:
    #     text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    processor = None
    if tokenizer_path is None:
        if tokenizer_type not in TOKENIZER_PATH:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.dirname(tokenizer_path), padding_side=padding_side
    )

    # If the checkpoint ships a chat template separately, load it when missing.
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        jinja_path = os.path.join(os.path.dirname(tokenizer_path), "chat_template.json")
        if os.path.exists(jinja_path):
            with open(jinja_path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None
    image_features: Optional[list] = None

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        i2v_mode = None,
        image_embed_interleave = None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if text_encoder_type != "llm":
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
        self.output_key = output_key or "last_hidden_state"

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

        # pre-calculate crop_start for image and video
        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")

    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        elif isinstance(template, list):
            # For JSON list template format (chat conversation)
            # Create a deep copy to avoid modifying the original template
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    # Replace placeholder with text in the content field
                    item["content"] = item["content"].format(text if text else (" " if prevent_empty_text else ""))
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input):
        """
        Automatically calculate the crop_start position based on identifying user tokens.
        
        Args:
            tokenized_input: The output from the tokenizer containing input_ids
            
        Returns:
            int: The position where the actual prompt content begins (after user markers)
        """
        input_ids = tokenized_input["input_ids"][0].tolist()  # Get the first example's tokens
        
        marker = "<|im_start|>user\n"
            
        # Tokenize just the marker to get its token IDs
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]
        
        # Find the end position of the marker in the input sequence
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i:i+len(marker_tokens)] == marker_tokens:
                # Return the position after the marker
                return i + len(marker_tokens)
                
        # If marker not found, try to find based on special tokens
        if hasattr(self.tokenizer, 'special_tokens_map'):
            # Check for user token or any other special token that might indicate user input start
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if 'user' in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1
        
        # Default fallback: return 0 (no cropping)
        return 0

    def text2tokens(self, text, data_type="image", max_length=300):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template)
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        
            # First pass: tokenize with arbitrary max_length to find crop_start
            if crop_start == -1:
                # Use temporary max_length for the first pass (large enough)
                temp_kwargs = dict(
                    truncation=True,
                    max_length=256,  # Temporary large value
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # First tokenization pass to calculate crop_start
                if tokenize_input_type == "str":
                    temp_tokenized = self.tokenizer(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = self.tokenizer.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )
                
                # Calculate the crop_start from this first pass
                crop_start = self.calculate_crop_start(temp_tokenized)
                
                # Store the calculated crop_start for future use
                if data_type == "image":
                    self.prompt_template["crop_start"] = crop_start
                else:
                    self.prompt_template_video["crop_start"] = crop_start
        else:
            crop_start = 0
        
        # Second pass: tokenize with the proper max_length using the found crop_start
        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )
        
        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")
                
        return tokenized_output

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        device=None,
        is_uncond=False
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)

        attention_mask = (
            batch_encoding["attention_mask"].to(device) if use_attention_mask else None
        )
        outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
            or hidden_state_skip_layer is not None,
        )
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if self.use_template:
            if data_type == "image":
                crop_start = self.prompt_template.get("crop_start", 0)
            elif data_type == "video":
                crop_start = self.prompt_template_video.get("crop_start", 0)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = (
                    attention_mask[:, crop_start:] if use_attention_mask else None
                )

        if output_hidden_states:
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, outputs.hidden_states
            )
        return TextEncoderModelOutput(last_hidden_state, attention_mask)


    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
