from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy
import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    AutoModel,
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
)
from transformers.utils import ModelOutput

from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
from ..constants import PRECISION_TO_TYPE
from .llava.modeling_llava import LlavaForConditionalGeneration 

from shared.utils import files_locator as fl

def use_default(value, default):
    return value if value is not None else default


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    device=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
        text_encoder_path = fl.locate_folder(text_encoder_path)

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True
        )
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "llm-i2v":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right"
):
    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
        tokenizer_path = fl.locate_folder(tokenizer_path)
    processor = None
    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    elif tokenizer_type == "llm-i2v":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
        processor = CLIPImageProcessor.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

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
        i2v_mode: bool = False,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        device=None,
#            image_embed_interleave (int): The number of times to interleave the image and text embeddings. Defaults to 2.
        image_embed_interleave=2,
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
            tokenizer_path if tokenizer_path is not None else None # text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.input_max_length = (
            input_max_length if input_max_length is not None else max_length
        )
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.i2v_mode = i2v_mode
        self.reproduce = reproduce
        self.image_embed_interleave = image_embed_interleave

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

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        if "llm" in text_encoder_type:
            from mmgp import offload
            # forcedConfigPath=  None if "i2v" in text_encoder_type  else "ckpts/llava-llama-3-8b/config.json"
            # self.model= offload.fast_load_transformers_model(self.model_path, modelPrefix="language_model" if forcedConfigPath != None else None,  forcedConfigPath=forcedConfigPath)

            if "i2v" in text_encoder_type:
                self.model= offload.fast_load_transformers_model(self.model_path, modelClass= LlavaForConditionalGeneration)
            else:
                self.model= offload.fast_load_transformers_model(self.model_path, modelPrefix="language_model", forcedConfigPath = fl.locate_file("llava-llama-3-8b/config.json"))
                self.model.final_layer_norm = self.model.model.norm


        
        else:
            self.model, self.model_path = load_text_encoder(
                text_encoder_type=self.text_encoder_type,
                text_encoder_precision=self.precision,
                text_encoder_path=self.model_path,
                device=device,
            )

        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
        )

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
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, data_type="image", name = None):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
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

        kwargs = dict(truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        if self.text_encoder_type == "llm-i2v" and name != None: #llava-llama-3-8b
            if isinstance(text, list):
                for i in range(len(text)):
                    text[i] = text[i] + '\nThe %s looks like<image>' % name
            elif isinstance(text, str):
                text = text + '\nThe %s looks like<image>' % name
            else:
                raise NotImplementedError

        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if tokenize_input_type == "str":
            return self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            return self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        semantic_images=None,
        device=None,
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
            hidden_state_skip_layer (PIL.Image): The reference images for i2v models.
            image_embed_interleave (int): The number of times to interleave the image and text embeddings. Defaults to 2.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)
        if not self.i2v_mode:
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )

            if 'pixel_value_llava' in batch_encoding:
                outputs = self.model(
                    input_ids=batch_encoding["input_ids"].to(self.model.device),
                    attention_mask=attention_mask,
                    pixel_values=batch_encoding["pixel_value_llava"].to(self.model.device),
                    output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None)
            else:
                outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(self.model.device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,)

            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", -1)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
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
        else:
            image_outputs = self.processor(semantic_images, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
                pixel_values=image_outputs,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]
            if self.use_template:
                if data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = (
                        crop_start
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    image_crop_start = self.prompt_template_video.get(
                        "image_emb_start", 5
                    )
                    image_crop_end = self.prompt_template_video.get(
                        "image_emb_end", 581
                    )
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"]
                        == self.prompt_template_video.get("double_return_token_id", 271)
                    )
                    if last_double_return_token_indices.shape[0] == 3:
                        # in case the prompt is too long
                        last_double_return_token_indices = torch.cat(
                            (
                                last_double_return_token_indices,
                                torch.tensor([batch_encoding["input_ids"].shape[-1]]),
                            )
                        )
                        batch_indices = torch.cat((batch_indices, torch.tensor([0])))
                    last_double_return_token_indices = (
                        last_double_return_token_indices.reshape(
                            batch_encoding["input_ids"].shape[0], -1
                        )[:, -1]
                    )
                    batch_indices = batch_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1
                    )[:, -1]
                    assistant_crop_start = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                        - 4
                    )
                    assistant_crop_end = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    attention_mask_assistant_crop_start = (
                        last_double_return_token_indices - 4
                    )
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                text_last_hidden_state = []

                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(
                        torch.cat(
                            [
                                last_hidden_state[
                                    i, text_crop_start : assistant_crop_start[i].item()
                                ],
                                last_hidden_state[i, assistant_crop_end[i].item() :],
                            ]
                        )
                    )
                    text_attention_mask.append(
                        torch.cat(
                            [
                                attention_mask[
                                    i,
                                    crop_start : attention_mask_assistant_crop_start[
                                        i
                                    ].item(),
                                ],
                                attention_mask[
                                    i, attention_mask_assistant_crop_end[i].item() :
                                ],
                            ]
                        )
                        if use_attention_mask
                        else None
                    )
                    image_last_hidden_state.append(
                        last_hidden_state[i, image_crop_start:image_crop_end]
                    )
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0])
                        .to(last_hidden_state.device)
                        .to(attention_mask.dtype)
                        if use_attention_mask
                        else None
                    )

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                if semantic_images is not None and 0 < self.image_embed_interleave < 6:
                    image_last_hidden_state = image_last_hidden_state[
                        :, ::self.image_embed_interleave, :
                    ]
                    image_attention_mask = image_attention_mask[
                        :, ::self.image_embed_interleave
                    ]

                assert (
                    text_last_hidden_state.shape[0] == text_attention_mask.shape[0]
                    and image_last_hidden_state.shape[0]
                    == image_attention_mask.shape[0]
                )

                last_hidden_state = torch.cat(
                    [image_last_hidden_state, text_last_hidden_state], dim=1
                )
                attention_mask = torch.cat(
                    [image_attention_mask, text_attention_mask], dim=1
                )
            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state,
                    attention_mask,
                    hidden_states_list=outputs.hidden_states,
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
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
