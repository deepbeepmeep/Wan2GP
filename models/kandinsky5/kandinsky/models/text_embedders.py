import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    BitsAndBytesConfig
)

from .utils import freeze
from mmgp import offload
import torchvision.transforms.functional as F


class ClipTextEmbedder:
    def __init__(self, conf, device):
        self.model = CLIPTextModel.from_pretrained(conf.checkpoint_path).to(device)
        self.model = freeze(self.model)
        self.tokenizer = CLIPTokenizer.from_pretrained(conf.checkpoint_path)
        self.max_length = conf.max_length

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            pooled_embed = self.model(**inputs)["pooler_output"]
        return pooled_embed


class Qwen2_5_VLTextEmbedder:
    INSTRUCTION_I2V_EXPAND = """You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
    Here are some example descriptions from the dataset that the model was trained:
    1. "Create a video showing a nighttime urban driving scene from inside a car. The driver is focused on the road ahead, with the city lights visible through the windshield. The GPS device on the dashboard continues to display navigation information. The camera remains steady, capturing the interior of the car and the changing street view outside as the vehicle moves forward. The background shifts slightly to show different parts of the cityscape, including illuminated buildings and street signs."
    2. "Create a video where the character, dressed in historical attire, is seen holding an umbrella with a logo. The character should move closer to the camera while maintaining a steady pace, keeping the umbrella raised. The background remains consistent with a foggy, outdoor setting, but the focus shifts more towards the character as they approach. The lighting should emphasize the details of the costume and the umbrella, enhancing the dramatic effect."
    3. "Darken the scene while keeping the characters and setting unchanged, emphasizing a serious atmosphere."
    IImportantly! These are just examples from a large training dataset of 20 mln videos.
    Rewrite Prompt: "{prompt}" to get high-quality image to video generation from this image. Pay main attention to information about changes of objects.
    Make prompt dynamic. Answer only with expanded prompt."""
    PROMPT_TEMPLATE = {
        "template": {
            "video": (
                "<|im_start|>system\nYou are a prompt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image2video": (
                "<|im_start|>system\nYou are a prompt engineer. Your task is to create a highly detailed and effective video description based on a provided input image.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe main characters actions.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image": (
                "<|im_start|>system\nYou are a prompt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image_edit": (
                "<|im_start|>system\nYou are a prompt engineer. Based on the provided source image (first image) and target image (second image), create an interesting text prompt that can be used together with the source image to create the target image:<|im_end|>",
                "<|im_start|>user\n{}"
            )
        },
        "crop_start": {
            "video": 129,
            "image": 41,
            "image_edit": 55,
            "image2video": 132
        },
    }

    def __init__(self, conf, device, quantized_qwen=False, text_token_padding=True):
        self._fast_loaded = False
        checkpoint_path = conf.checkpoint_path
        if os.path.isfile(checkpoint_path):
            config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
            self.model = offload.fast_load_transformers_model(
                checkpoint_path,
                writable_tensors=True,
                modelClass=Qwen2_5_VLForConditionalGeneration,
                defaultConfigPath=config_path,
                default_dtype=torch.bfloat16,
            )
            self.model = self.model.to(device)
            self._fast_loaded = True
            self.processor = AutoProcessor.from_pretrained(os.path.dirname(checkpoint_path), use_fast=True)
            self.max_length = conf.max_length
            self.text_token_padding = text_token_padding
            self.model = freeze(self.model)
            return

        quantization_config = None
        if quantized_qwen:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quantization_config
        )
        self.model = freeze(self.model)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, use_fast=True)
        self.max_length = conf.max_length
        self.text_token_padding = text_token_padding

    def __call__(self, texts, images=None, type_of_content="video"):
        prompt_template = "\n".join(self.PROMPT_TEMPLATE["template"][type_of_content])
        crop_start = self.PROMPT_TEMPLATE["crop_start"][type_of_content]
        full_texts = list(map(lambda x: prompt_template.format(x), texts))
        if type_of_content == "image_edit":
            if images is not None:
                for i in range(len(images)):
                    image_tokens = ''.join(['<|vision_start|><|image_pad|><|vision_end|>']*len(images[i]))
                    full_texts[i] = full_texts[i] + image_tokens + "<|im_end|>"
                images = [F.resize(i, (i.shape[-2] // 2, i.shape[-1] // 2)) for i in images]
            max_length = (self.max_length + crop_start) if images is None else None
            inputs = self.processor(
                text=full_texts,
                images=images, 
                truncation=True,
                return_tensors="pt", 
                padding=True,
                max_length=max_length
            ).to(self.model.device)

            with torch.no_grad():
                embeds = self.model(**inputs, output_hidden_states=True)["hidden_states"][-1][:, crop_start:]
        else:
            max_length = self.max_length + crop_start
            inputs = self.processor(
                text=full_texts,
                images=None,
                videos=None,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding="max_length",
            ).to(self.model.device)

            with torch.no_grad():
                embeds = self.model(
                    input_ids=inputs["input_ids"],
                    return_dict=True,
                    output_hidden_states=True,
                )["hidden_states"][-1][:, crop_start:]
        attention_mask = inputs["attention_mask"][:, crop_start:]
        if self.text_token_padding:
            seq_length = embeds.shape[1]
            cu_seqlens = torch.tensor([0, seq_length], dtype=torch.int32)
        else:
            embeds = embeds[attention_mask.bool()].unsqueeze(0)
            cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
            cu_seqlens = torch.cat([torch.zeros_like(cu_seqlens)[:1], cu_seqlens]).to(
                dtype=torch.int32
            )
            attention_mask = None
        return embeds, cu_seqlens, attention_mask

    def expand_text_prompt(self, prompt, image, device="cuda"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": self.INSTRUCTION_I2V_EXPAND.format(prompt=prompt),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]


class Kandinsky5TextEmbedder:
    def __init__(self, conf, device="cpu", quantized_qwen=False, text_token_padding=False):
        self.embedder = Qwen2_5_VLTextEmbedder(conf.qwen, device, quantized_qwen, text_token_padding)
        self.clip_embedder = ClipTextEmbedder(conf.clip, device)
        self.conf = conf

    def encode(self, texts, images=None, type_of_content="image"):
        text_embeds, cu_seqlens, attention_mask = self.embedder(texts, images=images, type_of_content=type_of_content)
        pooled_embed = self.clip_embedder(texts)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
        return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens, attention_mask

    def to(self, device):
        self.embedder.model = self.embedder.model.to(device)
        self.clip_embedder.model = self.clip_embedder.model.to(device)
        return self


def get_text_embedder(conf, device="cpu", quantized_qwen=False, text_token_padding=True):
    return Kandinsky5TextEmbedder(conf, device, quantized_qwen, text_token_padding)
