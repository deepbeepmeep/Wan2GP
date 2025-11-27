from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import AutoProcessor, Mistral3ForConditionalGeneration, pipeline
from shared.utils import files_locator as fl 
from .sampling import cap_pixels, concatenate_images
from .system_messages import (
    PROMPT_IMAGE_INTEGRITY,
    PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
    PROMPT_TEXT_INTEGRITY,
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
    SYSTEM_PROMPT_CONTENT_FILTER,
)

OUTPUT_LAYERS = [10, 20, 30]
MAX_LENGTH = 512
NSFW_THRESHOLD = 0.85
UPSAMPLING_MAX_IMAGE_SIZE = 768**2

from mmgp import offload
import os

class Mistral3SmallEmbedder(nn.Module):
    def __init__(
        self,
        model_spec = None,
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()
        file_path = model_spec
        self.model = offload.fast_load_transformers_model(file_path, writable_tensors= False, defaultConfigPath= os.path.join(os.path.dirname(file_path), "config.json"))
        self.processor = AutoProcessor.from_pretrained(os.path.dirname(file_path), use_fast=False)
        self.yes_token, self.no_token = self.processor.tokenizer.encode(
            ["yes", "no"], add_special_tokens=False
        )

        self.max_length = MAX_LENGTH
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

        self.nsfw_classifier = None 

    def _validate_and_process_images(
        self, img: list[list[Image.Image]] | list[Image.Image]
    ) -> list[list[Image.Image]]:
        # Simple validation: ensure it's a list of PIL images or list of lists of PIL images
        if not img:
            return []

        # Check if it's a list of lists or a list of images
        if isinstance(img[0], Image.Image):
            # It's a list of images, convert to list of lists
            img = [[im] for im in img]

        # potentially concatenate multiple images to reduce the size
        img = [[concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in img]

        # cap the pixels
        img = [[cap_pixels(img_i, self.upsampling_max_image_size) for img_i in img_i] for img_i in img]
        return img

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[list[dict]]:
        """
        Format a batch of text prompts into the conversation format expected by apply_chat_template.
        Optionally, add images to the input.

        Args:
            txt: List of text prompts
            system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
            img: List of images to add to the input.

        Returns:
            List of conversations, where each conversation is a list of message dicts
        """
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        if img is None or len(img) == 0:
            return [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                for prompt in cleaned_txt
            ]
        else:
            assert len(img) == len(txt), "Number of images must match number of prompts"
            img = self._validate_and_process_images(img)

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                ]
                for _ in cleaned_txt
            ]

            for i, (el, images) in enumerate(zip(messages, img)):
                # optionally add the images per batch element.
                if images is not None:
                    el.append(
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": image_obj} for image_obj in images],
                        }
                    )
                # add the text.
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cleaned_txt[i]}],
                    }
                )

            return messages

    @torch.no_grad()
    def upsample_prompt(
        self,
        txt: list[str],
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
        temperature: float = 0.15,
    ) -> list[str]:
        """
        Upsample prompts using the model's generate method.

        Args:
            txt: List of input prompts to upsample
            img: Optional list of images or list of lists of images. If None or all None, uses t2i mode, otherwise i2i mode.

        Returns:
            List of upsampled prompts
        """
        # Set system message based on whether images are provided
        if img is None or len(img) == 0 or img[0] is None:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I

        # Format input messages
        messages_batch = self.format_input(txt=txt, system_message=system_message, img=img)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        try:
            inputs = self.processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048,
            )
        except ValueError as e:
            print(
                f"Error processing input: {e}, your max length is probably too short, when you have images in the input."
            )
            raise e

        # Move to device
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device, self.model.dtype)

        # Generate text using the model's generate method
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
            )

            # Decode only the newly generated tokens (skip input tokens)
            # Extract only the generated portion
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]

            raw_txt = self.processor.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return raw_txt
        except Exception as e:
            print(f"Error generating upsampled prompt: {e}, returning original prompt")
            return txt

    @torch.no_grad()
    def forward(self, txt: list[str]):
        # Format input messages
        messages_batch = self.format_input(txt=txt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Sets all tokens but yes/no to the minimum.
        """
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    def test_image(self, image: Image.Image | str | Path | torch.Tensor) -> bool:
        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, (str, Path)):
            image = Image.open(image)

        classification = next(c for c in self.nsfw_classifier(image) if c["label"] == "nsfw")
        if classification["score"] > NSFW_THRESHOLD:
            return True

        # 512^2 pixels are enough for checking
        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )

        return generate_ids[0, -1].item() == self.yes_token

    def test_txt(self, txt: str) -> bool:
        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token
