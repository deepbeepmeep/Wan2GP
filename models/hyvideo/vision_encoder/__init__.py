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

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.utils import ModelOutput
import numpy as np

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


VISION_ENCODER_PATH = {}


def use_default(value, default):
    return value if value is not None else default


def load_vision_encoder(
    vision_encoder_type,
    vision_encoder_precision=None,
    vision_encoder_path=None,
    logger=None,
    device=None,
):
    if vision_encoder_path is None:
        vision_encoder_path = VISION_ENCODER_PATH[vision_encoder_type]

    if vision_encoder_type == "siglip":
        vision_encoder = SiglipVisionModel.from_pretrained(
            vision_encoder_path, 
            # subfolder='image_encoder'
        )
    else:
        raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")

    # from_pretrained will ensure that the model is in eval mode.
    if vision_encoder_precision is not None:
        vision_encoder = vision_encoder.to(dtype=PRECISION_TO_TYPE[vision_encoder_precision])

    vision_encoder.requires_grad_(False)

    if device is not None:
        vision_encoder = vision_encoder.to(device)

    return vision_encoder, vision_encoder_path


def load_image_processor(
    processor_type, 
    processor_path=None, 
    logger=None
):
    if processor_path is None:
        processor_path = VISION_ENCODER_PATH[processor_type]

    if processor_type == "siglip":
        processor = SiglipImageProcessor.from_pretrained(
            processor_path, 
            # subfolder='feature_extractor'
        )
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")

    return processor, processor_path


@dataclass
class VisionEncoderModelOutput(ModelOutput):
    """
    Base class for vision encoder model's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the first token of the sequence (classification token)
            after further processing through the layers used for the auxiliary pretraining task.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class VisionEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder_type: str,
        vision_encoder_precision: Optional[str] = None,
        vision_encoder_path: Optional[str] = None,
        processor_type: Optional[str] = None,
        processor_path: Optional[str] = None,
        output_key: Optional[str] = None,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.vision_encoder_type = vision_encoder_type
        self.precision = vision_encoder_precision
        self.model_path = vision_encoder_path
        self.processor_type = (
            processor_type if processor_type is not None else vision_encoder_type
        )
        self.processor_path = (
            processor_path if processor_path is not None else vision_encoder_path
        )
        self.logger = logger

        if "siglip" in vision_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported vision encoder type: {vision_encoder_type}")

        self.model, self.model_path = load_vision_encoder(
            vision_encoder_type=self.vision_encoder_type,
            vision_encoder_precision=self.precision,
            vision_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.processor, self.processor_path = load_image_processor(
            processor_type=self.processor_type,
            processor_path=self.processor_path,
            logger=self.logger,
        )

    def __repr__(self):
        return f"{self.vision_encoder_type} ({self.precision} - {self.model_path})"

    def encode_latents_to_images(self, latents, vae, reorg_token=False):
        """
        Convert latents to images using VAE decoder.
        
        Args:
            latents: Input latents tensor
            vae: VAE model for decoding
            reorg_token: Whether to reorg the token
        Returns:
            images: Decoded images as numpy array
        """
        # Handle both 4D and 5D latents (for video, take first frame)
        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
        first_image = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0].cpu()
        first_image = first_image[:, :, 0, :, :]
        first_image = (first_image / 2 + 0.5).clamp(0, 1)
        first_image = (first_image * 255.0).clamp(0, 255.0)
        first_image = first_image.to(torch.uint8).numpy()
        first_image = first_image.transpose(0, 2, 3, 1)

        assert isinstance(first_image, np.ndarray)
        assert first_image.ndim == 4 and first_image.shape[3] == 3
        assert first_image.dtype == np.uint8

        return first_image

    def encode_images(self, images):
        """
        Encode images using the vision encoder.
        
        Args:
            images: Input images (numpy array or preprocessed tensor)
            
        Returns:
            VisionEncoderModelOutput with encoded features
        """
        if isinstance(images, np.ndarray):
            # Preprocess images if they're numpy arrays
            preprocessed = self.processor.preprocess(images=images, return_tensors="pt").to(
                device=self.model.device, dtype=self.model.dtype)
        else:
            # Assume already preprocessed
            preprocessed = images

        outputs = self.model(**preprocessed)
        
        return VisionEncoderModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output if hasattr(outputs, 'pooler_output') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        )

    def encode_latents(self, latents, vae, reorg_token=False):
        """
        Encode latents by first converting to images, then encoding.
        This is the main function that replaces sigclip_vision_encode.
        
        Args:
            latents: Input latent tensors
            vae: VAE model for decoding latents to images
            
        Returns:
            Encoded image features
        """
        # Convert latents to images
        images = self.encode_latents_to_images(latents, vae, reorg_token)
        
        # Encode images
        outputs = self.encode_images(images)
        
        return outputs.last_hidden_state

    def forward(self, images):
        """
        Forward pass for direct image encoding.
        
        Args:
            images: Input images
            
        Returns:
            VisionEncoderModelOutput with encoded features
        """
        return self.encode_images(images) 