# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

from typing import Optional, Union

import torch
import numpy as np

from PIL import Image
from dataclasses import dataclass, field


dtype_mapping = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}


@dataclass
class SubjectInfo:
    name: str = ""
    image_pil: Optional[Image.Image] = None
    landmarks: Optional[Union[np.ndarray, torch.Tensor]] = None
    face_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None


@dataclass
class VideoStyleInfo:  # key names should match those used in style.yaml file
    style_name: str = 'none'
    num_frames: int = 81
    seed: int = -1
    guidance_scale: float = 5.0
    guidance_scale_i: float = 2.0
    num_inference_steps: int = 50
    width: int = 832
    height: int = 480
    prompt: str = ''
    negative_prompt: str = ''