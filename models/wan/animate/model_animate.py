# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy
from einops import  rearrange
from typing import List
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn

def after_patch_embedding(self, x: List[torch.Tensor], pose_latents, face_pixel_values):
    pose_latents = self.pose_patch_embedding(pose_latents)
    x[:, :, 1:] += pose_latents
    
    b,c,T,h,w = face_pixel_values.shape
    face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")
    encode_bs = 8
    face_pixel_values_tmp = []
    for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
        face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))

    motion_vec = torch.cat(face_pixel_values_tmp)
    
    motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
    motion_vec = self.face_encoder(motion_vec)

    B, L, H, C = motion_vec.shape
    pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
    motion_vec = torch.cat([pad_face, motion_vec], dim=1)
    return x, motion_vec
