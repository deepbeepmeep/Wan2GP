
import os
import cv2
import json
import time
import decord
import einops
import librosa
import torch
import random
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange



def get_facemask(ref_image, align_instance, area=1.25):
    # ref_image: (b f c h w)
    bsz, f, c, h, w = ref_image.shape
    images = rearrange(ref_image, "b f c h w -> (b f) h w c").data.cpu().numpy().astype(np.uint8)
    face_masks = []
    for image in images:
        image_pil = Image.fromarray(image).convert("RGB")
        _, _, bboxes_list = align_instance(np.array(image_pil)[:,:,[2,1,0]], maxface=True)
        try:
            bboxSrc = bboxes_list[0]
        except:
            bboxSrc = [0, 0, w, h]
        x1, y1, ww, hh = bboxSrc
        x2, y2 = x1 + ww, y1 + hh
        ww, hh = (x2-x1) * area, (y2-y1) * area
        center = [(x2+x1)//2, (y2+y1)//2]
        x1 = max(center[0] - ww//2, 0)
        y1 = max(center[1] - hh//2, 0)
        x2 = min(center[0] + ww//2, w)
        y2 = min(center[1] + hh//2, h)
        
        face_mask = np.zeros_like(np.array(image_pil))
        face_mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
        face_masks.append(torch.from_numpy(face_mask[...,:1]))
    face_masks = torch.stack(face_masks, dim=0)     # (b*f, h, w, c)
    face_masks = rearrange(face_masks, "(b f) h w c -> b c f h w", b=bsz, f=f)
    face_masks = face_masks.to(device=ref_image.device, dtype=ref_image.dtype)
    return face_masks


def encode_audio(wav2vec, audio_feats, fps, num_frames=129):
    if fps == 25:
        start_ts = [0]
        step_ts = [1]
    elif fps == 12.5:
        start_ts = [0]
        step_ts = [2]
    num_frames = min(num_frames, 400)
    audio_feats = wav2vec.encoder(audio_feats.unsqueeze(0)[:, :, :3000], output_hidden_states=True).hidden_states
    audio_feats = torch.stack(audio_feats, dim=2)
    audio_feats = torch.cat([torch.zeros_like(audio_feats[:,:4]), audio_feats], 1)
    
    audio_prompts = []
    for bb in range(1):
        audio_feats_list = []
        for f in range(num_frames):
            cur_t = (start_ts[bb] + f * step_ts[bb]) * 2
            audio_clip = audio_feats[bb:bb+1, cur_t: cur_t+10]
            audio_feats_list.append(audio_clip)
        audio_feats_list = torch.stack(audio_feats_list, 1)
        audio_prompts.append(audio_feats_list)
    audio_prompts = torch.cat(audio_prompts)
    return audio_prompts