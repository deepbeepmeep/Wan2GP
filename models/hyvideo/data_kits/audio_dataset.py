import os
import cv2
import math
import json
import torch
import random
import librosa
import traceback
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage



def get_audio_feature(feature_extractor, audio_path):
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
    assert sampling_rate == 16000

    audio_features = []
    window = 750*640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i+window], 
                                        sampling_rate=sampling_rate, 
                                        return_tensors="pt", 
                                        ).input_features
        audio_features.append(audio_feature)

    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640


class VideoAudioTextLoaderVal(Dataset):
    def __init__(
        self, 
        image_size: int,
        meta_file: str, 
        **kwargs,
    ):
        super().__init__()
        self.meta_file = meta_file
        self.image_size = image_size
        self.text_encoder = kwargs.get("text_encoder", None)            # llava_text_encoder
        self.text_encoder_2 = kwargs.get("text_encoder_2", None)        # clipL_text_encoder
        self.feature_extractor = kwargs.get("feature_extractor", None)
        self.meta_files = []
    
        csv_data = pd.read_csv(meta_file)
        for idx in range(len(csv_data)):
            self.meta_files.append(
                {
                    "videoid": str(csv_data["videoid"][idx]),
                    "image_path": str(csv_data["image"][idx]), 
                    "audio_path": str(csv_data["audio"][idx]), 
                    "prompt": str(csv_data["prompt"][idx]), 
                    "fps": float(csv_data["fps"][idx])
                }
            )
        
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        self.device = torch.device("cuda")
        self.weight_dtype = torch.float16

        
    def __len__(self):
        return len(self.meta_files)

    @staticmethod
    def get_text_tokens(text_encoder, description, dtype_encode="video"):
        text_inputs = text_encoder.text2tokens(description, data_type=dtype_encode)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask
    
    def get_batch_data(self, idx):
        meta_file = self.meta_files[idx]
        videoid = meta_file["videoid"]
        image_path = meta_file["image_path"]
        audio_path = meta_file["audio_path"]
        prompt = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, " + meta_file["prompt"]
        fps = meta_file["fps"]
        
        img_size = self.image_size
        ref_image = Image.open(image_path).convert('RGB')
        
        # Resize reference image
        w, h = ref_image.size
        scale = img_size / min(w, h)
        new_w = round(w * scale / 64) * 64
        new_h = round(h * scale / 64) * 64

        if img_size == 704:
            img_size_long = 1216
        if new_w * new_h > img_size * img_size_long:
            import math
            scale = math.sqrt(img_size * img_size_long / w / h)
            new_w = round(w * scale / 64) * 64
            new_h = round(h * scale / 64) * 64

        ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)
        
        ref_image = np.array(ref_image)
        ref_image = torch.from_numpy(ref_image)
         
        audio_input, audio_len = get_audio_feature(self.feature_extractor, audio_path)
        audio_prompts = audio_input[0]
        
        motion_bucket_id_heads = np.array([25] * 4)
        motion_bucket_id_exps = np.array([30] * 4)
        motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
        motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)
        fps = torch.from_numpy(np.array(fps))
        
        to_pil = ToPILImage()
        pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")   # (b c h w)
        
        pixel_value_ref_llava = [self.llava_transform(to_pil(image)) for image in pixel_value_ref]
        pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)
        pixel_value_ref_clip = self.clip_image_processor(
            images=Image.fromarray((pixel_value_ref[0].permute(1,2,0)).data.cpu().numpy().astype(np.uint8)), 
            return_tensors="pt"
        ).pixel_values[0]
        pixel_value_ref_clip = pixel_value_ref_clip.unsqueeze(0)
        
        # Encode text prompts
   
        text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt)
        text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt)
        
        # Output batch
        batch = {
            "text_prompt": prompt,                         # 
            "videoid": videoid, 
            "pixel_value_ref": pixel_value_ref.to(dtype=torch.float16),                 # 参考图，用于vae提特征 (1, 3, h, w), 取值范围(0, 255)
            "pixel_value_ref_llava": pixel_value_ref_llava.to(dtype=torch.float16),     # 参考图，用于llava提特征 (1, 3, 336, 336), 取值范围 = CLIP取值范围
            "pixel_value_ref_clip": pixel_value_ref_clip.to(dtype=torch.float16),       # 参考图，用于clip_image_encoder提特征 (1, 3, 244, 244), 取值范围 = CLIP取值范围
            "audio_prompts": audio_prompts.to(dtype=torch.float16), 
            "motion_bucket_id_heads": motion_bucket_id_heads.to(dtype=text_ids.dtype), 
            "motion_bucket_id_exps": motion_bucket_id_exps.to(dtype=text_ids.dtype), 
            "fps": fps.to(dtype=torch.float16), 
            "text_ids": text_ids.clone(),                                               # 对应llava_text_encoder
            "text_mask": text_mask.clone(),                                             # 对应llava_text_encoder
            "text_ids_2": text_ids_2.clone(),                                           # 对应clip_text_encoder
            "text_mask_2": text_mask_2.clone(),                                         # 对应clip_text_encoder
            "audio_len": audio_len,
            "image_path": image_path, 
            "audio_path": audio_path, 
        }
        return batch
    
    def __getitem__(self, idx):
        return self.get_batch_data(idx)
    

        
        