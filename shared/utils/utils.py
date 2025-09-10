# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import os.path as osp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
import torch
import decord
from PIL import Image
import numpy as np
from rembg import remove, new_session
import random
import ffmpeg
import os
import tempfile
import subprocess
import json
from functools import lru_cache
os.environ["U2NET_HOME"] = os.path.join(os.getcwd(), "ckpts", "rembg")


from PIL import Image
video_info_cache = []
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def resample(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame ):
    import math

    video_frame_duration = 1 /video_fps
    target_frame_duration = 1 / target_fps 
    
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)  
    cur_time = frame_no * video_frame_duration
    frame_ids =[]
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count :
            break
        diff = round( (target_time -cur_time) / video_frame_duration , 5)
        add_frames_count = math.ceil( diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:             
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids

import os
from datetime import datetime

def get_file_creation_date(file_path):
    # On Windows
    if os.name == 'nt':
        return datetime.fromtimestamp(os.path.getctime(file_path))
    # On Unix/Linux/Mac (gets last status change, not creation)
    else:
        stat = os.stat(file_path)
    return datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime)

def truncate_for_filesystem(s, max_bytes=255):
    if len(s.encode('utf-8')) <= max_bytes: return s
    l, r = 0, len(s)
    while l < r:
        m = (l + r + 1) // 2
        if len(s[:m].encode('utf-8')) <= max_bytes: l = m
        else: r = m - 1
    return s[:l]

@lru_cache(maxsize=100)
def get_video_info(video_path):
    global video_info_cache
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap.release()
    
    return fps, width, height, frame_count

def get_video_frame(file_name: str, frame_no: int, return_last_if_missing: bool = False, return_PIL = True) -> torch.Tensor:
    """Extract nth frame from video as PyTorch tensor normalized to [-1, 1]."""
    cap = cv2.VideoCapture(file_name)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {file_name}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle out of bounds
    if frame_no >= total_frames or frame_no < 0:
        if return_last_if_missing:
            frame_no = total_frames - 1
        else:
            cap.release()
            raise IndexError(f"Frame {frame_no} out of bounds (0-{total_frames-1})")
    
    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_no}")
    
    # Convert BGR->RGB, reshape to (C,H,W), normalize to [-1,1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if return_PIL:
          return Image.fromarray(frame)
    else:
        return (torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5) - 1.0
# def get_video_frame(file_name, frame_no):
#     decord.bridge.set_bridge('torch')
#     reader = decord.VideoReader(file_name)

#     frame = reader.get_batch([frame_no]).squeeze(0)
#     img = Image.fromarray(frame.numpy().astype(np.uint8))
#     return img

def convert_image_to_video(image):
    if image is None:
        return None
    
    # Convert PIL/numpy image to OpenCV format if needed
    if isinstance(image, np.ndarray):
        # Gradio images are typically RGB, OpenCV expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    height, width = img_bgr.shape[:2]
    
    # Create temporary video file (auto-cleaned by Gradio)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, 30.0, (width, height))
        out.write(img_bgr)
        out.release()
        return temp_video.name
    
def resize_lanczos(img, h, w):
    img = (img + 1).float().mul_(127.5)
    img = Image.fromarray(np.clip(img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = img.resize((w,h), resample=Image.Resampling.LANCZOS) 
    img = torch.from_numpy(np.array(img).astype(np.float32)).movedim(-1, 0)
    img = img.div(127.5).sub_(1)
    return img

def remove_background(img, session=None):
    if session ==None:
        session = new_session() 
    img = Image.fromarray(np.clip(255. * img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = remove(img, session=session, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).movedim(-1, 0)


def convert_image_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)

def convert_tensor_to_image(t, frame_no = -1):    
    t = t[:, frame_no] if frame_no >= 0 else t
    return Image.fromarray(t.clone().add_(1.).mul_(127.5).permute(1,2,0).to(torch.uint8).cpu().numpy())

def save_image(tensor_image, name, frame_no = -1):
    convert_tensor_to_image(tensor_image, frame_no).save(name)

def get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    frame_height = int(frame_height * (100 + outpainting_top + outpainting_bottom) / 100)
    frame_width =  int(frame_width * (100 + outpainting_left + outpainting_right) / 100)
    return frame_height, frame_width  

def rgb_bw_to_rgba_mask(img, thresh=127):
    a = img.convert('L').point(lambda p: 255 if p > thresh else 0)  # alpha
    out = Image.new('RGBA', img.size, (255, 255, 255, 0))           # white, transparent
    out.putalpha(a)                                                 # white where alpha=255
    return out
                        


def  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, block_size = 8):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    raw_height = int(final_height / ((100 + outpainting_top + outpainting_bottom) / 100))
    height = int(raw_height / block_size) * block_size
    extra_height = raw_height - height
          
    raw_width = int(final_width / ((100 + outpainting_left + outpainting_right) / 100)) 
    width = int(raw_width / block_size) * block_size
    extra_width = raw_width - width  
    margin_top = int(outpainting_top/(100 + outpainting_top + outpainting_bottom) * final_height)
    if extra_height != 0 and (outpainting_top + outpainting_bottom) != 0:
        margin_top += int(outpainting_top / (outpainting_top + outpainting_bottom) * extra_height)
    if (margin_top + height) > final_height or outpainting_bottom == 0: margin_top = final_height - height
    margin_left = int(outpainting_left/(100 + outpainting_left + outpainting_right) * final_width)
    if extra_width != 0 and (outpainting_left + outpainting_right) != 0:
        margin_left += int(outpainting_left / (outpainting_left + outpainting_right) * extra_height)
    if (margin_left + width) > final_width or outpainting_right == 0: margin_left = final_width - width
    return height, width, margin_top, margin_left

def rescale_and_crop(img, w, h):
    ow, oh = img.size
    target_ratio = w / h
    orig_ratio = ow / oh
    
    if orig_ratio > target_ratio:
        # Crop width first
        nw = int(oh * target_ratio)
        img = img.crop(((ow - nw) // 2, 0, (ow + nw) // 2, oh))
    else:
        # Crop height first
        nh = int(ow / target_ratio)
        img = img.crop((0, (oh - nh) // 2, ow, (oh + nh) // 2))
    
    return img.resize((w, h), Image.LANCZOS)

def calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas,  block_size = 16):
    if fit_into_canvas == None or fit_into_canvas == 2:
        # return image_height, image_width
        return canvas_height, canvas_width
    if fit_into_canvas == 1:
        scale1  = min(canvas_height / image_height, canvas_width / image_width)
        scale2  = min(canvas_width / image_height, canvas_height / image_width)
        scale = max(scale1, scale2) 
    else: #0 or #2 (crop)
        scale = (canvas_height * canvas_width / (image_height * image_width))**(1/2)

    new_height = round( image_height * scale / block_size) * block_size
    new_width = round( image_width * scale / block_size) * block_size
    return new_height, new_width

def calculate_dimensions_and_resize_image(image, canvas_height, canvas_width, fit_into_canvas, fit_crop, block_size = 16):
    if fit_crop:
        image = rescale_and_crop(image, canvas_width, canvas_height)
        new_width, new_height = image.size  
    else:
        image_width, image_height = image.size
        new_height, new_width = calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas, block_size = block_size )
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
    return image, new_height, new_width

def resize_and_remove_background(img_list, budget_width, budget_height, rm_background, any_background_ref, fit_into_canvas = 0, block_size= 16, outpainting_dims = None ):
    if rm_background:
        session = new_session() 

    output_list =[]
    for i, img in enumerate(img_list):
        width, height =  img.size 
        if fit_into_canvas == None or any_background_ref == 1 and i==0 or any_background_ref == 2:
            if outpainting_dims is not None:
                resized_image =img 
            elif img.size != (budget_width, budget_height):
                resized_image= img.resize((budget_width, budget_height), resample=Image.Resampling.LANCZOS) 
            else:
                resized_image =img
        elif fit_into_canvas == 1:
            white_canvas = np.ones((budget_height, budget_width, 3), dtype=np.uint8) * 255 
            scale = min(budget_height / height, budget_width / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
            top = (budget_height - new_height) // 2
            left = (budget_width - new_width) // 2
            white_canvas[top:top + new_height, left:left + new_width] = np.array(resized_image)            
            resized_image = Image.fromarray(white_canvas)  
        else:
            scale = (budget_height * budget_width / (height * width))**(1/2)
            new_height = int( round(height * scale / block_size) * block_size)
            new_width = int( round(width * scale / block_size) * block_size)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        if rm_background  and not (any_background_ref and i==0 or any_background_ref == 2) :
            # resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1,alpha_matting_background_threshold = 70, alpha_foreground_background_threshold = 100, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
            resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
        output_list.append(resized_image) #alpha_matting_background_threshold = 30, alpha_foreground_background_threshold = 200,
    return output_list




def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


import sys, time

# Global variables to track download progress
_start_time = None
_last_time = None
_last_downloaded = 0
_speed_history = []
_update_interval = 0.5  # Update speed every 0.5 seconds

def progress_hook(block_num, block_size, total_size, filename=None):
    """
    Simple progress bar hook for urlretrieve
    
    Args:
        block_num: Number of blocks downloaded so far
        block_size: Size of each block in bytes
        total_size: Total size of the file in bytes
        filename: Name of the file being downloaded (optional)
    """
    global _start_time, _last_time, _last_downloaded, _speed_history, _update_interval
    
    current_time = time.time()
    downloaded = block_num * block_size
    
    # Initialize timing on first call
    if _start_time is None or block_num == 0:
        _start_time = current_time
        _last_time = current_time
        _last_downloaded = 0
        _speed_history = []
    
    # Calculate download speed only at specified intervals
    speed = 0
    if current_time - _last_time >= _update_interval:
        if _last_time > 0:
            current_speed = (downloaded - _last_downloaded) / (current_time - _last_time)
            _speed_history.append(current_speed)
            # Keep only last 5 speed measurements for smoothing
            if len(_speed_history) > 5:
                _speed_history.pop(0)
            # Average the recent speeds for smoother display
            speed = sum(_speed_history) / len(_speed_history)
        
        _last_time = current_time
        _last_downloaded = downloaded
    elif _speed_history:
        # Use the last calculated average speed
        speed = sum(_speed_history) / len(_speed_history)
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    file_display = filename if filename else "Unknown file"
    
    if total_size <= 0:
        # If total size is unknown, show downloaded bytes
        speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
        line = f"\r{file_display}: {format_bytes(downloaded)}{speed_str}"
        # Clear any trailing characters by padding with spaces
        sys.stdout.write(line.ljust(80))
        sys.stdout.flush()
        return
    
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    
    # Create progress bar (40 characters wide to leave room for other info)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
    
    # Display progress with filename first
    line = f"\r{file_display}: [{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_str}"
    # Clear any trailing characters by padding with spaces
    sys.stdout.write(line.ljust(100))
    sys.stdout.flush()
    
    # Print newline when complete
    if percent >= 100:
        print()

# Wrapper function to include filename in progress hook
def create_progress_hook(filename):
    """Creates a progress hook with the filename included"""
    global _start_time, _last_time, _last_downloaded, _speed_history
    # Reset timing variables for new download
    _start_time = None
    _last_time = None
    _last_downloaded = 0
    _speed_history = []
    
    def hook(block_num, block_size, total_size):
        return progress_hook(block_num, block_size, total_size, filename)
    return hook


