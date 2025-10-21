import torchvision
from PIL import Image, ImageDraw
import imageio
import cv2
import torch
import numpy as np 
import zipfile

def render_video(tensor_fgr,
                tensor_pha,
                nrow=8,
                normalize=True,
                value_range=(-1, 1)):
    def to_tensor(arr_list):
        tensor_list= [torch.from_numpy(arr).float().div_(127.5).sub_(1) for arr in arr_list]
        tensor_list = torch.stack(tensor_list, dim = 0).permute(3,0,1,2).unsqueeze(0)
        return tensor_list
                
    if not torch.is_tensor(tensor_fgr):
        tensor_fgr = to_tensor(tensor_fgr)
    if not torch.is_tensor(tensor_pha):
        tensor_pha = to_tensor(tensor_pha)

    tensor_fgr = tensor_fgr.clamp(min(value_range), max(value_range))
    tensor_fgr = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor_fgr.unbind(2)
    ],
                            dim=1).permute(1, 2, 3, 0)
    tensor_fgr = (tensor_fgr * 255).type(torch.uint8).cpu()

    tensor_pha = tensor_pha.clamp(min(value_range), max(value_range))
    tensor_pha = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor_pha.unbind(2)
    ],
                            dim=1).permute(1, 2, 3, 0)
    tensor_pha = (tensor_pha * 255).type(torch.uint8).cpu()

    frames = []
    frames_fgr = []
    frames_pha = []
    for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
        if frame_pha.shape[-1] == 1:
            frame_pha = frame_pha[:,:,0]
        else:
            frame_pha = (0.0 + frame_pha[:,:,0:1] + frame_pha[:,:,1:2] + frame_pha[:,:,2:3]) / 3.
        frame = np.concatenate([frame_fgr[:,:,::-1], frame_pha.astype(np.uint8)], axis=2)
        frames.append(frame)
        frames_fgr.append(frame_fgr)
        frames_pha.append(frame_pha)

    def create_checkerboard(size=30, pattern_size=(830, 480), color1=(140, 140, 140), color2=(113, 113, 113)):
        img = Image.new('RGB', (pattern_size[0], pattern_size[1]), color1)
        draw = ImageDraw.Draw(img)
        for i in range(0, pattern_size[0], size):
            for j in range(0, pattern_size[1], size):
                if (i + j) // size % 2 == 0:
                    draw.rectangle([i, j, i+size, j+size], fill=color2)
        return img

    def blender_background(frame_rgba, checkerboard):
        alpha_channel = frame_rgba[:, :, 3:] / 255. 
        checkerboard = np.array(checkerboard)
        checkerboard = cv2.resize(checkerboard, (frame_rgba.shape[1], frame_rgba.shape[0]))

        frame_rgb = frame_rgba[:, :, :3] * alpha_channel + checkerboard * (1-alpha_channel)
        return frame_rgb.astype(np.uint8)[:,:,::-1]
    
    checkerboard = create_checkerboard()
    video_checkerboard = [torch.from_numpy(blender_background(f, checkerboard).copy()).float().div_(127.5).sub_(1) for f in frames]
    video_checkerboard = torch.stack(video_checkerboard ).permute(3, 0, 1, 2)
    return video_checkerboard, frames

def from_BRGA_numpy_to_RGBA_torch(video):
    video = [torch.from_numpy(f.copy()).float().div_(127.5).sub_(1) for f in video]
    video = torch.stack(video).permute(3, 0, 1, 2)
    video[[0, 2], ...] = video[[2, 0], ...]
    return video

def write_zip_file(zip_path, frames):
    # frames in BGRA format
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, img in enumerate(frames):
            success, buffer = cv2.imencode(".png", img)
            if not success:
                print(f"Failed to encode image {idx}, skipping...")
                continue
            
            filename = f"img_{idx:03d}.png"
            zipf.writestr(filename, buffer.tobytes())