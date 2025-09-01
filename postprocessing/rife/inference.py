import os
import torch
from torch.nn import functional as F
# from .model.pytorch_msssim import ssim_matlab
from .ssim import ssim_matlab

from .RIFE_HDv3 import Model

def get_frame(frames, frame_no):
    if frame_no >= frames.shape[1]:
        return None
    frame = (frames[:, frame_no] + 1) /2
    frame = frame.clip(0., 1.)
    return frame

def add_frame(frames, frame, h, w):
    frame = (frame * 2) - 1
    frame = frame.clip(-1., 1.)    
    frame = frame.squeeze(0)
    frame = frame[:, :h, :w]
    frame = frame.unsqueeze(1)
    frames.append(frame.cpu())

def process_frames(model, device, frames, exp):
    pos = 0
    output_frames = []

    lastframe = get_frame(frames, 0)
    _,  h, w = lastframe.shape
    scale = 1
    fp16 = False

    def make_inference(I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad_image(img):
        if(fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    I1 = lastframe.to(device, non_blocking=True).unsqueeze(0)
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            pos += 1
            frame = get_frame(frames, pos)
        if frame is None:
            break
        I0 = I1
        I1 = frame.to(device, non_blocking=True).unsqueeze(0)
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996 or pos > 100:        
            pos += 1
            frame = get_frame(frames, pos)
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = frame.to(device, non_blocking=True).unsqueeze(0)
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = I1[0][:, :h, :w]
        
        if ssim < 0.2:
            output = []
            for _ in range((2 ** exp) - 1):
                output.append(I0)
        else:
            output = make_inference(I0, I1, 2**exp-1) if exp else []

        add_frame(output_frames, lastframe, h, w)
        for mid in output:
            add_frame(output_frames, mid, h, w)
        lastframe = frame
        if break_flag:
            break

    add_frame(output_frames, lastframe, h, w)
    return torch.cat( output_frames, dim=1)

def temporal_interpolation(model_path, frames, exp, device ="cuda"):

    model = Model()
    model.load_model(model_path, -1, device=device)

    model.eval()
    model.to(device=device)

    with torch.no_grad():    
        output = process_frames(model, device, frames.float(), exp)

    return output
