import sys

import os
import json
import time
import psutil
# import ffmpeg
import imageio
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from datetime import datetime
from .tools.painter import mask_painter
from .tools.interact_tools import SamControler
from .tools.misc import get_device
from .tools.download_util import load_file_from_url
from segment_anything.modeling.image_encoder import window_partition, window_unpartition, get_rel_pos, Block as image_encoder_block
from .utils.get_default_model import get_matanyone_model
from .matanyone.inference.inference_core import InferenceCore
from .matanyone_wrapper import matanyone
from shared.utils.audio_video import save_video, save_image
from mmgp import offload
from shared.utils import files_locator as fl 
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, calculate_new_dimensions, get_default_workers
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources, any_GPU_process_running

arg_device = "cuda"
arg_sam_model_type="vit_h"
arg_mask_save = False
model_loaded = False
model = None
matanyone_model = None
model_in_GPU = False
matanyone_in_GPU = False
bfloat16_supported = False
PlugIn = None

# SAM generator
import copy
GPU_process_was_running = False
def acquire_GPU(state):
    global GPU_process_was_running
    GPU_process_was_running = any_GPU_process_running(state, "matanyone")
    acquire_GPU_ressources(state, "matanyone", "MatAnyone", gr= gr)      
def release_GPU(state):
    release_GPU_ressources(state, "matanyone")
    if GPU_process_was_running:
        global matanyone_in_GPU, model_in_GPU
        if model_in_GPU:  
            model.samcontroler.sam_controler.model.to("cpu")
            model_in_GPU = False
        if matanyone_in_GPU:
            matanyone_model.to("cpu")
            matanyone_in_GPU = False


def perform_spatial_upsampling(frames, new_dim):
    if new_dim =="":
        return frames
    h, w = frames[0].shape[:2]
    
    from shared.utils.utils import resize_lanczos 
    pos = new_dim.find(" ")
    fit_into_canvas = "Outer" in new_dim
    new_dim = new_dim[:pos]
    if new_dim == "1080p":
        canvas_w, canvas_h = 1920, 1088
    elif new_dim == "720p":
        canvas_w, canvas_h = 1280, 720
    else:
        canvas_w, canvas_h = 832, 480
    h, w = calculate_new_dimensions(canvas_h, canvas_w, h, w, fit_into_canvas=fit_into_canvas,  block_size= 16  )


    def upsample_frames(frame):
        return np.array(Image.fromarray(frame).resize((w,h), resample=Image.Resampling.LANCZOS))
    
    output_frames = process_images_multithread(upsample_frames, frames, "upsample", wrap_in_list = False, max_workers=get_default_workers(), in_place=True)    
    return output_frames

class MaskGenerator():
    def __init__(self, sam_checkpoint, device):
        global args_device
        args_device  = device
        self.samcontroler = SamControler(sam_checkpoint, arg_sam_model_type, arg_device)
       
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    
# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def get_frames_from_image(state, image_input, image_state, new_dim):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """

    if image_input is None:
       gr.Info("Please select an Image file")
       return [gr.update()] * 20


    if len(new_dim)  > 0:
        image_input = perform_spatial_upsampling([image_input], new_dim)[0]

    user_name = time.time()
    frames = [image_input] * 2  # hardcode: mimic a video with 2 frames
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    image_state = {
        "user_name": user_name,
        "image_name": "output.png",
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "last_frame_numer": 0,
        "fps": None,
        "new_dim": new_dim,
        }
        
    image_info = "Image Name: N/A,\nFPS: N/A,\nTotal Frames: {},\nImage Size:{}".format(len(frames), image_size)
    acquire_GPU(state)
    set_image_encoder_patch()
    select_SAM(state)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(image_state["origin_images"][0])
    torch.cuda.empty_cache()
    release_GPU(state)

    return image_state, gr.update(interactive=False), image_info, image_state["origin_images"][0], \
                        gr.update(visible=True, maximum=10, value=10), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(), \
                        gr.update(visible=False), gr.update(value="", visible=False),  gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)


# extract frames from upload video
def get_frames_from_video(state, video_input, video_state, new_dim):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    if video_input is None:
       gr.Info("Please select a Video file")
       return [gr.update()] * 19
     
        
    video_path = video_input
    frames = []
    user_name = time.time()

    # extract Audio
    # try:
    #     audio_path = video_input.replace(".mp4", "_audio.wav")
    #     ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100').run(overwrite_output=True, quiet=True)
    # except Exception as e:
    #     print(f"Audio extraction error: {str(e)}")
    #     audio_path = ""  # Set to "" if extraction fails
    # print(f'audio_path: {audio_path}')
    audio_path = ""     
    # extract frames
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 

    if len(new_dim) > 0:
        frames = perform_spatial_upsampling(frames, new_dim)
        image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # resize if resolution too big
    if image_size[0]>=1280 and image_size[0]>=1280:
        scale = 1080 / min(image_size)
        new_w = int(image_size[1] * scale)
        new_h = int(image_size[0] * scale)
        # update frames
        frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
        # update image_size
        image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "last_frame_number": 0,
        "fps": fps,
        "audio": audio_path,
        "new_dim": new_dim,
        }
    video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
    acquire_GPU(state)
    set_image_encoder_patch()
    select_SAM(state)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    torch.cuda.empty_cache()    
    release_GPU(state)
    return video_state, gr.update(interactive=False), video_info, video_state["origin_images"][0], \
                        gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# get the select frame from gradio slider
def select_video_template(image_selection_slider,  video_state, interactive_state):

    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

def select_image_template(image_selection_slider, video_state, interactive_state):

    image_selection_slider = 0 # fixed for image
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider

    return video_state["painted_images"][track_pause_number_slider],interactive_state


def patched_forward(self, x: torch.Tensor) -> torch.Tensor:        
    def split_mlp(mlp, x, divide = 4):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        chunk_size = int(x.shape[0]/divide)
        x_chunks = torch.split(x, chunk_size)
        for i, x_chunk  in enumerate(x_chunks):
            mlp_chunk = mlp.lin1(x_chunk)
            mlp_chunk = mlp.act(mlp_chunk)
            x_chunk[...] = mlp.lin2(mlp_chunk)
        return x.reshape(x_shape)     

    def get_decomposed_rel_pos( q, rel_pos_h, rel_pos_w, q_size, k_size) -> torch.Tensor:
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)
        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = torch.zeros(B, q_h, q_w, k_h, k_w, dtype=q.dtype, device=q.device)
        attn += rel_h[:, :, :, :, None]
        attn += rel_w[:, :, :, None, :]
        return attn.view(B, q_h * q_w, k_h * k_w)

    def pay_attention(self, x: torch.Tensor, split_heads = 1) -> torch.Tensor:
            B, H, W, _ = x.shape
            # qkv with shape (3, B, nHead, H * W, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

            if not bfloat16_supported: qkv = qkv.to(torch.float16)

            # q, k, v with shape (B * nHead, H * W, C)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            if split_heads == 1:
                attn_mask = None
                if self.use_rel_pos:
                    attn_mask = get_decomposed_rel_pos(q, self.rel_pos_h.to(q), self.rel_pos_w.to(q), (H, W), (H, W))
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
            else:
                chunk_size = self.num_heads // split_heads 
                x = torch.empty_like(q)
                q_chunks = torch.split(q, chunk_size)
                k_chunks = torch.split(k, chunk_size)
                v_chunks = torch.split(v, chunk_size)
                x_chunks = torch.split(x, chunk_size)
                for x_chunk, q_chunk, k_chunk, v_chunk  in zip(x_chunks, q_chunks, k_chunks, v_chunks):
                    attn_mask = None
                    if self.use_rel_pos:
                        attn_mask = get_decomposed_rel_pos(q_chunk, self.rel_pos_h.to(q), self.rel_pos_w.to(q), (H, W), (H, W))
                    x_chunk[...]  = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=attn_mask, scale=self.scale)
                del x_chunk, q_chunk, k_chunk, v_chunk
            del q, k, v, attn_mask
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            if not bfloat16_supported: x = x.to(torch.bfloat16)

            return self.proj(x)

    shortcut = x
    x = self.norm1(x)
    # Window partition
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)
    x_shape = x.shape

    if x_shape[0] > 10:
        chunk_size = int(x.shape[0]/4) + 1
        x_chunks = torch.split(x, chunk_size)
        for i, x_chunk  in enumerate(x_chunks):
            x_chunk[...] = pay_attention(self.attn,x_chunk)  
    else:
        x = pay_attention(self.attn,x, 4)


    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
    x += shortcut
    shortcut[...] = self.norm2(x)
    # x += self.mlp(shortcut)
    x +=  split_mlp(self.mlp, shortcut)

    return x

def set_image_encoder_patch():
    if not hasattr(image_encoder_block, "patched"):  #and False
        image_encoder_block.forward = patched_forward
        image_encoder_block.patched = True

# use sam to get the mask
def sam_refine(state, video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData ): #
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    acquire_GPU(state)
    select_SAM(state)
    # prompt for sam model
    set_image_encoder_patch()
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    torch.cuda.empty_cache()
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    torch.cuda.empty_cache()
    release_GPU(state)
    return painted_image, video_state, interactive_state

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    masks = video_state["masks"]
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update()]*4
    mask = masks[video_state["select_frame_number"]]
    interactive_state["multi_mask"]["masks"].append(mask)
    interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    select_frame = show_mask(video_state, interactive_state, mask_dropdown)

    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]]

def clear_click(video_state, click_state):
    masks = video_state["masks"]
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update()]*2

    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return template_frame, click_state

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    return interactive_state, gr.update(choices=[],value=[])

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if video_state["origin_images"]:
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["multi_mask"]["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        
        return select_frame


# def save_video(frames, output_path, fps):

#     writer = imageio.get_writer( output_path, fps=fps, codec='libx264', quality=8)
#     for frame in frames:
#         writer.append_data(frame)
#     writer.close()

#     return output_path

def mask_to_xyxy_box(mask):
    rows, cols = np.where(mask == 255)
    if len(rows) == 0 or len(cols) == 0: return []
    xmin = min(cols)
    xmax = max(cols) + 1
    ymin = min(rows)
    ymax = max(rows) + 1
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, mask.shape[1])
    ymax = min(ymax, mask.shape[0])
    box = [xmin, ymin, xmax, ymax]
    box = [int(x) for x in box]
    return box

def get_dim_file_suffix(new_dim):
    if not " " in new_dim: return ""
    pos = new_dim.find(" ")
    return new_dim[:pos]

# image matting
def image_matting(state, video_state, interactive_state, mask_type, matting_type, new_new_dim, mask_dropdown, erode_kernel_size, dilate_kernel_size, refine_iter):
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload an Image")
        return [gr.update(visible=False)]*12

    new_dim = video_state.get("new_dim", "")
    if new_new_dim != new_dim:
        gr.Info(f"You have changed the Input / Output Dimensions after loading the Video into Matanyone. The output dimension will be the ones when loading the image ({'original' if len(new_dim) == 0 else new_dim})")

    matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
    acquire_GPU(state)
    select_matanyone(state)
    foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=refine_iter)
    torch.cuda.empty_cache()    
    release_GPU(state)

    foreground_mat = matting_type == "Foreground"
    
    foreground_output = None
    foreground_title = "Image with Background"
    alpha_title = "Alpha Mask Image Output"

    if mask_type == "wangp":
        white_image = np.full_like(following_frames[-1], 255, dtype=np.uint8)
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 
        output_frame = (white_image.astype(np.uint16) * (255 - alpha_output.astype(np.uint16)) + 
                        following_frames[-1].astype(np.uint16) * alpha_output.astype(np.uint16))
        output_frame = output_frame // 255
        output_frame = output_frame.astype(np.uint8)
        foreground_output = output_frame
        control_output = following_frames[-1]
        alpha_output = alpha_output[:,:,0]     

        foreground_title = "Image without Background" if foreground_mat else "Image with Background"
        control_title = "Control Image"
        allow_export = True
        control_output = following_frames[-1]
        tab_label = "Control Image & Mask"
    elif mask_type == "greenscreen":
        green_image = np.zeros_like(following_frames[-1], dtype=np.uint8)
        green_image[:, :, 1] = 255          
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 

        output_frame = (following_frames[-1].astype(np.uint16) * (255 - alpha_output.astype(np.uint16)) + 
                        green_image.astype(np.uint16) * alpha_output.astype(np.uint16))
        output_frame = output_frame // 255
        output_frame = output_frame.astype(np.uint8)
        control_output = output_frame    
        alpha_output = alpha_output[:,:,0]                
        control_title = "Green Screen Output"
        tab_label = "Green Screen"
        allow_export = False
    elif mask_type == "alpha":
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 
        from models.wan.alpha.utils import render_video, from_BRGA_numpy_to_RGBA_torch
        from shared.utils.utils import convert_tensor_to_image
        _, BGRA_frames =  render_video(following_frames[-1:], [alpha_output])
        RGBA_image = from_BRGA_numpy_to_RGBA_torch(BGRA_frames).squeeze(1)
        control_output = convert_tensor_to_image(RGBA_image)
        alpha_output = alpha_output[:,:,0]                
        control_title = "RGBA Output"
        tab_label = "RGBA"
        allow_export = False


    bbox_info = mask_to_xyxy_box(alpha_output)
    h = alpha_output.shape[0]
    w = alpha_output.shape[1]
    if len(bbox_info) == 0:
        bbox_info = ""
    else:
        bbox_info = [str(int(bbox_info[0]/ w * 100 )), str(int(bbox_info[1]/ h * 100 )),  str(int(bbox_info[2]/ w * 100 )), str(int(bbox_info[3]/ h * 100 )) ]
        bbox_info = ":".join(bbox_info)
    alpha_output = Image.fromarray(alpha_output)
 
    return gr.update(visible=True, selected =0), gr.update(label=tab_label, visible=True), gr.update(visible = foreground_output is not None), foreground_output, control_output, alpha_output, gr.update(visible=foreground_output is not None, label=foreground_title),gr.update(visible=True, label=control_title), gr.update(visible=True, label=alpha_title), gr.update(value=bbox_info, visible= True), gr.update(visible=allow_export), gr.update(visible=allow_export)


# video matting
def video_matting(state, video_state, mask_type, video_input, end_slider, matting_type, new_new_dim, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size):
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update(visible=False)]*6

    # if interactive_state["track_end_number"]:
    #     following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    # else:
    end_slider = max(video_state["select_frame_number"] +1, end_slider)
    following_frames = video_state["origin_images"][video_state["select_frame_number"]: end_slider]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]
    new_dim = video_state.get("new_dim", "")
    if new_new_dim != new_dim:
        gr.Info(f"You have changed the Input / Output Dimensions after loading the Video into Matanyone. The output dimension will be the ones when loading the video ({'original' if len(new_dim) == 0 else new_dim})")
    audio_path = video_state["audio"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
    acquire_GPU(state)
    select_matanyone(state)
    matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
    foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size)
    torch.cuda.empty_cache()    
    release_GPU(state)
    foreground_mat = matting_type == "Foreground"
    alpha_title = "Alpha Mask Video Output"
    alpha_suffix = "_alpha"
    output_frames = []
    new_alpha = []
    BGRA_frames = None
    if mask_type == "" or mask_type == "wangp":
        if not foreground_mat:
            alpha = [255 - frame_alpha for frame_alpha in alpha ]
        output_frames = following_frames
        foreground_title = "Original Video Input"
        foreground_suffix = ""
        allow_export = True
    elif mask_type == "greenscreen":
        green_image = np.zeros_like(following_frames[0], dtype=np.uint8)
        green_image[:, :, 1] = 255          
        for frame_origin, frame_alpha in zip(following_frames, alpha):
            if not foreground_mat:
                frame_alpha = 255 - frame_alpha 

            output_frame = (frame_origin.astype(np.uint16) * (255 - frame_alpha.astype(np.uint16)) + 
                            green_image.astype(np.uint16) * frame_alpha.astype(np.uint16))
            output_frame = output_frame // 255
            output_frame = output_frame.astype(np.uint8)            
            output_frames.append(output_frame)
            new_alpha.append(frame_alpha)
        alpha = new_alpha 
        foreground_title = "Green Screen Output"
        foreground_suffix = "_greenscreen"
        allow_export = False
    elif mask_type == "alpha":
        if not foreground_mat:
            alpha = [255 - frame_alpha for frame_alpha in alpha ]
        from models.wan.alpha.utils import render_video
        output_frames, BGRA_frames =  render_video(following_frames, alpha)
        foreground_title = "Checkboard Output"
        foreground_suffix = "_RGBA"
        allow_export = False

    if not os.path.exists("mask_outputs"):
        os.makedirs("mask_outputs")

    file_name= video_state["video_name"]
    file_name = ".".join(file_name.split(".")[:-1]) 
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
    file_name = f"{file_name}_{time_flag}"
    if len(new_dim) > 0: file_name += "_" + get_dim_file_suffix(new_dim) 
 
    from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, cleanup_temp_audio_files    
    source_audio_tracks, audio_metadata  = extract_audio_tracks(video_input, verbose= offload.default_verboseLevel )
    output_fg_path =  f"./mask_outputs/{file_name}{foreground_suffix}.mp4"
    output_fg_temp_path =  f"./mask_outputs/{file_name}{foreground_suffix}_tmp.mp4"
    if len(source_audio_tracks) == 0:
        foreground_output = save_video(output_frames, output_fg_path , fps=fps, codec_type= video_output_codec)
    else:
        foreground_output_tmp = save_video(output_frames, output_fg_temp_path , fps=fps,  codec_type= video_output_codec)
        combine_video_with_audio_tracks(output_fg_temp_path, source_audio_tracks, output_fg_path, audio_metadata=audio_metadata)
        cleanup_temp_audio_files(source_audio_tracks)
        os.remove(foreground_output_tmp)
        foreground_output = output_fg_path

    alpha_output = save_video(alpha, f"./mask_outputs/{file_name}{alpha_suffix}.mp4", fps=fps, codec_type= video_output_codec)
    if BGRA_frames is not None:
        from models.wan.alpha.utils import write_zip_file
        write_zip_file(f"./mask_outputs/{file_name}{foreground_suffix}.zip", BGRA_frames)
    return foreground_output, alpha_output, gr.update(visible=True, label=foreground_title), gr.update(visible=True, label=alpha_title), gr.update(visible=allow_export), gr.update(visible=allow_export)


def show_outputs():
    return gr.update(visible=True), gr.update(visible=True)

def add_audio_to_video(video_path, audio_path, output_path):
    pass
    # try:
    #     video_input = ffmpeg.input(video_path)
    #     audio_input = ffmpeg.input(audio_path)

    #     _ = (
    #         ffmpeg
    #         .output(video_input, audio_input, output_path, vcodec="copy", acodec="aac")
    #         .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    #     )
    #     return output_path
    # except ffmpeg.Error as e:
    #     print(f"FFmpeg error:\n{e.stderr.decode()}")
    #     return None


def generate_video_from_frames(frames, output_path, fps=30, gray2rgb=False, audio_path=""):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    _, h, w, _ = frames.shape
    if gray2rgb:
        frames = np.repeat(frames, 3, axis=3)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    video_temp_path = output_path.replace(".mp4", "_temp.mp4")
    
    # resize back to ensure input resolution
    imageio.mimwrite(video_temp_path, frames, fps=fps, quality=7, 
                     codec='libx264', ffmpeg_params=["-vf", f"scale={w}:{h}"])
    
    # add audio to video if audio path exists
    if audio_path != "" and os.path.exists(audio_path):
        output_path = add_audio_to_video(video_temp_path, audio_path, output_path)    
        os.remove(video_temp_path)
        return output_path
    else:
        return video_temp_path

# reset all states for a new input

def get_default_states():
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": False,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        }, [[],[]]

def restart():
    return *(get_default_states()), gr.update(interactive=True), gr.update(visible=False), None,  None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False, choices=[], value=[]), "", gr.update(visible=False)

# def load_sam():
#     global model_loaded
#     global model
#     model.samcontroler.sam_controler.model.to(arg_device)

#     global matanyone_model 
#     matanyone_model.to(arg_device)


def select_matanyone(state):
    global matanyone_in_GPU, model_in_GPU 
    if matanyone_model is None: 
        load_unload_models(state, True, True)
    if matanyone_in_GPU: return
    model.samcontroler.sam_controler.model.to("cpu")
    model_in_GPU = False
    torch.cuda.empty_cache()
    matanyone_model.to(arg_device)
    matanyone_in_GPU = True

def select_SAM(state):
    global matanyone_in_GPU, model_in_GPU 
    if matanyone_model is None: 
        load_unload_models(state, True, True)
    if model_in_GPU: return
    matanyone_model.to("cpu")
    matanyone_in_GPU = False
    torch.cuda.empty_cache()
    model.samcontroler.sam_controler.model.to(arg_device)
    model_in_GPU = True

load_in_progress = False

def load_unload_models(state = None, selected = True, force = False):
    global model_loaded, load_in_progress
    global model
    global matanyone_model, matanyone_processor, matanyone_in_GPU , model_in_GPU, bfloat16_supported

    if selected:
        if (not force) and any_GPU_process_running(state, "matanyone"):
            return

        if load_in_progress:
            while model == None:
                time.sleep(1)
            return
        # print("Matanyone Tab Selected")
        if model_loaded or load_in_progress:
            pass
            # load_sam()
        else:
            load_in_progress = True
            # args, defined in track_anything.py
            sam_checkpoint_url_dict = {
                'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
            # os.path.join('.')


            # sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[arg_sam_model_type], ".")
            sam_checkpoint = None

            transfer_stream = torch.cuda.Stream()
            with torch.cuda.stream(transfer_stream):
                # initialize sams
                major, minor = torch.cuda.get_device_capability(arg_device)
                if  major < 8:
                    bfloat16_supported = False
                else:
                    bfloat16_supported = True

                model = MaskGenerator(sam_checkpoint, "cpu")
                model.samcontroler.sam_controler.model.to("cpu").to(torch.bfloat16).to(arg_device)
                model_in_GPU = True
                from .matanyone.model.matanyone import MatAnyone
                # matanyone_model = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
                matanyone_model = MatAnyone.from_pretrained(fl.locate_folder("mask"))
                # pipe ={"mat" : matanyone_model, "sam" :model.samcontroler.sam_controler.model }
                # offload.profile(pipe)
                matanyone_model = matanyone_model.to("cpu").eval()
                matanyone_in_GPU = False
                matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
            model_loaded  = True
            load_in_progress = False

    else:
        # print("Matanyone Tab UnSelected")
        import gc
        # model.samcontroler.sam_controler.model.to("cpu")
        # matanyone_model.to("cpu")
        model = matanyone_model = matanyone_processor = None
        matanyone_in_GPU = model_in_GPU = False
        gc.collect()
        torch.cuda.empty_cache()
        model_loaded = False


def get_vmc_event_handler():
    return load_unload_models


def export_image(state, image_output):
    ui_settings = get_current_model_settings(state)
    image_refs = ui_settings.get("image_refs", None)
    if image_refs == None:
        image_refs =[]
    image_refs.append( image_output)
    ui_settings["image_refs"] = image_refs 
    gr.Info("Masked Image transferred to Current Image Generator")
    return time.time()

def export_image_mask(state, image_input, image_mask):
    ui_settings = get_current_model_settings(state)
    ui_settings["image_guide"] = image_input
    ui_settings["image_mask"] = image_mask

    gr.Info("Input Image & Mask transferred to Current Image Generator")
    return time.time()


def export_to_current_video_engine(state, foreground_video_output, alpha_video_output):
    ui_settings = get_current_model_settings(state)
    ui_settings["video_guide"] = foreground_video_output
    ui_settings["video_mask"] = alpha_video_output

    gr.Info("Original Video and Full Mask have been transferred")
    return time.time()


def teleport_to_video_tab(tab_state, state):
    return PlugIn.goto_video_tab(state)


def display(tabs, tab_state, state, refresh_form_trigger, server_config, get_current_model_settings_fn): #,  vace_video_input, vace_image_input, vace_video_mask, vace_image_mask, vace_image_refs):
    # my_tab.select(fn=load_unload_models, inputs=[], outputs=[])
    global image_output_codec, video_output_codec, get_current_model_settings
    get_current_model_settings = get_current_model_settings_fn

    image_output_codec = server_config.get("image_output_codec", None)
    video_output_codec = server_config.get("video_output_codec", None)

    media_url = "https://github.com/pq-yang/MatAnyone/releases/download/media/"

    click_brush_js = """
    () => {
        setTimeout(() => {
            const brushButton = document.querySelector('button[aria-label="Brush"]');
            if (brushButton) {
                brushButton.click();
                console.log('Brush button clicked');
            } else {
                console.log('Brush button not found');
            }
        }, 1000);
    }    """

    # download assets

    gr.Markdown("<B>Mast Edition is provided by MatAnyone, VRAM optimizations & Extended Masks by DeepBeepMeep</B>")
    gr.Markdown("If you have some trouble creating the perfect mask, be aware of these tips:")
    gr.Markdown("- Using the Matanyone Settings you can also define Negative Point Prompts to remove parts of the current selection.")
    gr.Markdown("- Sometime it is very hard to fit everything you want in a single mask, it may be much easier to combine multiple independent sub Masks before producing the Matting : each sub Mask is created by selecting an  area of an image and by clicking the Add Mask button. Sub masks can then be enabled / disabled in the Matanyone settings.")
    gr.Markdown("The Mask Generation time and the VRAM consumed are proportional to the number of frames and the resolution. So if relevant, you may reduce the number of frames in the Matanyone Settings. You will need for the moment to resize yourself the video if needed.")
    
    with gr.Column( visible=True):
        with gr.Row():
            with gr.Accordion("Video Tutorial (click to expand)", open=False, elem_classes="custom-bg"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Case 1: Single Target")
                        gr.Video(value="preprocessing/matanyone/tutorial_single_target.mp4", elem_classes="video")

                    with gr.Column():
                        gr.Markdown("### Case 2: Multiple Targets")
                        gr.Video(value="preprocessing/matanyone/tutorial_multi_targets.mp4", elem_classes="video")

        with gr.Row():
            new_dim= gr.Dropdown(
                choices=[
                    ("Original Dimensions", ""),
                    ("1080p - Pixels Budgets", "1080p - Pixels Budget"),
                    ("720p - Pixels Budgets", "720p - Pixels Budget"),
                    ("480p - Pixels Budgets", "480p - Pixels Budget"),                     
                    ("1080p - Outer Frame", "1080p - Outer Frame"),
                    ("720p - Outer Frame", "720p - Outer Frame"),
                    ("480p - Outer Frame", "480p - Outer Frame"),                     
                ],   label = "Resize Input / Output", value = ""
            ) 

            mask_type= gr.Dropdown(
                choices=[
                    ("Grey with Alpha (used by WanGP)", "wangp"),
                    ("Green Screen", "greenscreen"),
                    ("RGB With Alpha Channel (local Zip file)", "alpha")
                ],   label = "Mask Type", value = "wangp"
            ) 

            matting_type = gr.Radio(
                choices=["Foreground", "Background"],
                value="Foreground",
                label="Type of Video Matting to Generate",
                scale=1)

            
        with gr.Row(visible=False):
            dummy = gr.Text()        

        with gr.Tabs():
            with gr.TabItem("Video"):

                click_state = gr.State([[],[]])

                interactive_state = gr.State({
                    "inference_times": 0,
                    "negative_click_times" : 0,
                    "positive_click_times": 0,
                    "mask_save": arg_mask_save,
                    "multi_mask": {
                        "mask_names": [],
                        "masks": []
                    },
                    "track_end_number": None,
                    }
                )

                video_state = gr.State(
                    {
                    "user_name": "",
                    "video_name": "",
                    "origin_images": None,
                    "painted_images": None,
                    "masks": None,
                    "inpaint_masks": None,
                    "logits": None,
                    "select_frame_number": 0,
                    "fps": 16,
                    "audio": "",
                    }
                )

                with gr.Column( visible=True):
                    with gr.Row():
                        with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                            with gr.Row():
                                erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Erosion on the added mask",
                                                        interactive=True)
                                dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Dilation on the added mask",
                                                        interactive=True)

                            with gr.Row():
                                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Start Frame", info="Choose the start frame for target assignment and video matting", visible=False)
                                end_selection_slider = gr.Slider(minimum=1, maximum=300, step=1, value=81, label="Last Frame to Process", info="Last Frame to Process", visible=False)

                                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="End frame", visible=False)
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    info="Click to add positive or negative point for target mask",
                                    interactive=True,
                                    visible=False,
                                    min_width=100,
                                    scale=1)
                                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False, scale=2, allow_custom_value=True)

                    # input video
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2): 
                            gr.Markdown("## Step1: Upload video")
                        with gr.Column(scale=2): 
                            step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):      
                            video_input = gr.Video(label="Input Video", elem_classes="video")
                            extract_frames_button = gr.Button(value="Load Video", interactive=True, elem_classes="new_button")
                        with gr.Column(scale=2):
                            video_info = gr.Textbox(label="Video Info", visible=False)
                            template_frame = gr.Image(label="Start Frame", type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                            with gr.Row():
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False,  min_width=100)
                                add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, min_width=100)
                                remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False,  min_width=100) # no use
                                matting_button = gr.Button(value="Generate Video Matting", interactive=True, visible=False,  min_width=100)
                            with gr.Row():
                                gr.Markdown("")            

                    # output video
                    with gr.Column() as output_row: #equal_height=True
                        with gr.Row():
                            with gr.Column(scale=2):
                                foreground_video_output = gr.Video(label="Original Video Input", visible=False, elem_classes="video")
                                foreground_output_button = gr.Button(value="Black & White Video Output", visible=False, elem_classes="new_button")
                            with gr.Column(scale=2):
                                alpha_video_output = gr.Video(label="Mask Video Output", visible=False, elem_classes="video")
                                export_image_mask_btn = gr.Button(value="Alpha Mask Output", visible=False, elem_classes="new_button")
                        with gr.Row():
                            with gr.Row(visible= False):
                                export_to_vace_video_14B_btn = gr.Button("Export to current Video Input Video For Inpainting", visible= False)
                            with gr.Row(visible= True):
                                export_to_current_video_engine_btn = gr.Button("Export to Control Video Input and Video Mask Input", visible= False)
                                    
                export_to_current_video_engine_btn.click(  fn=export_to_current_video_engine, inputs= [state, foreground_video_output, alpha_video_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_video_tab, inputs= [tab_state, state], outputs= [tabs])


                # first step: get the video information     
                extract_frames_button.click(
                    fn=get_frames_from_video,
                    inputs=[
                        state, video_input, video_state, new_dim
                    ],
                    outputs=[video_state, extract_frames_button, video_info, template_frame,
                            image_selection_slider, end_selection_slider,  track_pause_number_slider, point_prompt, dummy, clear_button_click, add_mask_button, matting_button, template_frame,
                            foreground_video_output, alpha_video_output, foreground_output_button, export_image_mask_btn, mask_dropdown, step2_title]
                )   

                # second step: select images from slider
                image_selection_slider.release(fn=select_video_template, 
                                            inputs=[image_selection_slider, video_state, interactive_state], 
                                            outputs=[template_frame, video_state, interactive_state], api_name="select_image")
                track_pause_number_slider.release(fn=get_end_number, 
                                            inputs=[track_pause_number_slider, video_state, interactive_state], 
                                            outputs=[template_frame, interactive_state], api_name="end_image")
                
                # click select image to get mask using sam
                template_frame.select(
                    fn=sam_refine,
                    inputs=[state, video_state, point_prompt, click_state, interactive_state],
                    outputs=[template_frame, video_state, interactive_state]
                )

                # add different mask
                add_mask_button.click(
                    fn=add_multi_mask,
                    inputs=[video_state, interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame, click_state]
                )

                remove_mask_button.click(
                    fn=remove_multi_mask,
                    inputs=[interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown]
                )

                # video matting
                matting_button.click(
                    fn=show_outputs,
                    inputs=[],
                    outputs=[foreground_video_output, alpha_video_output]).then(
                    fn=video_matting,
                    inputs=[state, video_state, mask_type, video_input, end_selection_slider, matting_type, new_dim, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size],
                    outputs=[foreground_video_output, alpha_video_output,foreground_video_output, alpha_video_output, export_to_vace_video_14B_btn, export_to_current_video_engine_btn]
                )

                # click to get mask
                mask_dropdown.change(
                    fn=show_mask,
                    inputs=[video_state, interactive_state, mask_dropdown],
                    outputs=[template_frame]
                )
                
                # clear input
                video_input.change(
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        video_state,
                        interactive_state,
                        click_state, 
                        extract_frames_button, dummy,
                        foreground_video_output, dummy, alpha_video_output,
                        template_frame,
                        image_selection_slider, end_selection_slider, track_pause_number_slider,point_prompt, export_to_vace_video_14B_btn, export_to_current_video_engine_btn, dummy, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, export_image_mask_btn, mask_dropdown, video_info, step2_title
                    ],
                    queue=False,
                    show_progress=False)
                
                video_input.clear(
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        video_state,
                        interactive_state,
                        click_state,
                        extract_frames_button, dummy,
                        foreground_video_output, dummy, alpha_video_output,
                        template_frame,
                        image_selection_slider , end_selection_slider, track_pause_number_slider,point_prompt, export_to_vace_video_14B_btn, export_to_current_video_engine_btn, dummy, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, export_image_mask_btn, mask_dropdown, video_info, step2_title
                    ],
                    queue=False,
                    show_progress=False)
                
                # points clear
                clear_button_click.click(
                    fn = clear_click,
                    inputs = [video_state, click_state,],
                    outputs = [template_frame,click_state],
                )



            with gr.TabItem("Image"):
                click_state = gr.State([[],[]])

                interactive_state = gr.State({
                    "inference_times": 0,
                    "negative_click_times" : 0,
                    "positive_click_times": 0,
                    "mask_save": False,
                    "multi_mask": {
                        "mask_names": [],
                        "masks": []
                    },
                    "track_end_number": None,
                    }
                )

                image_state = gr.State(
                    {
                    "user_name": "",
                    "image_name": "",
                    "origin_images": None,
                    "painted_images": None,
                    "masks": None,
                    "inpaint_masks": None,
                    "logits": None,
                    "select_frame_number": 0,
                    "fps": 30
                    }
                )

                with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                    with gr.Row():
                        with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                            with gr.Row():
                                erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Erosion on the added mask",
                                                        interactive=True)
                                dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Dilation on the added mask",
                                                        interactive=True)
                                
                            with gr.Row():
                                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Num of Refinement Iterations", info="More iterations → More details & More time", visible=False)
                                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    info="Click to add positive or negative point for target mask",
                                    interactive=True,
                                    visible=False,
                                    min_width=100,
                                    scale=1)
                                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)
                

                with gr.Column():
                    # input image
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2): 
                            gr.Markdown("## Step1: Upload image")
                        with gr.Column(scale=2): 
                            step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):      
                            image_input = gr.Image(label="Input Image", elem_classes="image")
                            extract_frames_button = gr.Button(value="Load Image", interactive=True, elem_classes="new_button")
                        with gr.Column(scale=2):
                            image_info = gr.Textbox(label="Image Info", visible=False)
                            template_frame = gr.Image(type="pil", label="Start Frame", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                            with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                matting_button = gr.Button(value="Image Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)

                    # output image
                    with gr.Tabs(visible = False) as image_tabs:
                        with gr.TabItem("Control Image & Mask", visible = False) as image_first_tab:
                            with gr.Row(equal_height=True):
                                control_image_output = gr.Image(type="pil", label="Control Image", visible=False, elem_classes="image")
                                alpha_image_output = gr.Image(type="pil", label="Mask", visible=False, elem_classes="image")
                            with gr.Row():
                                export_image_mask_btn = gr.Button(value="Set to Control Image & Mask", visible=False, elem_classes="new_button")
                        with gr.TabItem("Reference Image", visible = False) as image_second_tab:
                            with gr.Row():
                                foreground_image_output = gr.Image(type="pil", label="Foreground Output", visible=False, elem_classes="image")
                            with gr.Row():
                                export_image_btn = gr.Button(value="Add to current Reference Images", visible=False, elem_classes="new_button")

                    with gr.Row(equal_height=True):
                        bbox_info = gr.Text(label ="Mask BBox Info (Left:Top:Right:Bottom)", visible = False, interactive= False)

                export_image_btn.click(  fn=export_image, inputs= [state, foreground_image_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_video_tab, inputs= [tab_state, state], outputs= [tabs])
                export_image_mask_btn.click(  fn=export_image_mask, inputs= [state, control_image_output, alpha_image_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_video_tab, inputs= [tab_state, state], outputs= [tabs]).then(fn=None, inputs=None, outputs=None, js=click_brush_js)

                # first step: get the image information 
                extract_frames_button.click(
                    fn=get_frames_from_image,
                    inputs=[
                        state, image_input, image_state, new_dim
                    ],
                    outputs=[image_state, extract_frames_button, image_info, template_frame,
                            image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, add_mask_button, matting_button, template_frame,
                            foreground_image_output, alpha_image_output, control_image_output, image_tabs, bbox_info, export_image_btn, export_image_mask_btn, mask_dropdown, step2_title]
                )   

                # points clear
                clear_button_click.click(
                    fn = clear_click,
                    inputs = [image_state, click_state,],
                    outputs = [template_frame,click_state],
                )


                # second step: select images from slider
                image_selection_slider.release(fn=select_image_template, 
                                            inputs=[image_selection_slider, image_state, interactive_state], 
                                            outputs=[template_frame, image_state, interactive_state], api_name="select_image")
                track_pause_number_slider.release(fn=get_end_number, 
                                            inputs=[track_pause_number_slider, image_state, interactive_state], 
                                            outputs=[template_frame, interactive_state], api_name="end_image")
                
                # click select image to get mask using sam
                template_frame.select(
                    fn=sam_refine,
                    inputs=[state, image_state, point_prompt, click_state, interactive_state],
                    outputs=[template_frame, image_state, interactive_state]
                )

                # add different mask
                add_mask_button.click(
                    fn=add_multi_mask,
                    inputs=[image_state, interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame, click_state]
                )

                remove_mask_button.click(
                    fn=remove_multi_mask,
                    inputs=[interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown]
                )

                # image matting
                matting_button.click(
                    fn=image_matting,
                    inputs=[state, image_state, interactive_state, mask_type, matting_type, new_dim, mask_dropdown, erode_kernel_size, dilate_kernel_size, image_selection_slider],
                    outputs=[image_tabs, image_first_tab, image_second_tab, foreground_image_output, control_image_output, alpha_image_output, foreground_image_output, control_image_output, alpha_image_output, bbox_info, export_image_btn, export_image_mask_btn]
                )

                nada = gr.State({})
                # clear input
                gr.on(
                    triggers=[image_input.clear], #image_input.change,
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        image_state,
                        interactive_state,
                        click_state,
                        extract_frames_button, image_tabs,
                        foreground_image_output, control_image_output, alpha_image_output,
                        template_frame,
                        image_selection_slider, image_selection_slider, track_pause_number_slider,point_prompt, export_image_btn, export_image_mask_btn, bbox_info, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, export_image_btn, export_image_mask_btn, mask_dropdown, nada, step2_title
                    ],
                    queue=False,
                    show_progress=False)
                
