import os
import time
import threading
import argparse
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
import random
import json
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video
from wan.modules.attention import get_attention_modes, get_supported_attention_modes
import torch
import gc
import traceback
import math
import asyncio
from wan.utils import prompt_parser
import base64
import io
from PIL import Image
PROMPT_VARS_MAX = 10

target_mmgp_version = "3.3.4"
from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
queue = []
lock = threading.Lock()
current_task_id = None
task_id = 0
progress_tracker = {}
tracker_lock = threading.Lock()
file_list = []
last_model_type = None

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None
    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None

def runner():
    global current_task_id
    while True:
        with lock:
            for item in queue:
                task_id_runner = item['id']
                with tracker_lock:
                    progress = progress_tracker.get(task_id_runner, {})
                
                if item['state'] == "Processing":
                    current_step = progress.get('current_step', 0)
                    total_steps = progress.get('total_steps', 0)
                    elapsed = time.time() - progress.get('start_time', time.time())
                    status = progress.get('status', "")
                    repeats = progress.get("repeats", "0/0")
                    item.update({
                        'progress': f"{((current_step/total_steps)*100 if total_steps > 0 else 0):.1f}%",
                        'steps': f"{current_step}/{total_steps}",
                        'time': format_time(elapsed),
                        'repeats': f"{repeats}",
                        'status': f"{status}"
                    })
            if not any(item['state'] == "Processing" for item in queue):
                for item in queue:
                    if item['state'] == "Queued":
                        item['status'] = "Processing"
                        item['state'] = "Processing"
                        current_task_id = item['id']
                        threading.Thread(target=process_task, args=(item,)).start()
                        break
        time.sleep(1)

def process_prompt_and_add_tasks(
    prompt,
    negative_prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    multi_images_gen_type,
    tea_cache,
    tea_cache_start_step_perc,
    loras_choices,
    loras_mult_choices,
    image_prompt_type,
    image_to_continue,
    image_to_end,
    video_to_continue,
    max_frames,
    RIFLEx_setting,
    slg_switch,
    slg_layers,
    slg_start,
    slg_end,
    cfg_star_switch,
    cfg_zero_step,
    state_arg,
    image2video
):

    if state_arg.get("validate_success",0) != 1:
        print("Validation failed, not adding tasks.")
        return
    if len(prompt) ==0:
        return
    prompt, errors = prompt_parser.process_template(prompt)
    if len(errors) > 0:
        print("Error processing prompt template: " + errors)
        return
    prompts = prompt.replace("\r", "").split("\n")
    prompts = [prompt.strip() for prompt in prompts if len(prompt.strip())>0 and not prompt.startswith("#")]
    if len(prompts) ==0:
        return

    for single_prompt in prompts:
        task_params = (
            single_prompt,
            negative_prompt,
            resolution,
            video_length,
            seed,
            num_inference_steps,
            guidance_scale,
            flow_shift,
            embedded_guidance_scale,
            repeat_generation,
            multi_images_gen_type,
            tea_cache,
            tea_cache_start_step_perc,
            loras_choices,
            loras_mult_choices,
            image_prompt_type,
            image_to_continue,
            image_to_end,
            video_to_continue,
            max_frames,
            RIFLEx_setting,
            slg_switch,
            slg_layers,
            slg_start,
            slg_end,
            cfg_star_switch,
            cfg_zero_step,
            state_arg,
            image2video
        )
        add_video_task(*task_params)
    return update_queue_data()

def process_task(task):
    try:
        task_id, *params = task['params']
        generate_video(task_id, *params)
    finally:
        with lock:
            queue[:] = [item for item in queue if item['id'] != task['id']]
        with tracker_lock:
            if task['id'] in progress_tracker:
                del progress_tracker[task['id']]

def add_video_task(*params):
    global task_id
    with lock:
        task_id += 1
        current_task_id = task_id
        start_image_data = params[16] if len(params) > 16 else None
        end_image_data = params[17] if len(params) > 17 else None

        queue.append({
            "id": current_task_id,
            "params": (current_task_id,) + params,
            "state": "Queued",
            "status": "Queued",
            "repeats": "0/0",
            "progress": "0.0%",
            "steps": f"0/{params[5]}",
            "time": "--",
            "prompt": params[0],
            "start_image_data": start_image_data,
            "end_image_data": end_image_data
        })
    return update_queue_data()

def move_up(selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data()
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        if idx > 0:
            queue[idx], queue[idx-1] = queue[idx-1], queue[idx]
    return update_queue_data()

def move_down(selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data()
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        if idx < len(queue)-1:
            queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
    return update_queue_data()

def remove_task(selected_indices):
    if not selected_indices or len(selected_indices) == 0:
        return update_queue_data()
    idx = selected_indices[0]
    if isinstance(idx, list):
        idx = idx[0]
    idx = int(idx)
    with lock:
        if idx < len(queue):
            if idx == 0:
                wan_model._interrupt = True
            del queue[idx]
    return update_queue_data()

def update_queue_data():
    with lock:
        data = []
        for item in queue:
            truncated_prompt = (item['prompt'][:97] + '...') if len(item['prompt']) > 100 else item['prompt']
            full_prompt = item['prompt'].replace('"', '&quot;')
            prompt_cell = f'<span title="{full_prompt}">{truncated_prompt}</span>'
            start_img_uri = pil_to_base64_uri(item.get('start_image_data'), format="jpeg", quality=70)
            end_img_uri = pil_to_base64_uri(item.get('end_image_data'), format="jpeg", quality=70)
            thumbnail_size = "50px"
            start_img_md = ""
            end_img_md = ""
            if start_img_uri:
                start_img_md = f'<img src="{start_img_uri}" alt="Start" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
            if end_img_uri:
                end_img_md = f'<img src="{end_img_uri}" alt="End" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" />'
            data.append([
                item.get('status', "Starting"),
                item.get('repeats', "0/0"),
                item.get('progress', "0.0%"),
                item.get('steps', ''),
                item.get('time', '--'),
                prompt_cell,
                start_img_md,
                end_img_md,
                "↑",
                "↓",
                "✖"
            ])
        return data

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")

    parser.add_argument(
        "--quantize-transformer",
        action="store_true",
        help="On the fly 'transformer' quantization"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shared URL to access webserver remotely"
    )

    parser.add_argument(
        "--lock-config",
        action="store_true",
        help="Prevent modifying the configuration from the web interface"
    )

    parser.add_argument(
        "--preload",
        type=str,
        default="0",
        help="Megabytes of the diffusion model to preload in VRAM"
    )

    parser.add_argument(
        "--multiple-images",
        action="store_true",
        help="Allow inputting multiple images with image to video"
    )


    parser.add_argument(
        "--lora-dir-i2v",
        type=str,
        default="",
        help="Path to a directory that contains Loras for i2v"
    )


    parser.add_argument(
        "--lora-dir",
        type=str,
        default="", 
        help="Path to a directory that contains Loras"
    )

    parser.add_argument(
        "--check-loras",
        action="store_true",
        help="Filter Loras that are not valid"
    )


    parser.add_argument(
        "--lora-preset",
        type=str,
        default="",
        help="Lora preset to preload"
    )

    parser.add_argument(
        "--i2v-settings",
        type=str,
        default="i2v_settings.json",
        help="Path to settings file for i2v"
    )

    parser.add_argument(
        "--t2v-settings",
        type=str,
        default="t2v_settings.json",
        help="Path to settings file for t2v"
    )

    # parser.add_argument(
    #     "--lora-preset-i2v",
    #     type=str,
    #     default="",
    #     help="Lora preset to preload for i2v"
    # )

    parser.add_argument(
        "--profile",
        type=str,
        default=-1,
        help="Profile No"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default=1,
        help="Verbose level"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="default denoising steps"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="default number of frames"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="default generation seed"
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Access advanced options by default"
    )

    parser.add_argument(
        "--server-port",
        type=str,
        default=0,
        help="Server port"
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Server name"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="Default GPU Device"
    )

    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open browser"
    )

    parser.add_argument(
        "--t2v",
        action="store_true",
        help="text to video mode"
    )

    parser.add_argument(
        "--i2v",
        action="store_true",
        help="image to video mode"
    )

    parser.add_argument(
        "--t2v-14B",
        action="store_true",
        help="text to video mode 14B model"
    )

    parser.add_argument(
        "--t2v-1-3B",
        action="store_true",
        help="text to video mode 1.3B model"
    )
    parser.add_argument(
        "--i2v-1-3B",
        action="store_true",
        help="Fun InP image to video mode 1.3B model"
    )

    parser.add_argument(
        "--i2v-14B",
        action="store_true",
        help="image to video mode 14B model"
    )


    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable pytorch compilation"
    )

    parser.add_argument(
        "--listen",
        action="store_true",
        help="Server accessible on local network"
    )

    # parser.add_argument(
    #     "--fast",
    #     action="store_true",
    #     help="use Fast model"
    # )

    # parser.add_argument(
    #     "--fastest",
    #     action="store_true",
    #     help="activate the best config"
    # )

    parser.add_argument(
    "--attention",
    type=str,
    default="",
    help="attention mode"
    )

    parser.add_argument(
    "--vae-config",
    type=str,
    default="",
    help="vae config mode"
    )    

    args = parser.parse_args()

    return args

def get_lora_dir(i2v):
    lora_dir =args.lora_dir
    if i2v and len(lora_dir)==0:
        lora_dir =args.lora_dir_i2v
    if len(lora_dir) > 0:
        return lora_dir

    root_lora_dir = "loras_i2v" if i2v else "loras"

    if  "1.3B" in (transformer_filename_i2v if i2v else transformer_filename_t2v) :
        lora_dir_1_3B = os.path.join(root_lora_dir, "1.3B")
        if os.path.isdir(lora_dir_1_3B ):
            return lora_dir_1_3B
    else:
        lora_dir_14B = os.path.join(root_lora_dir, "14B")
        if os.path.isdir(lora_dir_14B ):
            return lora_dir_14B
    return root_lora_dir    

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
args = _parse_args()
args.flow_reverse = True
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

preload =int(args.preload)
force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer
check_loras = args.check_loras ==1
advanced = args.advanced

transformer_choices_t2v=["ckpts/wan2.1_text2video_1.3B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors"]   
transformer_choices_i2v=["ckpts/wan2.1_image2video_480p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_480p_14B_quanto_int8.safetensors", "ckpts/wan2.1_image2video_720p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_720p_14B_quanto_int8.safetensors", "ckpts/wan2.1_Fun_InP_1.3B_bf16.safetensors", "ckpts/wan2.1_Fun_InP_14B_bf16.safetensors", "ckpts/wan2.1_Fun_InP_14B_quanto_int8.safetensors", ]
text_encoder_choices = ["ckpts/models_t5_umt5-xxl-enc-bf16.safetensors", "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors"]

server_config_filename = "gradio_config.json"

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "auto",  
                     "transformer_filename": transformer_choices_t2v[0], 
                     "transformer_filename_i2v": transformer_choices_i2v[1],  
                     "text_encoder_filename" : text_encoder_choices[1],
                     "save_path": os.path.join(os.getcwd(), "gradio_outputs"),
                     "compile" : "",
                     "metadata_type": "metadata",
                     "default_ui": "t2v",
                     "boost" : 1,
                     "clear_file_list" : 0,
                     "vae_config": 0,
                     "profile" : profile_type.LowRAM_LowVRAM,
                     "reload_model": 2 }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)

def get_settings_file_name(i2v):
    return args.i2v_settings if i2v else args.t2v_settings

def get_default_settings(filename, i2v):
    def  get_default_prompt(i2v):
        if i2v:
            return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        else:
            return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."

    defaults_filename = get_settings_file_name(i2v)
    if not Path(defaults_filename).is_file():
        ui_defaults = {
            "prompts": get_default_prompt(i2v),
            "resolution": "832x480",
            "video_length": 81,
            "image_prompt_type" : 0, 
            "num_inference_steps": 30,
            "seed": -1,
            "repeat_generation": 1,
            "multi_images_gen_type": 0,        
            "guidance_scale": 5.0,
            "flow_shift": get_default_flow(filename, i2v),
            "negative_prompt": "",
            "activated_loras": [],
            "loras_multipliers": "",
            "tea_cache": 0.0,
            "tea_cache_start_step_perc": 0,
            "RIFLEx_setting": 0,
            "slg_switch": 0,
            "slg_layers": [9],
            "slg_start_perc": 10,
            "slg_end_perc": 90
        }
        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)

    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults

transformer_filename_t2v = server_config["transformer_filename"]
transformer_filename_i2v = server_config.get("transformer_filename_i2v", transformer_choices_i2v[1]) 

text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")
boost = server_config.get("boost", 1)
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
default_ui = server_config.get("default_ui", "t2v") 
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "gradio_outputs"))
use_image2video = default_ui != "t2v"
if args.t2v:
    use_image2video = False
if args.i2v:
    use_image2video = True
if args.t2v_14B:
    use_image2video = False
    if not "14B" in transformer_filename_t2v: 
        transformer_filename_t2v = transformer_choices_t2v[2]
    lock_ui_transformer = False

if args.i2v_14B:
    use_image2video = True
    if not "14B" in transformer_filename_i2v: 
        transformer_filename_i2v = transformer_choices_t2v[3]
    lock_ui_transformer = False

if args.t2v_1_3B:
    transformer_filename_t2v = transformer_choices_t2v[0]
    use_image2video = False
    lock_ui_transformer = False

if args.i2v_1_3B:
    transformer_filename_i2v = transformer_choices_i2v[4]
    use_image2video = True
    lock_ui_transformer = False

only_allow_edit_in_advanced = False
lora_preselected_preset = args.lora_preset
lora_preselected_preset_for_i2v = use_image2video
# if args.fast : #or args.fastest
#     transformer_filename_t2v = transformer_choices_t2v[2]
#     attention_mode="sage2" if "sage2" in attention_modes_supported else "sage"
#     default_tea_cache = 0.15
#     lock_ui_attention = True
#     lock_ui_transformer = True

if  args.compile: #args.fastest or
    compile="transformer"
    lock_ui_compile = True

model_filename = ""
lora_model_filename = ""
#attention_mode="sage"
#attention_mode="sage2"
#attention_mode="flash"
#attention_mode="sdpa"
#attention_mode="xformers"
# compile = "transformer"

def preprocess_loras(sd):
    if not use_image2video:
        return sd
    
    new_sd = {}
    first = next(iter(sd), None)
    if first == None:
        return sd
    if  not first.startswith("lora_unet_"):
        return sd
    print("Converting Lora Safetensors format to Lora Diffusers format")
    alphas = {}
    repl_list = ["cross_attn", "self_attn", "ffn"]
    src_list = ["_" + k + "_" for k in repl_list]
    tgt_list = ["." + k + "." for k in repl_list]

    for k,v in sd.items():
        k = k.replace("lora_unet_blocks_","diffusion_model.blocks.")

        for s,t in zip(src_list, tgt_list):
            k = k.replace(s,t)

        k = k.replace("lora_up","lora_B")
        k = k.replace("lora_down","lora_A")

        if "alpha" in k:
            alphas[k] = v
        else:
            new_sd[k] = v

    new_alphas = {}
    for k,v in new_sd.items():
        if "lora_B" in k:
            dim = v.shape[1]
        elif "lora_A" in k:
            dim = v.shape[0]
        else:
            continue
        alpha_key = k[:-len("lora_X.weight")] +"alpha"
        if alpha_key in alphas:
            scale = alphas[alpha_key] / dim
            new_alphas[alpha_key] = scale
        else:
            print(f"Lora alpha'{alpha_key}' is missing")
    new_sd.update(new_alphas)
    return new_sd


def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/Wan2.1" 
    sourceFolderList = ["xlm-roberta-large", "",  ]
    fileList = [ [], ["Wan2.1_VAE_bf16.safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors" ] + computeList(text_encoder_filename) + computeList(transformer_filename) ]   
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:     
                if len(sourceFolder) > 0: 
                    if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if not os.path.isfile(targetRoot + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)


offload.default_verboseLevel = verbose_level
to_remove = ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "Wan2.1_VAE.pth"]
for file_name in to_remove:
    file_name = os.path.join("ckpts",file_name)
    if os.path.isfile(file_name):
        try:
            os.remove(file_name)
        except:
            pass

download_models(transformer_filename_i2v if use_image2video else transformer_filename_t2v, text_encoder_filename) 

def sanitize_file_name(file_name, rep =""):
    return file_name.replace("/",rep).replace("\\",rep).replace(":",rep).replace("|",rep).replace("?",rep).replace("<",rep).replace(">",rep).replace("\"",rep) 

def extract_preset(lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(use_image2video)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:
        missing_loras = []

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices_files = lset["loras"]
        for lora_file in loras_choices_files:
            choice = os.path.join(lora_dir, lora_file)
            if choice not in loras:
                missing_loras.append(lora_file)
            else:
                loras_choice_no = loras.index(choice)
                loras_choices.append(str(loras_choice_no))

        if len(missing_loras) > 0:
            error = f"Unable to apply Lora preset '{lset_name} because the following Loras files are missing or invalid: {missing_loras}"
        
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


    
def setup_loras(i2v, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path

    lora_dir = get_lora_dir(i2v)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        import glob
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets =  glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=preprocess_loras, split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset


def load_t2v_model(model_filename, value):

    cfg = WAN_CONFIGS['t2v-14B']
    # cfg = WAN_CONFIGS['t2v-1.3B']    
    print(f"Loading '{model_filename}' model...")

    wan_model = wan.WanT2V(
        config=cfg,
        checkpoint_dir="ckpts",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        model_filename=model_filename,
        text_encoder_filename= text_encoder_filename
    )

    pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "vae": wan_model.vae.model } 

    return wan_model, pipe

def load_i2v_model(model_filename, value):

    print(f"Loading '{model_filename}' model...")

    if value == '720P':
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= True,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )            
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #

    elif value == '480P':
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= False,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename

        )
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #
    else:
        raise Exception("Model i2v {value} not supported")
    return wan_model, pipe

def model_needed(i2v):
    return transformer_filename_i2v if i2v else transformer_filename_t2v

def load_models(i2v):
    global model_filename
    model_filename = model_needed(i2v)
    download_models(model_filename, text_encoder_filename)
    if i2v:
        res720P = "720p" in model_filename
        wan_model, pipe = load_i2v_model(model_filename, "720P" if res720P else "480P")
    else:
        wan_model, pipe = load_t2v_model(model_filename, "")
    kwargs = { "extraModelsToQuantize": None}
    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100, "*" : 1000 }
        # if profile == 4:
        #     kwargs["partialPinning"] = True
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer, loras = "transformer", **kwargs)  
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)

    return wan_model, offloadobj, pipe["transformer"] 

wan_model, offloadobj, transformer = load_models(use_image2video)
if check_loras:
    setup_loras(use_image2video, transformer,  get_lora_dir(use_image2video), "", None)
    exit()
del transformer

gen_in_progress = False

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def get_default_flow(filename, i2v):
    return 7.0 if "480p" in filename and i2v else 5.0 


def get_model_name(model_filename):
    if "Fun" in model_filename:
        model_name = "Fun InP image2video"
        model_name += " 14B" if "14B" in model_filename else " 1.3B"
    elif "image" in model_filename:
        model_name = "Wan2.1 image2video"
        model_name += " 720p" if "720p" in model_filename else " 480p"
    else:
        model_name = "Wan2.1 text2video"
        model_name += " 14B" if "14B" in model_filename else " 1.3B"

    return model_name

def generate_header(model_filename, compile, attention_mode):
    
    header = "<div class='title-with-lines'><div class=line></div><h2>"
    
    model_name = get_model_name(model_filename)

    header += model_name 
    header += " (attention mode: " + (attention_mode if attention_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_installed:
        header += " -NOT INSTALLED-"
    elif attention_mode not in attention_modes_supported:
        header += " -NOT SUPPORTED-"

    if compile:
        header += ", pytorch compilation ON"
    header += ") </h2><div class=line></div>    "


    return header

def apply_changes(  state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    save_path_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    default_ui_choice ="t2v",
                    boost_choice = 1,
                    clear_file_list = 0,
                    reload_choice = 1
):
    if args.lock_config:
        return
    if gen_in_progress:
        yield "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>"
        return
    global offloadobj, wan_model, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset, loras_presets
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices_t2v[transformer_t2v_choice], 
                     "transformer_filename_i2v": transformer_choices_i2v[transformer_i2v_choice],  
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "save_path" : save_path_choice,
                     "compile" : compile_choice,
                     "profile" : profile_choice,
                     "vae_config" : vae_config_choice,
                     "metadata_choice": metadata_choice,
                     "default_ui" : default_ui_choice,
                     "boost" : boost_choice,
                     "clear_file_list" : clear_file_list,
                     "reload_model" : reload_choice,
                       }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_transformer:
            server_config["transformer_filename"] = old_server_config["transformer_filename"]
            server_config["transformer_filename_i2v"] = old_server_config["transformer_filename_i2v"]
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config.get(k, None)
        if v != v_old:
            changes.append(k)

    global attention_mode, profile, compile, transformer_filename_t2v, transformer_filename_i2v, text_encoder_filename, vae_config, boost, lora_dir, reload_needed
    attention_mode = server_config["attention_mode"]
    profile = server_config["profile"]
    compile = server_config["compile"]
    transformer_filename_t2v = server_config["transformer_filename"]
    transformer_filename_i2v = server_config["transformer_filename_i2v"]
    text_encoder_filename = server_config["text_encoder_filename"]
    vae_config = server_config["vae_config"]
    boost = server_config["boost"]
    if  all(change in ["attention_mode", "vae_config", "default_ui", "boost", "save_path", "metadata_choice", "clear_file_list"] for change in changes ):
        pass
    else:
        reload_needed = True


    yield "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>"



from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)

def build_callback(taskid, state, pipe, num_inference_steps, repeats):
    start_time = time.time()
    def update_progress(step_idx, _):
        with tracker_lock:
            step_idx += 1
            if state.get("abort", False):
                # pipe._interrupt = True
                phase = "Aborting"
            elif step_idx  == num_inference_steps:
                phase = "VAE Decoding"
            else:
                phase = "Denoising"
            elapsed = time.time() - start_time
            progress_tracker[taskid] = {
                'current_step': step_idx,
                'total_steps': num_inference_steps,
                'start_time': start_time,
                'last_update': time.time(),
                'repeats': repeats,
                'status': phase
            }
    return update_progress

def refresh_gallery(state):
    return state

def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
        if state.get("last_selected", True):
            file_list = state.get("file_list", [])
            choice = len(file_list) - 1
            

    state["extra_orders"] = 0
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=True), gr.Button(visible=False), gr.Checkbox(visible=False), gr.Text(visible=False, value="")

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        choice = data.get("index",0)
        file_list = state.get("file_list", [])
        state["last_selected"] = (choice + 1) >= len(file_list)
        state["selected"] = choice
    return 

def expand_slist(slist, num_inference_steps ):
    new_slist= []
    inc =  len(slist) / num_inference_steps 
    pos = 0
    for i in range(num_inference_steps):
        new_slist.append(slist[ int(pos)])
        pos += inc
    return new_slist

def convert_image(image):
    from PIL import ExifTags

    image = image.convert('RGB')
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break            
    exif = image.getexif()
    if not orientation in exif:
        return image
    if exif[orientation] == 3:
        image=image.rotate(180, expand=True)
    elif exif[orientation] == 6:
        image=image.rotate(270, expand=True)
    elif exif[orientation] == 8:
        image=image.rotate(90, expand=True)
    return image

def generate_video(
    task_id,
    prompt,
    negative_prompt,    
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    multi_images_gen_type,
    tea_cache,
    tea_cache_start_step_perc,    
    loras_choices,
    loras_mult_choices,
    image_prompt_type,
    image_to_continue,
    image_to_end,
    video_to_continue,
    max_frames,
    RIFLEx_setting,
    slg_switch,
    slg_layers,    
    slg_start,
    slg_end,
    cfg_star_switch,
    cfg_zero_step,
    state,
    image2video,
    progress=gr.Progress() #track_tqdm= True

):

    global wan_model, offloadobj, reload_needed, last_model_type
    file_model_needed = model_needed(image2video)
    with lock:
        queue_not_empty = len(queue) > 0
    if(last_model_type != image2video and (queue_not_empty or server_config.get("reload_model",1) == 2) and (file_model_needed !=  model_filename or reload_needed)):
        del wan_model
        if offloadobj is not None:
            offloadobj.release()
            del offloadobj
        gc.collect()
        print(f"Loading model {get_model_name(file_model_needed)}...")
        wan_model, offloadobj, trans = load_models(image2video)
        print(f"Model loaded")
        reload_needed=  False

    if wan_model == None:
        raise gr.Error("Unable to generate a Video while a new configuration is being applied.")
    if attention_mode == "auto":
        attn = get_auto_attention()
    elif attention_mode in attention_modes_supported:
        attn = attention_mode
    else:
        gr.Info(f"You have selected attention mode '{attention_mode}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        return

    raw_resolution = resolution
    width, height = resolution.split("x")
    width, height = int(width), int(height)

    if slg_switch == 0:
        slg_layers = None
    if image2video:
        if "480p" in  model_filename and not "Fun" in model_filename and width * height > 848*480:
            gr.Info("You must use the 720P image to video model to generate videos with a resolution equivalent to 720P")
            return

        resolution = str(width) + "*" + str(height)  
        if  resolution not in ['720*1280', '1280*720', '480*832', '832*480']:
            gr.Info(f"Resolution {resolution} not supported by image 2 video")
            return

    if "1.3B" in  model_filename and width * height > 848*480:
        gr.Info("You must use the 14B model to generate videos with a resolution equivalent to 720P")
        return
    
    offload.shared_state["_attention"] =  attn
 
     # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if vae_config == 0:
        if device_mem_capacity >= 24000:
            use_vae_config = 1            
        elif device_mem_capacity >= 8000:
            use_vae_config = 2
        else:          
            use_vae_config = 3
    else:
        use_vae_config = vae_config

    if use_vae_config == 1:
        VAE_tile_size = 0  
    elif use_vae_config == 2:
        VAE_tile_size = 256  
    else: 
        VAE_tile_size = 128  

    trans = wan_model.model

    global gen_in_progress
    gen_in_progress = True
    temp_filename = None
    if image2video:
        if video_to_continue != None and len(video_to_continue) >0 :
            input_image_or_video_path = video_to_continue
            # pipeline.num_input_frames = max_frames
            # pipeline.max_frames = max_frames
    else:
        input_image_or_video_path = None

    loras = state["loras"]
    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_nums = []
        if len(loras_mult_choices) > 0:
            loras_mult_choices_list = loras_mult_choices.replace("\r", "").split("\n")
            loras_mult_choices_list = [multi for multi in loras_mult_choices_list if len(multi)>0 and not multi.startswith("#")]
            loras_mult_choices = " ".join(loras_mult_choices_list)
            list_mult_choices_str = loras_mult_choices.split(" ")
            for i, mult in enumerate(list_mult_choices_str):
                mult = mult.strip()
                if "," in mult:
                    multlist = mult.split(",")
                    slist = []
                    for smult in multlist:
                        if not is_float(smult):                
                            raise gr.Error(f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid")
                        slist.append(float(smult))
                    slist = expand_slist(slist, num_inference_steps )
                    list_mult_choices_nums.append(slist)
                else:
                    if not is_float(mult):                
                        raise gr.Error(f"Lora Multiplier no {i+1} ({mult}) is invalid")
                    list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) < len(loras_choices):
            list_mult_choices_nums  += [1.0] * ( len(loras_choices) - len(list_mult_choices_nums ) )
        loras_selected = [ lora for i, lora in enumerate(loras) if str(i) in loras_choices]
        pinnedLora = profile !=5 #False # # # 
        offload.load_loras_into_model(trans, loras_selected, list_mult_choices_nums, activate_all_loras=True, preprocess_sd=preprocess_loras, pinnedLora=pinnedLora, split_linear_modules_map = None) 
        errors = trans._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise gr.Error("Error while loading Loras: " + ", ".join(error_files))
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
 
    enable_RIFLEx = RIFLEx_setting == 0 and video_length > (6* 16) or RIFLEx_setting == 1
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
   # TeaCache   
    trans.enable_teacache = tea_cache > 0
    if trans.enable_teacache:
        trans.teacache_multiplier = tea_cache
        trans.rel_l1_thresh = 0
        trans.teacache_start_step =  int(tea_cache_start_step_perc*num_inference_steps/100)

        if image2video:
            if '480p' in transformer_filename_i2v: 
                # teacache_thresholds = [0.13, .19, 0.26]
                trans.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
            elif '720p' in transformer_filename_i2v:
                teacache_thresholds = [0.18, 0.2 , 0.3]
                trans.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
            else:
                raise gr.Error("Teacache not supported for this model")
        else:
            if '1.3B' in transformer_filename_t2v:
                # teacache_thresholds= [0.05, 0.07, 0.08]
                trans.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            elif '14B' in transformer_filename_t2v:
                # teacache_thresholds = [0.14, 0.15, 0.2]
                trans.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            else:
                    raise gr.Error("Teacache not supported for this model")

    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    global file_list
    clear_file_list = server_config.get("clear_file_list", 0)    
    file_list = state.get("file_list", [])
    if clear_file_list > 0:
        file_list_current_size = len(file_list)
        keep_file_from = max(file_list_current_size - clear_file_list, 0)
        files_removed = keep_file_from
        choice = state.get("selected",0)
        choice = max(choice- files_removed, 0)
        file_list = file_list[ keep_file_from: ]
    else:
        file_list = []
        choice = 0
    state["selected"] = choice         
    state["file_list"] = file_list    


    global save_path
    os.makedirs(save_path, exist_ok=True)
    video_no = 0
    abort = False
    repeats = f"{video_no}/{repeat_generation}"
    callback = build_callback(task_id, state, trans, num_inference_steps, repeats)
    offload.shared_state["callback"] = callback
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    for i in range(repeat_generation):
        try:
            with tracker_lock:
                start_time = time.time()
                progress_tracker[task_id] = {
                    'current_step': 0,
                    'total_steps': num_inference_steps,
                    'start_time': start_time,
                    'last_update': start_time,
                    'repeats': f"{video_no}/{repeat_generation}",
                    'status': "Encoding Prompt"
                }
            if trans.enable_teacache:
                trans.teacache_counter = 0
                trans.num_steps = num_inference_steps                
                trans.teacache_skipped_steps = 0
                trans.previous_residual_uncond = None
                trans.previous_residual_cond = None

            video_no += 1
            if image2video:
                samples = wan_model.generate(
                    prompt,
                    convert_image(image_to_continue),  
                    convert_image(image_to_end) if image_to_end != None else None,  
                    frame_num=(video_length // 4)* 4 + 1,
                    max_area=MAX_AREA_CONFIGS[resolution], 
                    shift=flow_shift,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    offload_model=False,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start/100,
                    slg_end = slg_end/100,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    add_frames_for_end_image = not "Fun" in transformer_filename_i2v                       
                )
            else:
                samples = wan_model.generate(
                    prompt,
                    frame_num=(video_length // 4)* 4 + 1,
                    size=(width, height),
                    shift=flow_shift,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    offload_model=False,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start/100,
                    slg_end = slg_end/100,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                )
        except Exception as e:
            gen_in_progress = False
            if temp_filename!= None and  os.path.isfile(temp_filename):
                os.remove(temp_filename)
            offload.last_offload_obj.unload_all()
            offload.unload_loras_from_model(trans)
            # if compile:
            #     cache_size = torch._dynamo.config.cache_size_limit                                      
            #     torch.compiler.reset()
            #     torch._dynamo.config.cache_size_limit = cache_size

            gc.collect()
            torch.cuda.empty_cache()
            s = str(e)
            keyword_list = ["vram", "VRAM", "memory","allocat"]
            VRAM_crash= False
            if any( keyword in s for keyword in keyword_list):
                VRAM_crash = True
            else:
                stack = traceback.extract_stack(f=None, limit=5)
                for frame in stack:
                    if any( keyword in frame.name for keyword in keyword_list):
                        VRAM_crash = True
                        break
            state["prompt"] = ""
            if VRAM_crash:
                raise gr.Error("The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames.")
            else:
                raise gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")
        finally:
            with tracker_lock:
                if task_id in progress_tracker:
                    del progress_tracker[task_id]

        if trans.enable_teacache:
            print(f"Teacache Skipped Steps:{trans.teacache_skipped_steps}/{num_inference_steps}" )
            trans.previous_residual_uncond = None
            trans.previous_residual_cond = None

        if samples != None:
            samples = samples.to("cpu")
        offload.last_offload_obj.unload_all()
        gc.collect()
        torch.cuda.empty_cache()

        if samples == None:
            end_time = time.time()
            abort = True
            state["prompt"] = ""
            print(f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s")
        else:
            sample = samples.cpu()
            # video = rearrange(sample.cpu().numpy(), "c t h w -> t h w c")

            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
            if os.name == 'nt':
                file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(prompt[:50]).strip()}.mp4"
            else:
                file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(prompt[:100]).strip()}.mp4"
            video_path = os.path.join(save_path, file_name)        
            cache_video(
                tensor=sample[None],
                save_file=video_path,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            configs = get_settings_dict(state, use_image2video, prompt, 0 if image_to_end == None else 1 , video_length, raw_resolution, num_inference_steps, seed, repeat_generation, multi_images_gen_type, guidance_scale, flow_shift, negative_prompt, loras_choices, 
                  loras_mult_choices, tea_cache , tea_cache_start_step_perc, RIFLEx_setting, slg_switch, slg_layers, slg_start, slg_end, cfg_star_switch, cfg_zero_step)

            metadata_choice = server_config.get("metadata_choice","metadata")
            if metadata_choice == "json":
                with open(video_path.replace('.mp4', '.json'), 'w') as f:
                    json.dump(configs, f, indent=4)
            elif metadata_choice == "metadata":
                from mutagen.mp4 import MP4
                file = MP4(video_path)
                file.tags['©cmt'] = [json.dumps(configs)]
                file.save()

            print(f"New video saved to Path: "+video_path)
            file_list.append(video_path)
            state['update_gallery'] = True
        seed += 1

        last_model_type = image2video
  
    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)
    gen_in_progress = False
    offload.unload_loras_from_model(trans)


def get_new_preset_msg(advanced = True):
    if advanced:
        return "Enter here a Name for a Lora Preset or Choose one in the List"
    else:
        return "Choose a Lora Preset in this List to Apply a Special Effect"


def validate_delete_lset(lset_name):
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info(f"Choose a Preset to delete")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Checkbox(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True) 
    
def validate_save_lset(lset_name):
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info("Please enter a name for the preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False),gr.Checkbox(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True),gr.Checkbox(visible= True)

def cancel_lset():
    return gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)



def save_lset(state, lset_name, loras_choices, loras_mult_choices, prompt, save_lset_prompt_cbox):    
    loras_presets = state["loras_presets"] 
    loras = state["loras"]
    if state.get("validate_success",0) == 0:
        pass
    if len(lset_name) == 0 or lset_name == get_new_preset_msg(True) or lset_name == get_new_preset_msg(False):
        gr.Info("Please enter a name for the preset")
        lset_choices =[("Please enter a name for a Lora Preset","")]
    else:
        lset_name = sanitize_file_name(lset_name)

        loras_choices_files = [ Path(loras[int(choice_no)]).parts[-1] for choice_no in loras_choices  ]
        lset  = {"loras" : loras_choices_files, "loras_mult" : loras_mult_choices}
        if save_lset_prompt_cbox!=1:
            prompts = prompt.replace("\r", "").split("\n")
            prompts = [prompt for prompt in prompts if len(prompt)> 0 and prompt.startswith("#")]
            prompt = "\n".join(prompts)

        if len(prompt) > 0:
            lset["prompt"] = prompt
        lset["full_prompt"] = save_lset_prompt_cbox ==1
        

        lset_name_filename = lset_name + ".lset" 
        full_lset_name_filename = os.path.join(get_lora_dir(use_image2video), lset_name_filename) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset, indent=4))

        if lset_name in loras_presets:
            gr.Info(f"Lora Preset '{lset_name}' has been updated")
        else:
            gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(Path(Path(lset_name_filename).parts[-1]).stem )
        lset_choices = [ ( preset, preset) for preset in loras_presets ]
        lset_choices.append( (get_new_preset_msg(), ""))
        state["loras_presets"] = loras_presets
    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)

def delete_lset(state, lset_name):
    loras_presets = state["loras_presets"]
    lset_name_filename = os.path.join( get_lora_dir(use_image2video),  sanitize_file_name(lset_name) + ".lset" )
    if len(lset_name) > 0 and lset_name != get_new_preset_msg(True) and  lset_name != get_new_preset_msg(False):
        if not os.path.isfile(lset_name_filename):
            raise gr.Error(f"Preset '{lset_name}' not found ")
        os.remove(lset_name_filename)
        pos = loras_presets.index(lset_name) 
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = len(loras_presets) 
        gr.Info(f"Choose a Preset to delete")

    state["loras_presets"] = loras_presets

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg(), ""))
    return  gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1]), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Checkbox(visible= False)

def refresh_lora_list(state, lset_name, loras_choices):
    loras_names = state["loras_names"]
    prev_lora_names_selected = [ loras_names[int(i)] for i in loras_choices]

    loras, loras_names, loras_presets, _, _, _, _  = setup_loras(use_image2video, None,  get_lora_dir(use_image2video), lora_preselected_preset, None)
    state["loras"] = loras
    state["loras_names"] = loras_names
    state["loras_presets"] = loras_presets

    gc.collect()
    new_loras_choices = [ (loras_name, str(i)) for i,loras_name in enumerate(loras_names)]
    new_loras_dict = { loras_name: str(i) for i,loras_name in enumerate(loras_names) }
    lora_names_selected = []
    for lora in prev_lora_names_selected:
        lora_id = new_loras_dict.get(lora, None)
        if lora_id!= None:
            lora_names_selected.append(lora_id)

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg( state["advanced"]), "")) 
    if lset_name in loras_presets:
        pos = loras_presets.index(lset_name) 
    else:
        pos = len(loras_presets)
        lset_name =""
    
    errors = getattr(wan_model.model, "_loras_errors", "")
    if errors !=None and len(errors) > 0:
        error_files = [path for path, _ in errors]
        gr.Info("Error while refreshing Lora List, invalid Lora files: " + ", ".join(error_files))
    else:
        gr.Info("Lora List has been refreshed")


    return gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1]), gr.Dropdown(choices=new_loras_choices, value= lora_names_selected) 

def apply_lset(state, wizard_prompt_activated, lset_name, loras_choices, loras_mult_choices, prompt):

    state["apply_success"] = 0

    if len(lset_name) == 0 or lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False):
        gr.Info("Please choose a preset in the list or create one")
    else:
        loras = state["loras"]
        loras_choices, loras_mult_choices, preset_prompt, full_prompt, error = extract_preset(lset_name, loras)
        if len(error) > 0:
            gr.Info(error)
        else:
            if full_prompt:
                prompt = preset_prompt
            elif len(preset_prompt) > 0:
                prompts = prompt.replace("\r", "").split("\n")
                prompts = [prompt for prompt in prompts if len(prompt)>0 and not prompt.startswith("#")]
                prompt = "\n".join(prompts) 
                prompt = preset_prompt + '\n' + prompt
            gr.Info(f"Lora Preset '{lset_name}' has been applied")
            state["apply_success"] = 1
            wizard_prompt_activated = "on"

    return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt


def extract_prompt_from_wizard(state, variables_names, prompt, wizard_prompt, allow_null_values, *args):

    prompts = wizard_prompt.replace("\r" ,"").split("\n")

    new_prompts = [] 
    macro_already_written = False
    for prompt in prompts:
        if not macro_already_written and not prompt.startswith("#") and "{"  in prompt and "}"  in prompt:
            variables =  variables_names.split("\n")   
            values = args[:len(variables)]
            macro = "! "
            for i, (variable, value) in enumerate(zip(variables, values)):
                if len(value) == 0 and not allow_null_values:
                    return prompt, "You need to provide a value for '" + variable + "'" 
                sub_values= [ "\"" + sub_value + "\"" for sub_value in value.split("\n") ]
                value = ",".join(sub_values)
                if i>0:
                    macro += " : "    
                macro += "{" + variable + "}"+ f"={value}"
            if len(variables) > 0:
                macro_already_written = True
                new_prompts.append(macro)
            new_prompts.append(prompt)
        else:
            new_prompts.append(prompt)

    prompt = "\n".join(new_prompts)
    return prompt, ""

def validate_wizard_prompt(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):
    state["validate_success"] = 0

    if wizard_prompt_activated != "on":
        state["validate_success"] = 1
        return prompt

    prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt, wizard_prompt, False, *args)
    if len(errors) > 0:
        gr.Info(errors)
        return prompt

    state["validate_success"] = 1

    return prompt

def fill_prompt_from_wizard(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):

    if wizard_prompt_activated == "on":
        prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt,  wizard_prompt, True, *args)
        if len(errors) > 0:
            gr.Info(errors)

        wizard_prompt_activated = "off"

    return wizard_prompt_activated, "", gr.Textbox(visible= True, value =prompt) , gr.Textbox(visible= False), gr.Column(visible = True), *[gr.Column(visible = False)] * 2,  *[gr.Textbox(visible= False)] * PROMPT_VARS_MAX

def extract_wizard_prompt(prompt):
    variables = []
    values = {}
    prompts = prompt.replace("\r" ,"").split("\n")
    if sum(prompt.startswith("!") for prompt in prompts) > 1:
        return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

    new_prompts = [] 
    errors = ""
    for prompt in prompts:
        if prompt.startswith("!"):
            variables, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"
            values, errors = prompt_parser.extract_variable_values(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
        else:
            variables_extra, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            variables += variables_extra
            variables = [var for pos, var in enumerate(variables) if var not in variables[:pos]]
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

            new_prompts.append(prompt)
    wizard_prompt = "\n".join(new_prompts)
    return  wizard_prompt, variables, values, errors

def fill_wizard_prompt(state, wizard_prompt_activated, prompt, wizard_prompt):
    def get_hidden_textboxes(num = PROMPT_VARS_MAX ):
        return [gr.Textbox(value="", visible=False)] * num

    hidden_column =  gr.Column(visible = False)
    visible_column =  gr.Column(visible = True)

    wizard_prompt_activated  = "off"  
    if state["advanced"] or state.get("apply_success") != 1:
        return wizard_prompt_activated, gr.Text(), prompt, wizard_prompt, gr.Column(), gr.Column(), hidden_column,  *get_hidden_textboxes() 
    prompt_parts= []

    wizard_prompt, variables, values, errors =  extract_wizard_prompt(prompt)
    if len(errors) > 0:
        gr.Info( errors )
        return wizard_prompt_activated, "", gr.Textbox(prompt, visible=True), gr.Textbox(wizard_prompt, visible=False), visible_column, *[hidden_column] * 2, *get_hidden_textboxes()

    for variable in variables:
        value = values.get(variable, "")
        prompt_parts.append(gr.Textbox( placeholder=variable, info= variable, visible= True, value= "\n".join(value) ))
    any_macro = len(variables) > 0

    prompt_parts += get_hidden_textboxes(PROMPT_VARS_MAX-len(prompt_parts))

    variables_names= "\n".join(variables)
    wizard_prompt_activated  = "on"

    return wizard_prompt_activated, variables_names,  gr.Textbox(prompt, visible = False),  gr.Textbox(wizard_prompt, visible = True),   hidden_column, visible_column, visible_column if any_macro else hidden_column, *prompt_parts

def switch_prompt_type(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars):
    if state["advanced"]:
        return fill_prompt_from_wizard(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars)
    else:
        state["apply_success"] = 1
        return fill_wizard_prompt(state, wizard_prompt_activated_var, prompt, wizard_prompt)

visible= False
def switch_advanced(state, new_advanced, lset_name):
    state["advanced"] = new_advanced
    loras_presets = state["loras_presets"]
    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((get_new_preset_msg(new_advanced), ""))
    if lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False) or lset_name=="":
        lset_name =  get_new_preset_msg(new_advanced)

    if only_allow_edit_in_advanced:
        return  gr.Row(visible=new_advanced), gr.Row(visible=new_advanced), gr.Button(visible=new_advanced), gr.Row(visible= not new_advanced), gr.Dropdown(choices=lset_choices, value= lset_name)
    else:
        return  gr.Row(visible=new_advanced), gr.Row(visible=True), gr.Button(visible=True), gr.Row(visible= False), gr.Dropdown(choices=lset_choices, value= lset_name)


def get_settings_dict(state, i2v, prompt, image_prompt_type, video_length, resolution, num_inference_steps, seed, repeat_generation, multi_images_gen_type, guidance_scale, flow_shift, negative_prompt, loras_choices, 
                      loras_mult_choices, tea_cache_setting, tea_cache_start_step_perc, RIFLEx_setting, slg_switch, slg_layers, slg_start_perc, slg_end_perc, cfg_star_switch, cfg_zero_step):

    loras = state["loras"]
    activated_loras = [Path( loras[int(no)]).parts[-1]  for no in loras_choices ]

    ui_settings = {
        "prompts": prompt,
        "resolution": resolution,
        "video_length": video_length,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "repeat_generation": repeat_generation,
        "multi_images_gen_type": multi_images_gen_type,        
        "guidance_scale": guidance_scale,
        "flow_shift": flow_shift,
        "negative_prompt": negative_prompt,
        "activated_loras": activated_loras,
        "loras_multipliers": loras_mult_choices,
        "tea_cache": tea_cache_setting,
        "tea_cache_start_step_perc": tea_cache_start_step_perc,
        "RIFLEx_setting": RIFLEx_setting,
        "slg_switch": slg_switch,
        "slg_layers": slg_layers,
        "slg_start_perc": slg_start_perc,
        "slg_end_perc": slg_end_perc,
        "cfg_star_switch": cfg_star_switch, 
        "cfg_zero_step": cfg_zero_step
    }

    if i2v:
        ui_settings["type"] = "Wan2.1GP by DeepBeepMeep - image2video"
        ui_settings["image_prompt_type"] = image_prompt_type
    else: 
        ui_settings["type"] = "Wan2.1GP by DeepBeepMeep - text2video"

    return ui_settings

def save_settings(state, prompt, image_prompt_type, video_length, resolution, num_inference_steps, seed, repeat_generation, multi_images_gen_type, guidance_scale, flow_shift, negative_prompt, loras_choices, 
                      loras_mult_choices, tea_cache_setting, tea_cache_start_step_perc, RIFLEx_setting, slg_switch, slg_layers, slg_start_perc, slg_end_perc, cfg_star_switch, cfg_zero_step):

    if state.get("validate_success",0) != 1:
        return

    ui_defaults = get_settings_dict(state, use_image2video, prompt, image_prompt_type, video_length, resolution, num_inference_steps, seed, repeat_generation, multi_images_gen_type, guidance_scale, flow_shift, negative_prompt, loras_choices, 
                      loras_mult_choices, tea_cache_setting, tea_cache_start_step_perc, RIFLEx_setting, slg_switch, slg_layers, slg_start_perc, slg_end_perc, cfg_star_switch, cfg_zero_step)

    defaults_filename = get_settings_file_name(use_image2video)

    with open(defaults_filename, "w", encoding="utf-8") as f:
        json.dump(ui_defaults, f, indent=4)

    gr.Info("New Default Settings saved")

def download_loras():
    from huggingface_hub import  snapshot_download    
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Please wait while the Loras are being downloaded</B></FONT>", *[gr.Column(visible=False)] * 2
    lora_dir = get_lora_dir(True)
    log_path = os.path.join(lora_dir, "log.txt")
    if not os.path.isfile(log_path):
        import shutil 
        tmp_path = os.path.join(lora_dir, "tmp_lora_dowload")
        import glob
        snapshot_download(repo_id="DeepBeepMeep/Wan2.1",  allow_patterns="loras_i2v/*", local_dir= tmp_path)
        for f in glob.glob(os.path.join(tmp_path, "loras_i2v", "*.*")):
            target_file = os.path.join(lora_dir,  Path(f).parts[-1] )
            if os.path.isfile(target_file):
                os.remove(f)
            else:
                shutil.move(f, lora_dir) 
    try:
        os.remove(tmp_path)
    except:
        pass
    yield gr.Row(visible=True), "<B><FONT SIZE=3>Loras have been completely downloaded</B></FONT>", *[gr.Column(visible=True)] * 2

    from datetime import datetime
    dt = datetime.today().strftime('%Y-%m-%d')
    with open( log_path, "w", encoding="utf-8") as writer:
        writer.write(f"Loras downloaded on the {dt} at {time.time()} on the {time.time()}")
    return

def generate_video_tab(image2video=False):
    filename = transformer_filename_i2v if image2video else transformer_filename_t2v
    ui_defaults=  get_default_settings(filename, image2video)

    state_dict = {}

    state_dict["advanced"] = advanced
    state_dict["loras_model"] = filename
    preset_to_load = lora_preselected_preset if lora_preselected_preset_for_i2v == image2video else "" 

    loras, loras_names, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(image2video,  None,  get_lora_dir(image2video), preset_to_load, None)

    state_dict["loras"] = loras
    state_dict["loras_presets"] = loras_presets
    state_dict["loras_names"] = loras_names

    launch_prompt = ""
    launch_preset = ""
    launch_loras = []
    launch_multis_str = ""

    if len(default_lora_preset) > 0 and image2video == lora_preselected_preset_for_i2v:
        launch_preset = default_lora_preset
        launch_prompt = default_lora_preset_prompt 
        launch_loras = default_loras_choices
        launch_multis_str = default_loras_multis_str

    if len(launch_prompt) == 0:
        launch_prompt = ui_defaults["prompts"]
    if len(launch_loras) == 0:
        activated_loras = ui_defaults["activated_loras"]
        launch_multis_str = ui_defaults["loras_multipliers"]

        if len(activated_loras) > 0:
            lora_filenames = [os.path.basename(lora_path) for lora_path in loras]
            activated_indices = []
            for lora_file in ui_defaults["activated_loras"]:
                try:
                    idx = lora_filenames.index(lora_file)
                    activated_indices.append(str(idx))
                except ValueError:
                    print(f"Warning: Lora file {lora_file} from config not found in loras directory")
            launch_loras = activated_indices


    header = gr.Markdown(generate_header(model_filename, compile, attention_mode))
    with gr.Row(visible= image2video):
        with gr.Row(scale =2):
            gr.Markdown("<I>Wan2GP's Lora Festival ! Press the following button to download i2v <B>Remade</B> Loras collection (and bonuses Loras).")
        with gr.Row(scale =1):
            download_loras_btn = gr.Button("---> Let the Lora's Festival Start !", scale =1)
        with gr.Row(scale =1):
            gr.Markdown("")
    with gr.Row(visible= image2video) as download_status_row: 
        download_status = gr.Markdown()
    with gr.Row():
        with gr.Column():
            with gr.Column(visible=False, elem_id="image-modal-container") as modal_container:
                with gr.Row(elem_id="image-modal-close-button-row"):
                     close_modal_button = gr.Button("❌", size="sm")
                modal_image_display = gr.Image(label="Full Resolution Image", interactive=False, show_label=False)
            gallery_update_trigger = gr.Textbox(value="0", visible=False, label="_gallery_trigger")
            with gr.Row(visible= len(loras)>0) as presets_column:
                lset_choices = [ (preset, preset) for preset in loras_presets ] + [(get_new_preset_msg(advanced), "")]
                with gr.Column(scale=6):
                    lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=5, filterable=True, choices= lset_choices, value=launch_preset)
                with gr.Column(scale=1):
                    with gr.Row(height=17):
                        apply_lset_btn = gr.Button("Apply Lora Preset", size="sm", min_width= 1)
                        refresh_lora_btn = gr.Button("Refresh", size="sm", min_width= 1, visible=advanced or not only_allow_edit_in_advanced)
                        save_lset_prompt_drop= gr.Dropdown(
                            choices=[
                                ("Save Prompt Comments Only", 0),
                                ("Save Full Prompt", 1)
                            ],  show_label= False, container=False, value =1, visible= False
                        ) 
                    with gr.Row(height=17, visible=False) as refresh2_row:
                        refresh_lora_btn2 = gr.Button("Refresh", size="sm", min_width= 1)

                    with gr.Row(height=17, visible=advanced or not only_allow_edit_in_advanced) as preset_buttons_rows:
                        confirm_save_lset_btn = gr.Button("Go Ahead Save it !", size="sm", min_width= 1, visible=False) 
                        confirm_delete_lset_btn = gr.Button("Go Ahead Delete it !", size="sm", min_width= 1, visible=False) 
                        save_lset_btn = gr.Button("Save", size="sm", min_width= 1)
                        delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1)
                        cancel_lset_btn = gr.Button("Don't do it !", size="sm", min_width= 1 , visible=False)  
            video_to_continue = gr.Video(label= "Video to continue", visible= image2video and False) #######
            image_prompt_type= ui_defaults.get("image_prompt_type",0)
            image_prompt_type_radio = gr.Radio( [("Use only a Start Image", 0),("Use both a Start and an End Image", 1)], value =image_prompt_type, label="Location", show_label= False, scale= 3, visible=image2video)

            if args.multiple_images:  
                image_to_continue = gr.Gallery(
                        label="Images as starting points for new videos", type ="pil", #file_types= "image", 
                        columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible=image2video)
            else:
                image_to_continue = gr.Image(label= "Image as a starting point for a new video", type ="pil", visible=image2video)

            if args.multiple_images:  
                image_to_end  = gr.Gallery(
                        label="Images as ending points for new videos", type ="pil", #file_types= "image", 
                        columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible=image_prompt_type==1)
            else:
                image_to_end = gr.Image(label= "Last Image for a new video", type ="pil", visible=image_prompt_type==1)

            def switch_image_prompt_type_radio(image_prompt_type_radio):
                if args.multiple_images:
                    return gr.Gallery(visible = (image_prompt_type_radio == 1)  )
                else:
                    return gr.Image(visible = (image_prompt_type_radio == 1)  )

            image_prompt_type_radio.change(fn=switch_image_prompt_type_radio, inputs=[image_prompt_type_radio], outputs=[image_to_end]) 


            advanced_prompt = advanced
            prompt_vars=[]

            if advanced_prompt:
                default_wizard_prompt, variables, values= None, None, None
            else:                 
                default_wizard_prompt, variables, values, errors =  extract_wizard_prompt(launch_prompt)
                advanced_prompt  = len(errors) > 0
            with gr.Column(visible= advanced_prompt) as prompt_column_advanced:
                prompt = gr.Textbox( visible= advanced_prompt, label="Prompts (each new line of prompt will generate a new video, # lines = comments, ! lines = macros)", value=launch_prompt, lines=3)

            with gr.Column(visible=not advanced_prompt and len(variables) > 0) as prompt_column_wizard_vars:
                gr.Markdown("<B>Please fill the following input fields to adapt automatically the Prompt:</B>")
                wizard_prompt_activated = "off"
                wizard_variables = ""
                with gr.Row():
                    if not advanced_prompt:
                        for variable in variables:
                            value = values.get(variable, "")
                            prompt_vars.append(gr.Textbox( placeholder=variable, min_width=80, show_label= False, info= variable, visible= True, value= "\n".join(value) ))
                        wizard_prompt_activated = "on"
                        if len(variables) > 0:
                            wizard_variables = "\n".join(variables)
                    for _ in range( PROMPT_VARS_MAX - len(prompt_vars)):
                        prompt_vars.append(gr.Textbox(visible= False, min_width=80, show_label= False))
 			
            with gr.Column(not advanced_prompt) as prompt_column_wizard:
                wizard_prompt = gr.Textbox(visible = not advanced_prompt, label="Prompts (each new line of prompt will generate a new video, # lines = comments)", value=default_wizard_prompt, lines=3)
                wizard_prompt_activated_var = gr.Text(wizard_prompt_activated, visible= False)
                wizard_variables_var = gr.Text(wizard_variables, visible = False)
            state = gr.State(state_dict)     
            with gr.Row():
                if image2video:
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("720p", "1280x720"),
                            ("480p", "832x480"),
                        ],
                        value=ui_defaults["resolution"],
                        label="Resolution (video will have the same height / width ratio than the original image)"
                    )
                else:
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("720x1280 (9:16, 720p)", "720x1280"), 
                            ("1024x1024 (4:3, 720p)", "1024x024"),
                            # ("832x1104 (3:4, 720p)", "832x1104"),
                            # ("960x960 (1:1, 720p)", "960x960"),
                            # 480p
                            # ("960x544 (16:9, 480p)", "960x544"),
                            ("832x480 (16:9, 480p)", "832x480"),
                            ("480x832 (9:16, 480p)", "480x832"),
                            # ("832x624 (4:3, 540p)", "832x624"), 
                            # ("624x832 (3:4, 540p)", "624x832"),
                            # ("720x720 (1:1, 540p)", "720x720"),
                        ],
                        value=ui_defaults["resolution"],
                        label="Resolution"
                    )
            with gr.Row():
                with gr.Column():
                    video_length = gr.Slider(5, 193, value=ui_defaults["video_length"], step=4, label="Number of frames (16 = 1s)")
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=ui_defaults["num_inference_steps"], step=1, label="Number of Inference Steps")
            with gr.Row():
                max_frames = gr.Slider(1, 100, value=9, step=1, label="Number of input frames to use for Video2World prediction", visible=image2video and False) #########
            show_advanced = gr.Checkbox(label="Advanced Mode", value=advanced)
            with gr.Row(visible=advanced) as advanced_row:
                with gr.Column():
                    seed = gr.Slider(-1, 999999999, value=ui_defaults["seed"], step=1, label="Seed (-1 for random)") 
                    with gr.Row():
                        repeat_generation = gr.Slider(1, 25.0, value=ui_defaults["repeat_generation"], step=1, label="Default Number of Generated Videos per Prompt") 
                        multi_images_gen_type = gr.Dropdown( value=ui_defaults["multi_images_gen_type"], 
                            choices=[
                                ("Generate every combination of images and texts", 0),
                                ("Match images and text prompts", 1),
                            ], visible= args.multiple_images, label= "Multiple Images as Texts Prompts"
                        )
                    with gr.Row():
                        guidance_scale = gr.Slider(1.0, 20.0, value=ui_defaults["guidance_scale"], step=0.5, label="Guidance Scale", visible=True)
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale", visible=False)
                        flow_shift = gr.Slider(0.0, 25.0, value=ui_defaults["flow_shift"], step=0.1, label="Shift Scale") 
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative Prompt", value=ui_defaults["negative_prompt"])
                    with gr.Column(visible = len(loras)>0) as loras_column:
                        gr.Markdown("<B>Loras can be used to create special effects on the video by mentioning a trigger word in the Prompt. You can save Loras combinations in presets.</B>")
                        loras_choices = gr.Dropdown(
                            choices=[
                                (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                            ],
                            value= launch_loras,
                            multiselect= True,
                            label="Activated Loras"
                        )
                        loras_mult_choices = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters or carriage returns, line that starts with # are ignored", value=launch_multis_str)
                    with gr.Row():
                        gr.Markdown("<B>Tea Cache accelerates by skipping intelligently some steps, the more steps are skipped the lower the quality of the video (Tea Cache consumes also VRAM)</B>")
                    with gr.Row():
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Tea Cache Disabled", 0),
                                ("around x1.5 speed up", 1.5), 
                                ("around x1.75 speed up", 1.75), 
                                ("around x2 speed up", 2.0), 
                                ("around x2.25 speed up", 2.25), 
                                ("around x2.5 speed up", 2.5), 
                            ],
                            value=float(ui_defaults["tea_cache"]),
                            visible=True,
                            label="Tea Cache Global Acceleration"
                        )
                        tea_cache_start_step_perc = gr.Slider(0, 100, value=ui_defaults["tea_cache_start_step_perc"], step=1, label="Tea Cache starting moment in % of generation") 

                    gr.Markdown("<B>With Riflex you can generate videos longer than 5s which is the default duration of videos used to train the model</B>")
                    RIFLEx_setting = gr.Dropdown(
                        choices=[
                            ("Auto (ON if Video longer than 5s)", 0),
                            ("Always ON", 1), 
                            ("Always OFF", 2), 
                        ],
                        value=ui_defaults["RIFLEx_setting"],
                        label="RIFLEx positional embedding to generate long video"
                    )
                    with gr.Row():
                        gr.Markdown("<B>Experimental: Skip Layer Guidance, should improve video quality</B>")
                    with gr.Row():
                        slg_switch = gr.Dropdown(
                            choices=[
                                ("OFF", 0),
                                ("ON", 1), 
                            ],
                            value=ui_defaults["slg_switch"],
                            visible=True,
                            scale = 1,
                            label="Skip Layer guidance"
                        )
                        slg_layers = gr.Dropdown(
                            choices=[
                                (str(i), i ) for i in range(40)
                            ],
                            value=ui_defaults["slg_layers"],
                            multiselect= True,
                            label="Skip Layers",
                            scale= 3
                        )
                    with gr.Row():
                        slg_start_perc = gr.Slider(0, 100, value=ui_defaults["slg_start_perc"], step=1, label="Denoising Steps % start") 
                        slg_end_perc = gr.Slider(0, 100, value=ui_defaults["slg_end_perc"], step=1, label="Denoising Steps % end") 

                    with gr.Row():
                        gr.Markdown("<B>Experimental: Classifier-Free Guidance Zero Star, better adherence to Text Prompt")
                    with gr.Row():
                        cfg_star_switch = gr.Dropdown(
                            choices=[
                                ("OFF", 0),
                                ("ON", 1), 
                            ],
                            value=ui_defaults.get("cfg_star_switch",0),
                            visible=True,
                            scale = 1,
                            label="CFG Star"
                        )
                        with gr.Row():
                            cfg_zero_step = gr.Slider(-1, 39, value=ui_defaults.get("cfg_zero_step",-1), step=1, label="CFG Zero below this Layer (Extra Process)") 

                    with gr.Row():
                        save_settings_btn = gr.Button("Set Settings as Default")
            show_advanced.change(fn=switch_advanced, inputs=[state, show_advanced, lset_name], outputs=[advanced_row, preset_buttons_rows, refresh_lora_btn, refresh2_row ,lset_name ]).then(
                fn=switch_prompt_type, inputs = [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars], outputs = [wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars])
        with gr.Column():
            output = gr.Gallery(
                    label="Generated videos", show_label=False, elem_id="gallery"
                , columns=[3], rows=[1], object_fit="contain", height=450, selected_index=0, interactive= False)
            generate_btn = gr.Button("Generate")
            queue_df = gr.DataFrame(
                headers=["Status", "Completed", "Progress", "Steps", "Time", "Prompt", "Start", "End", "", "", ""],
                datatype=["str", "str", "str", "str", "str", "markdown", "markdown", "markdown", "str", "str", "str"],
                interactive=False,
                col_count=(11, "fixed"),
                wrap=True,
                value=update_queue_data,
                every=1,
                elem_id="queue_df"
            )
            def handle_selection(evt: gr.SelectData):
                if evt.index is None:
                     return gr.update(), gr.update(), gr.update(visible=False)
                row_index, col_index = evt.index
                cell_value = None
                if col_index in [8, 9, 10]:
                     if col_index == 8: cell_value = "↑"
                     elif col_index == 9: cell_value = "↓"
                     elif col_index == 10: cell_value = "✖"
                if col_index == 8:
                     new_df_data = move_up([row_index])
                     return new_df_data, gr.update(), gr.update(visible=False)
                elif col_index == 9:
                     new_df_data = move_down([row_index])
                     return new_df_data, gr.update(), gr.update(visible=False)
                elif col_index == 10:
                     new_df_data = remove_task([row_index])
                     return new_df_data, gr.update(), gr.update(visible=False)
                start_img_col_idx = 6
                end_img_col_idx = 7
                image_data_to_show = None
                if col_index == start_img_col_idx:
                    with lock:
                        if row_index < len(queue):
                            image_data_to_show = queue[row_index].get('start_image_data')
                elif col_index == end_img_col_idx:
                     with lock:
                        if row_index < len(queue):
                            image_data_to_show = queue[row_index].get('end_image_data')

                if image_data_to_show:
                    return gr.update(), gr.update(value=image_data_to_show), gr.update(visible=True)
                else:
                    return gr.update(), gr.update(), gr.update(visible=False)
            def refresh_gallery_on_trigger(state):
                if(state.get("update_gallery", False)):
                    state['update_gallery'] = False
                    return gr.update(value=state.get("file_list", []))
            selected_indices = gr.State([])
            queue_df.select(
                fn=handle_selection,
                inputs=None,
                outputs=[queue_df, modal_image_display, modal_container],
            )
            gallery_update_trigger.change(
                fn=refresh_gallery_on_trigger,
                inputs=[state],
                outputs=[output]
            )
            queue_df.change(
                fn=refresh_gallery,
                inputs=[state],
                outputs=[gallery_update_trigger]
            )
        save_settings_btn.click( fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
            save_settings, inputs = [state, prompt, image_prompt_type_radio, video_length, resolution, num_inference_steps, seed, repeat_generation, multi_images_gen_type, guidance_scale, flow_shift, negative_prompt, 
                                                         loras_choices, loras_mult_choices, tea_cache_setting, tea_cache_start_step_perc, RIFLEx_setting, slg_switch, slg_layers,
                                                         slg_start_perc, slg_end_perc, cfg_star_switch, cfg_zero_step  ], outputs = [])
        save_lset_btn.click(validate_save_lset, inputs=[lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
        confirm_save_lset_btn.click(fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
        save_lset, inputs=[state, lset_name, loras_choices, loras_mult_choices, prompt, save_lset_prompt_drop], outputs=[lset_name, apply_lset_btn,refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
        delete_lset_btn.click(validate_delete_lset, inputs=[lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
        confirm_delete_lset_btn.click(delete_lset, inputs=[state, lset_name], outputs=[lset_name, apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
        cancel_lset_btn.click(cancel_lset, inputs=[], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_delete_lset_btn,confirm_save_lset_btn, cancel_lset_btn,save_lset_prompt_drop ])
        apply_lset_btn.click(apply_lset, inputs=[state, wizard_prompt_activated_var, lset_name,loras_choices, loras_mult_choices, prompt], outputs=[wizard_prompt_activated_var, loras_choices, loras_mult_choices, prompt]).then(
            fn = fill_wizard_prompt, inputs = [state, wizard_prompt_activated_var, prompt, wizard_prompt], outputs = [ wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars]
        )
        refresh_lora_btn.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
        refresh_lora_btn2.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
        download_loras_btn.click(fn=download_loras, inputs=[], outputs=[download_status_row, download_status, presets_column, loras_column]).then(fn=refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
        output.select(select_video, state, None )

        generate_btn.click(
            fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]
        ).then(
            fn=process_prompt_and_add_tasks,
            inputs=[
                prompt,
                negative_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                multi_images_gen_type,
                tea_cache_setting,
                tea_cache_start_step_perc,
                loras_choices,
                loras_mult_choices,
                image_prompt_type_radio,
                image_to_continue,
                image_to_end,
                video_to_continue,
                max_frames,
                RIFLEx_setting,
                slg_switch, 
                slg_layers,
                slg_start_perc,
                slg_end_perc,
                cfg_star_switch,
                cfg_zero_step,
                state,
                gr.State(image2video)
            ],
            outputs=queue_df
        )
        close_modal_button.click(
            lambda: gr.update(visible=False),
            inputs=[],
            outputs=[modal_container]
        )
    return loras_column, loras_choices, presets_column, lset_name, header, state

def generate_configuration_tab():
    state_dict = {}
    state = gr.State(state_dict)
    gr.Markdown("Please click Apply Changes at the bottom so that the changes are effective. Some choices below may be locked if the app has been launched by specifying a config preset.")
    with gr.Column():
        index = transformer_choices_t2v.index(transformer_filename_t2v)
        index = 0 if index ==0 else index
        transformer_t2v_choice = gr.Dropdown(
            choices=[
                ("WAN 2.1 1.3B Text to Video 16 bits (recommended)- the small model for fast generations with low VRAM requirements", 0),
                ("WAN 2.1 14B Text to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 1),
                ("WAN 2.1 14B Text to Video quantized to 8 bits (recommended) - the default engine but quantized", 2),
            ],
            value= index,
            label="Transformer model for Text to Video",
            interactive= not lock_ui_transformer,   
            visible=True #not use_image2video
         )
        index = transformer_choices_i2v.index(transformer_filename_i2v)
        index = 0 if index ==0 else index
        transformer_i2v_choice = gr.Dropdown(
            choices=[
                ("WAN 2.1 - 480p 14B Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                ("WAN 2.1 - 480p 14B Image to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                ("WAN 2.1 - 720p 14B Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 2),
                ("WAN 2.1 - 720p 14B Image to Video quantized to 8 bits - the default engine but quantized", 3),
                ("WAN 2.1 - Fun InP 1.3B 16 bits - the small model for fast generations with low VRAM requirements", 4),
                ("WAN 2.1 - Fun InP 14B 16 bits - Fun InP version in its original glory, offers a slightly better image quality but slower and requires more RAM", 5),
                ("WAN 2.1 - Fun InP 14B quantized to 8 bits - quantized Fun InP version", 6),
            ],
            value= index,
            label="Transformer model for Image to Video",
            interactive= not lock_ui_transformer,
            visible = True # use_image2video,
         )
        index = text_encoder_choices.index(text_encoder_filename)
        index = 0 if index ==0 else index
        text_encoder_choice = gr.Dropdown(
            choices=[
                ("UMT5 XXL 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                ("UMT5 XXL quantized to 8 bits - quantized text encoder, slightly worse quality but uses less RAM", 1),
            ],
            value= index,
            label="Text Encoder model"
         )
        save_path_choice = gr.Textbox(
            label="Output Folder for Generated Videos",
            value=server_config.get("save_path", save_path)
        )
        def check(mode): 
            if not mode in attention_modes_installed:
                return " (NOT INSTALLED)"
            elif not mode in attention_modes_supported:
                return " (NOT SUPPORTED)"
            else:
                return ""
        attention_choice = gr.Dropdown(
            choices=[
                ("Auto : pick sage2 > sage > sdpa depending on what is installed", "auto"),
                ("Scale Dot Product Attention: default, always available", "sdpa"),
                ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                # ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                ("Sage" + check("sage")+ ": 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                ("Sage2" + check("sage2")+ ": 40% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage2"),
            ],
            value= attention_mode,
            label="Attention Type",
            interactive= not lock_ui_attention
         )
        gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
        compile_choice = gr.Dropdown(
            choices=[
                ("ON: works only on Linux / WSL", "transformer"),
                ("OFF: no other choice if you have Windows without using WSL", "" ),
            ],
            value= compile,
            label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)",
            interactive= not lock_ui_compile
         )              
        vae_config_choice = gr.Dropdown(
            choices=[
        ("Auto", 0),
        ("Disabled (faster but may require up to 22 GB of VRAM)", 1),
        ("256 x 256 : If at least 8 GB of VRAM", 2),
        ("128 x 128 : If at least 6 GB of VRAM", 3),
            ],
            value= vae_config,
            label="VAE Tiling - reduce the high VRAM requirements for VAE decoding and VAE encoding (if enabled it will be slower)"
         )
        boost_choice = gr.Dropdown(
            choices=[
                # ("Auto (ON if Video longer than 5s)", 0),
                ("ON", 1), 
                ("OFF", 2), 
            ],
            value=boost,
            label="Boost: Give a 10% speed speedup without losing quality at the cost of a litle VRAM (up to 1GB for max frames and resolution)"
        )
        profile_choice = gr.Dropdown(
            choices=[
        ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for short videos a RTX 3090 / RTX 4090", 1),
        ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
        ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
        ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
        ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
            ],
            value= profile,
            label="Profile (for power users only, not needed to change it)"
         )
        default_ui_choice = gr.Dropdown(
            choices=[
                ("Text to Video", "t2v"),
                ("Image to Video", "i2v"),
            ],
            value= default_ui,
            label="Default mode when launching the App if not '--t2v' ot '--i2v' switch is specified when launching the server ",
         )                
        metadata_choice = gr.Dropdown(
            choices=[
                ("Export JSON files", "json"),
                ("Add metadata to video", "metadata"),
                ("Neither", "none")
            ],
            value=server_config.get("metadata_type", "metadata"),
            label="Metadata Handling"
        )
        reload_choice = gr.Dropdown(
            choices=[
                ("When changing tabs", 1), 
                ("When pressing generate", 2), 
            ],
            value=server_config.get("reload_model",2),
            label="Reload model"
        )

        clear_file_list_choice = gr.Dropdown(
            choices=[
                ("None", 0),
                ("Keep the last video", 1),
                ("Keep the last 5 videos", 5),
                ("Keep the last 10 videos", 10),
                ("Keep the last 20 videos", 20),
                ("Keep the last 30 videos", 30),
            ],
            value=server_config.get("clear_file_list", 0),
            label="Keep Previously Generated Videos when starting a Generation Batch"
        )

        
        msg = gr.Markdown()            
        apply_btn  = gr.Button("Apply Changes")
        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    save_path_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    vae_config_choice,
                    metadata_choice,
                    default_ui_choice,
                    boost_choice,
                    clear_file_list_choice,
                    reload_choice,
                ],
                outputs= msg
        )

def generate_about_tab():
    gr.Markdown("<H2>Wan2.1GP - Wan 2.1 model for the GPU Poor by <B>DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>GitHub</A>)</H2>")
    gr.Markdown("Original Wan 2.1 Model by <B>Alibaba</B> (<A HREF='https://github.com/Wan-Video/Wan2.1'>GitHub</A>)")
    gr.Markdown("Many thanks to:")
    gr.Markdown("- <B>Cocktail Peanuts</B> : QA and simple installation via Pinokio.computer")
    gr.Markdown("- <B>AmericanPresidentJimmyCarter</B> : added original support for Skip Layer Guidance")
    gr.Markdown("- <B>Tophness</B> : created multi tabs framework")
    gr.Markdown("- <B>Remade_AI</B> : for creating their awesome Loras collection")
    

def on_tab_select(t2v_state, i2v_state, evt: gr.SelectData):
    global lora_model_filename, use_image2video

    t2v_header = generate_header(transformer_filename_t2v, compile, attention_mode)
    i2v_header = generate_header(transformer_filename_i2v, compile, attention_mode)

    new_t2v = evt.index == 0
    new_i2v = evt.index == 1
    use_image2video = new_i2v

    if(server_config.get("reload_model",2) == 1):
        with lock:
            queue_empty = len(queue) == 0    
        if queue_empty:
            global wan_model, offloadobj
            if wan_model is not None:
                if offloadobj is not None:
                    offloadobj.release()
                offloadobj = None
                wan_model = None
                gc.collect()
                torch.cuda.empty_cache()
            wan_model, offloadobj, trans = load_models(use_image2video)
            del trans

    if new_t2v or new_i2v:
        state = i2v_state if new_i2v else t2v_state
        lora_model_filename = state["loras_model"]
        model_filename = model_needed(new_i2v)
        if ("1.3B" in model_filename and not "1.3B" in lora_model_filename or "14B" in model_filename and not "14B" in lora_model_filename):
            lora_dir = get_lora_dir(new_i2v)
            loras, loras_names, loras_presets, _, _, _, _ = setup_loras(new_i2v, None,  lora_dir, lora_preselected_preset, None)
            state["loras"] = loras
            state["loras_names"] = loras_names
            state["loras_presets"] = loras_presets
            state["loras_model"] = model_filename

            advanced =  state["advanced"]
            new_loras_choices = [(name, str(i)) for i, name in enumerate(loras_names)]
            lset_choices = [(preset, preset) for preset in loras_presets] + [(get_new_preset_msg(advanced), "")]
            visible = len(loras_names)>0
            if new_t2v:
                return [
                    gr.Column(visible= visible),
                    gr.Dropdown(choices=new_loras_choices,  visible=visible, value=[]),
                    gr.Column(visible= visible),
                    gr.Dropdown(choices=lset_choices, value=get_new_preset_msg(advanced), visible=visible),
                    t2v_header,
                    gr.Column(),
                    gr.Dropdown(),
                    gr.Column(),
                    gr.Dropdown(),
                    i2v_header,            
                ]
            else:
                return [
                    gr.Column(),
                    gr.Dropdown(),
                    gr.Column(),
                    gr.Dropdown(),
                    t2v_header,
                    gr.Column(visible= visible),
                    gr.Dropdown(choices=new_loras_choices,  visible=visible, value=[]),
                    gr.Column(visible= visible),
                    gr.Dropdown(choices=lset_choices, value=get_new_preset_msg(advanced), visible=visible),
                    i2v_header,            
                ]

    return [gr.Column(), gr.Dropdown(), gr.Column(), gr.Dropdown(), t2v_header,
            gr.Column(), gr.Dropdown(), gr.Column(), gr.Dropdown(), i2v_header]


def create_demo():
    css = """
        .title-with-lines {
            display: flex;
            align-items: center;
            margin: 30px 0;
        }
        .line {
            flex-grow: 1;
            height: 1px;
            background-color: #333;
        }
        h2 {
            margin: 0 20px;
            white-space: nowrap;
        }
        .queue-item {
            border: 1px solid #ccc;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .current {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
        }
        .task-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .progress-container {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #007bff;
            transition: width 0.3s ease;
        }
        .task-details {
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        .task-prompt {
            font-size: 0.8em;
            color: #868e96;
            margin-top: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        #queue_df td:nth-child(-n+6) {
            cursor: default !important;
            pointer-events: none;
        }
        #queue_df th {
            pointer-events: none;
        }
        #queue_df table {
            overflow: hidden !important;
        }
        #queue_df::-webkit-scrollbar {
            display: none !important;
        }
        #queue_df {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        #queue_df td:nth-child(1) {
            width: 100px;
        }
        #queue_df td:nth-child(7) img,
        #queue_df td:nth-child(8) img,
            max-width: 50px;
            max-height: 50px;
            object-fit: contain;
            display: block;
            margin: auto;            
            cursor: pointer;
            text-align: center;
        }
        #queue_df td:nth-child(9),
        #queue_df td:nth-child(10),
        #queue_df td:nth-child(11) {            
            width: 60px;
            padding: 2px !important;
            cursor: pointer;
            text-align: center;
            font-weight: bold;
        }
        #queue_df td:nth-child(7):hover,
        #queue_df td:nth-child(8):hover,
        #queue_df td:nth-child(9):hover,
        #queue_df td:nth-child(10):hover,
        #queue_df td:nth-child(11):hover {
            background-color: #e0e0e0;
        }
        #image-modal-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
            box-sizing: border-box;
        }
        #image-modal-container > div {
             background-color: white;
             padding: 15px;
             border-radius: 8px;
             max-width: 90%;
             max-height: 90%;
             overflow: auto;
             position: relative;
             display: flex;
             flex-direction: column;
        }
         #image-modal-container img {
             max-width: 100%;
             max-height: 80vh;
             object-fit: contain;
             margin-top: 10px;
         }
         #image-modal-close-button-row {
             display: flex;
             justify-content: flex-end;
         }
         #image-modal-close-button-row button {
            cursor: pointer;
         }
    """
    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate", text_size="md")) as demo:
        gr.Markdown("<div align=center><H1>Wan 2.1<SUP>GP</SUP> v3.3 <FONT SIZE=4>by <I>DeepBeepMeep</I></FONT> <FONT SIZE=3> (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A>)</FONT SIZE=3></H1></div>")
        gr.Markdown("<FONT SIZE=3>Welcome to Wan 2.1GP a super fast and low VRAM AI Video Generator !</FONT>")
        
        with gr.Accordion("Click here for some Info on how to use Wan2GP", open = False):
            gr.Markdown("The VRAM requirements will depend greatly of the resolution and the duration of the video, for instance :")
            gr.Markdown("- 848 x 480 with a 14B model: 80 frames (5s) : 8 GB of VRAM")
            gr.Markdown("- 848 x 480 with the 1.3B model: 80 frames (5s) : 5 GB of VRAM")
            gr.Markdown("- 1280 x 720 with a 14B model: 80 frames (5s): 11 GB of VRAM")
            gr.Markdown("It is not recommmended to generate a video longer than 8s (128 frames) even if there is still some VRAM left as some artifacts may appear")
            gr.Markdown("Please note that if your turn on compilation, the first denoising step of the first video generation will be slow due to the compilation. Therefore all your tests should be done with compilation turned off.")


        with gr.Tabs(selected="i2v" if use_image2video else "t2v") as main_tabs:
            with gr.Tab("Text To Video", id="t2v") as t2v_tab:
                t2v_loras_column, t2v_loras_choices, t2v_presets_column, t2v_lset_name, t2v_header, t2v_state = generate_video_tab()
            with gr.Tab("Image To Video", id="i2v") as i2v_tab:
                i2v_loras_column, i2v_loras_choices, i2v_presets_column, i2v_lset_name, i2v_header, i2v_state = generate_video_tab(True)
            if not args.lock_config:
                with gr.Tab("Configuration"):
                    generate_configuration_tab()
            with gr.Tab("About"):
                generate_about_tab()
        main_tabs.select(
            fn=on_tab_select,
            inputs=[t2v_state, i2v_state],
            outputs=[
                t2v_loras_column, t2v_loras_choices, t2v_presets_column, t2v_lset_name, t2v_header,
                i2v_loras_column, i2v_loras_choices, i2v_presets_column, i2v_lset_name, i2v_header
            ]
        )
        return demo

if __name__ == "__main__":
    threading.Thread(target=runner, daemon=True).start()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))
    server_name = args.server_name
    if args.listen:
        server_name = "0.0.0.0"
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")      
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)
    demo.launch(server_name=server_name, server_port=server_port, share=args.share, allowed_paths=[save_path])