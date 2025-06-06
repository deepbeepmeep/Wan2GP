import os
import time
import argparse
import json
import torch
import traceback
import gc
import random

# These imports rely on your existing code structure
# They must match the location of your WAN code, etc.
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.modules.attention import get_attention_modes
from wan.utils.utils import cache_video
from mmgp import offload, safetensors2, profile_type

try:
    import triton
except ImportError:
    pass

DATA_DIR = "ckpts"

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def sanitize_file_name(file_name):
    """Clean up file name from special chars."""
    return (
        file_name.replace("/", "")
        .replace("\\", "")
        .replace(":", "")
        .replace("|", "")
        .replace("?", "")
        .replace("<", "")
        .replace(">", "")
        .replace('"', "")
    )

def extract_preset(lset_name, lora_dir, loras):
    """
    Load a .lset JSON that lists the LoRA files to apply, plus multipliers
    and possibly a suggested prompt prefix.
    """
    lset_name = sanitize_file_name(lset_name)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset")
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name)

    if not os.path.isfile(lset_name_filename):
        raise ValueError(f"Preset '{lset_name}' not found in {lora_dir}")

    with open(lset_name_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    lset = json.loads(text)

    loras_choices_files = lset["loras"]
    loras_choices = []
    missing_loras = []
    for lora_file in loras_choices_files:
        # Build absolute path and see if it is in loras
        full_lora_path = os.path.join(lora_dir, lora_file)
        if full_lora_path in loras:
            idx = loras.index(full_lora_path)
            loras_choices.append(str(idx))
        else:
            missing_loras.append(lora_file)

    if len(missing_loras) > 0:
        missing_list = ", ".join(missing_loras)
        raise ValueError(f"Missing LoRA files for preset: {missing_list}")

    loras_mult_choices = lset["loras_mult"]
    prompt_prefix = lset.get("prompt", "")
    full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt_prefix, full_prompt

def get_attention_mode(args_attention, installed_modes):
    """
    Decide which attention mode to use: either the user choice or auto fallback.
    """
    if args_attention == "auto":
        for candidate in ["sage2", "sage", "sdpa"]:
            if candidate in installed_modes:
                return candidate
        return "sdpa"  # last fallback
    elif args_attention in installed_modes:
        return args_attention
    else:
        raise ValueError(
            f"Requested attention mode '{args_attention}' not installed. "
            f"Installed modes: {installed_modes}"
        )

def load_i2v_model(model_filename, text_encoder_filename, is_720p):
    """
    Load the i2v model with a specific size config and text encoder.
    """
    if is_720p:
        print("Loading 14B-720p i2v model ...")
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=DATA_DIR,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )
    else:
        print("Loading 14B-480p i2v model ...")
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=DATA_DIR,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )
    # Pipe structure
    pipe = {
        "transformer": wan_model.model,
        "text_encoder": wan_model.text_encoder.model,
        "text_encoder_2": wan_model.clip.model,
        "vae": wan_model.vae.model
    }
    return wan_model, pipe

def setup_loras(pipe, lora_dir, lora_preset, num_inference_steps):
    """
    Load loras from a directory, optionally apply a preset.
    """
    from pathlib import Path
    import glob

    if not lora_dir or not Path(lora_dir).is_dir():
        print("No valid --lora-dir provided or directory doesn't exist, skipping LoRA setup.")
        return [], [], [], "", "", False

    # Gather LoRA files
    loras = sorted(
        glob.glob(os.path.join(lora_dir, "*.sft"))
        + glob.glob(os.path.join(lora_dir, "*.safetensors"))
    )
    loras_names = [Path(x).stem for x in loras]

    # Offload them with no activation
    offload.load_loras_into_model(pipe["transformer"], loras, activate_all_loras=False)

    # If user gave a preset, apply it
    default_loras_choices = []
    default_loras_multis_str = ""
    default_prompt_prefix = ""
    preset_applied_full_prompt = False
    if lora_preset:
        loras_choices, loras_mult, prefix, full_prompt = extract_preset(lora_preset, lora_dir, loras)
        default_loras_choices = loras_choices
        # If user stored loras_mult as a list or string in JSON, unify that to str
        if isinstance(loras_mult, list):
            # Just store them in a single line
            default_loras_multis_str = " ".join([str(x) for x in loras_mult])
        else:
            default_loras_multis_str = str(loras_mult)
        default_prompt_prefix = prefix
        preset_applied_full_prompt = full_prompt

    return (
        loras,
        loras_names,
        default_loras_choices,
        default_loras_multis_str,
        default_prompt_prefix,
        preset_applied_full_prompt
    )

def parse_loras_and_activate(
    transformer,
    loras,
    loras_choices,
    loras_mult_str,
    num_inference_steps
):
    """
    Activate the chosen LoRAs with multipliers over the pipeline's transformer.
    Supports stepwise expansions (like "0.5,0.8" for partial steps).
    """
    if not loras or not loras_choices:
        # no LoRAs selected
        return

    # Handle multipliers
    def is_float_or_comma_list(x):
        """
        Example: "0.5", or "0.8,1.0", etc. is valid.
        """
        if not x:
            return False
        for chunk in x.split(","):
            try:
                float(chunk.strip())
            except ValueError:
                return False
        return True

    # Convert multiline or spaced lines to a single list
    lines = [
        line.strip()
        for line in loras_mult_str.replace("\r", "\n").split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    # Now combine them by space
    joined_line = " ".join(lines)  # "1.0 2.0,3.0"
    if not joined_line.strip():
        multipliers = []
    else:
        multipliers = joined_line.split(" ")

    # Expand each item
    final_multipliers = []
    for mult in multipliers:
        mult = mult.strip()
        if not mult:
            continue
        if is_float_or_comma_list(mult):
            # Could be "0.7" or "0.5,0.6"
            if "," in mult:
                # expand over steps
                chunk_vals = [float(x.strip()) for x in mult.split(",")]
                expanded = expand_list_over_steps(chunk_vals, num_inference_steps)
                final_multipliers.append(expanded)
            else:
                final_multipliers.append(float(mult))
        else:
            raise ValueError(f"Invalid LoRA multiplier: '{mult}'")

    # If fewer multipliers than chosen LoRAs => pad with 1.0
    needed = len(loras_choices) - len(final_multipliers)
    if needed > 0:
        final_multipliers += [1.0]*needed

    # Actually activate them
    offload.activate_loras(transformer, loras_choices, final_multipliers)

def expand_list_over_steps(short_list, num_steps):
    """
    If user gave (0.5, 0.8) for example, expand them over `num_steps`.
    The expansion is simply linear slice across steps.
    """
    result = []
    inc = len(short_list) / float(num_steps)
    idxf = 0.0
    for _ in range(num_steps):
        value = short_list[int(idxf)]
        result.append(value)
        idxf += inc
    return result

def download_models_if_needed(transformer_filename_i2v, text_encoder_filename, local_folder=DATA_DIR):
    """
    Checks if all required WAN 2.1 i2v files exist locally under 'ckpts/'.
    If not, downloads them from a Hugging Face Hub repo.
    Adjust the 'repo_id' and needed files as appropriate.
    """
    import os
    from pathlib import Path

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for automatic model download. "
            "Please install it via `pip install huggingface_hub`."
        ) from e

    # Identify just the filename portion for each path
    def basename(path_str):
        return os.path.basename(path_str)

    repo_id = "DeepBeepMeep/Wan2.1"
    target_root = local_folder

    # You can customize this list as needed for i2v usage.
    # At minimum you need:
    #   1) The requested i2v transformer file
    #   2) The requested text encoder file
    #   3) VAE file
    #   4) The open-clip xlm-roberta-large weights
    #
    # If your i2v config references additional files, add them here.
    needed_files = [
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        basename(text_encoder_filename),
        basename(transformer_filename_i2v),
    ]

    # The original script also downloads an entire "xlm-roberta-large" folder
    # via snapshot_download. If you require that for your pipeline,
    # you can add it here, for example:
    subfolder_name = "xlm-roberta-large"
    if not Path(os.path.join(target_root, subfolder_name)).exists():
        snapshot_download(repo_id=repo_id, allow_patterns=subfolder_name + "/*", local_dir=target_root)

    for filename in needed_files:
        local_path = os.path.join(target_root, filename)
        if not os.path.isfile(local_path):
            print(f"File '{filename}' not found locally. Downloading from {repo_id} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_root
            )
        else:
            # Already present
            pass

    print("All required i2v files are present.")


# --------------------------------------------------
# ARGUMENT PARSER
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image-to-Video inference using WAN 2.1 i2v"
    )
    # Model + Tools
    parser.add_argument(
        "--quantize-transformer",
        action="store_true",
        help="Use on-the-fly transformer quantization"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 compile for the transformer"
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="auto",
        help="Which attention to use: auto, sdpa, sage, sage2, flash"
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=4,
        help="Memory usage profile number [1..5]; see original script or use 2 if you have low VRAM"
    )
    parser.add_argument(
        "--preload",
        type=int,
        default=0,
        help="Megabytes of the diffusion model to preload in VRAM (only used in some profiles)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level [0..5]"
    )

    # i2v Model
    parser.add_argument(
        "--transformer-file",
        type=str,
        default=f"{DATA_DIR}/wan2.1_image2video_480p_14B_quanto_int8.safetensors",
        help="Which i2v model to load"
    )
    parser.add_argument(
        "--text-encoder-file",
        type=str,
        default=f"{DATA_DIR}/models_t5_umt5-xxl-enc-quanto_int8.safetensors",
        help="Which text encoder to use"
    )

    # LoRA
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="",
        help="Path to a directory containing i2v LoRAs"
    )
    parser.add_argument(
        "--lora-preset",
        type=str,
        default="",
        help="A .lset preset name in the lora_dir to auto-apply"
    )

    # Generation Options
    parser.add_argument("--prompt", type=str, default=None, required=True, help="Prompt for generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--resolution", type=str, default="832x480", help="WxH")
    parser.add_argument("--frames", type=int, default=64, help="Number of frames (16=1s if fps=16). Must be multiple of 4 +/- 1 in WAN.")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--flow-shift", type=float, default=3.0, help="Flow shift parameter. Generally 3.0 for 480p, 5.0 for 720p.")
    parser.add_argument("--riflex", action="store_true", help="Enable RIFLEx for longer videos")
    parser.add_argument("--teacache", type=float, default=0.25, help="TeaCache multiplier, e.g. 0.5, 2.0, etc.")
    parser.add_argument("--teacache-start", type=float, default=0.1, help="Teacache start step percentage [0..100]")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. -1 means random each time.")
    parser.add_argument("--slg-layers", type=str, default=None, help="Which layers to use for skip layer guidance")
    parser.add_argument("--slg-start", type=float, default=0.0, help="Percentage in to start SLG")
    parser.add_argument("--slg-end", type=float, default=1.0, help="Percentage in to end SLG")

    # LoRA usage
    parser.add_argument("--loras-choices", type=str, default="", help="Comma-separated list of chosen LoRA indices or preset names to load. Usually you only use the preset.")
    parser.add_argument("--loras-mult", type=str, default="", help="Multipliers for each chosen LoRA. Example: '1.0 1.2,1.3' etc.")

    # Input
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        required=True,
        help="Path to an input image (or multiple)."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.mp4",
        help="Where to save the resulting video."
    )

    return parser.parse_args()

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    args = parse_args()

    # Setup environment
    offload.default_verboseLevel = args.verbose
    installed_attn_modes = get_attention_modes()

    # Decide attention
    chosen_attention = get_attention_mode(args.attention, installed_attn_modes)
    offload.shared_state["_attention"] = chosen_attention

    # Determine i2v resolution format
    if "720" in args.transformer_file:
        is_720p = True
    else:
        is_720p = False

    # Make sure we have the needed models locally
    download_models_if_needed(args.transformer_file, args.text_encoder_file)

    # Load i2v
    wan_model, pipe = load_i2v_model(
        model_filename=args.transformer_file,
        text_encoder_filename=args.text_encoder_file,
        is_720p=is_720p
    )
    wan_model._interrupt = False

    # Offload / profile
    # e.g. for your script:  offload.profile(pipe, profile_no=args.profile, compile=..., quantizeTransformer=...)
    # pass the budgets if you want, etc.
    kwargs = {}
    if args.profile == 2 or args.profile == 4:
        # preload is in MB
        if args.preload == 0:
            budgets = {"transformer": 100, "text_encoder": 100, "*": 1000}
        else:
            budgets = {"transformer": args.preload, "text_encoder": 100, "*": 1000}
        kwargs["budgets"] = budgets
    elif args.profile == 3:
        kwargs["budgets"] = {"*": "70%"}

    compile_choice = "transformer" if args.compile else ""
    # Create the offload object
    offloadobj = offload.profile(
        pipe,
        profile_no=args.profile,
        compile=compile_choice,
        quantizeTransformer=args.quantize_transformer,
        **kwargs
    )

    # If user wants to use LoRAs
    (
        loras,
        loras_names,
        default_loras_choices,
        default_loras_multis_str,
        preset_prompt_prefix,
        preset_full_prompt
    ) = setup_loras(pipe, args.lora_dir, args.lora_preset, args.steps)

    # Combine user prompt with preset prompt if the preset indicates so
    if preset_prompt_prefix:
        if preset_full_prompt:
            # Full override
            user_prompt = preset_prompt_prefix
        else:
            # Just prefix
            user_prompt = preset_prompt_prefix + "\n" + args.prompt
    else:
        user_prompt = args.prompt

    # Actually parse user LoRA choices if they did not rely purely on the preset
    if args.loras_choices:
        # If user gave e.g. "0,1", we treat that as new additions
        lora_choice_list = [x.strip() for x in args.loras_choices.split(",")]
    else:
        # Use the defaults from the preset
        lora_choice_list = default_loras_choices

    # Activate them
    parse_loras_and_activate(
        pipe["transformer"], loras, lora_choice_list, args.loras_mult or default_loras_multis_str, args.steps
    )

    # Negative prompt
    negative_prompt = args.negative_prompt or ""

    # Sanity check resolution
    if "*" in args.resolution.lower():
        print("ERROR: resolution must be e.g. 832x480 not '832*480'. Fixing it.")
        resolution_str = args.resolution.lower().replace("*", "x")
    else:
        resolution_str = args.resolution

    try:
        width, height = [int(x) for x in resolution_str.split("x")]
    except:
        raise ValueError(f"Invalid resolution: '{resolution_str}'")

    # Parse slg_layers from comma-separated string to a Python list of ints (or None if not provided)
    if args.slg_layers:
        slg_list = [int(x) for x in args.slg_layers.split(",")]
    else:
        slg_list = None

    # Additional checks (from your original code).
    if "480p" in args.transformer_file:
        # Then we cannot exceed certain area for 480p model
        if width * height > 832*480:
            raise ValueError("You must use the 720p i2v model to generate bigger than 832x480.")
    # etc.

    # Handle random seed
    if args.seed < 0:
        args.seed = random.randint(0, 999999999)
    print(f"Using seed={args.seed}")

    # Setup tea cache if needed
    trans = wan_model.model
    trans.enable_teacache = (args.teacache > 0)
    if trans.enable_teacache:
        if "480p" in args.transformer_file:
            # example from your code
            trans.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
        elif "720p" in args.transformer_file:
            trans.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
        else:
            raise ValueError("Teacache not supported for this model variant")

    # Attempt generation
    print("Starting generation ...")
    start_time = time.time()

    # Read the input image
    if not os.path.isfile(args.input_image):
        raise ValueError(f"Input image does not exist: {args.input_image}")

    from PIL import Image
    input_img = Image.open(args.input_image).convert("RGB")

    # Possibly load more than one image if you want "multiple images" – but here we'll just do single for demonstration

    # Define the generation call
    #  - frames => must be multiple of 4 plus 1 as per original script's note, e.g. 81, 65, ...
    #    You can correct to that if needed:
    frame_count = (args.frames // 4)*4 + 1  # ensures it's 4*N+1
    # RIFLEx
    enable_riflex = args.riflex

    # If teacache => reset counters
    if trans.enable_teacache:
        trans.teacache_counter = 0
        trans.teacache_multiplier = args.teacache
        trans.teacache_start_step = int(args.teacache_start * args.steps / 100.0)
        trans.num_steps = args.steps
        trans.teacache_skipped_steps = 0
        trans.previous_residual_uncond = None
        trans.previous_residual_cond = None

     # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if device_mem_capacity >= 28000:  # 81 frames 720p requires about 28 GB VRAM
        use_vae_config = 1            
    elif device_mem_capacity >= 8000:
        use_vae_config = 2
    else:          
        use_vae_config = 3

    if use_vae_config == 1:
        VAE_tile_size = 0  
    elif use_vae_config == 2:
        VAE_tile_size = 256  
    else: 
        VAE_tile_size = 128  

    print('Using VAE tile size of', VAE_tile_size)

    # Actually run the i2v generation
    try:
        sample_frames = wan_model.generate(
            input_prompt = user_prompt,
            image_start = input_img,
            frame_num=frame_count,
            width=width,
            height=height,
            # max_area=MAX_AREA_CONFIGS[f"{width}*{height}"],  # or you can pass your custom
            shift=args.flow_shift,
            sampling_steps=args.steps,
            guide_scale=args.guidance_scale,
            n_prompt=negative_prompt,
            seed=args.seed,
            offload_model=False,
            callback=None,  # or define your own callback if you want
            enable_RIFLEx=enable_riflex,
            VAE_tile_size=VAE_tile_size,
            joint_pass=slg_list is None,  # set if you want a small speed improvement without SLG
            slg_layers=slg_list,
            slg_start=args.slg_start,
            slg_end=args.slg_end,
        )
    except Exception as e:
        offloadobj.unload_all()
        gc.collect()
        torch.cuda.empty_cache()

        err_str = f"Generation failed with error: {e}"
        # Attempt to detect OOM errors
        s = str(e).lower()
        if any(keyword in s for keyword in ["memory", "cuda", "alloc"]):
            raise RuntimeError("Likely out-of-VRAM or out-of-RAM error. " + err_str)
        else:
            traceback.print_exc()
            raise RuntimeError(err_str)

    # After generation
    offloadobj.unload_all()
    gc.collect()
    torch.cuda.empty_cache()

    if sample_frames is None:
        raise RuntimeError("No frames were returned (maybe generation was aborted or failed).")

    # If teacache was used, we can see how many steps were skipped
    if trans.enable_teacache:
        print(f"TeaCache skipped steps: {trans.teacache_skipped_steps} / {args.steps}")

    # Save result
    sample_frames = sample_frames.cpu()  # shape = c, t, h, w => [3, T, H, W]
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Use the provided helper from your code to store the MP4
    # By default, you used cache_video(tensor=..., save_file=..., fps=16, ...)
    # or you can do your own. We'll do the same for consistency:
    cache_video(
        tensor=sample_frames[None],  # shape => [1, c, T, H, W]
        save_file=args.output_file,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"Done! Output written to {args.output_file}. Generation time: {elapsed_s:.1f} seconds.")

if __name__ == "__main__":
    main()
