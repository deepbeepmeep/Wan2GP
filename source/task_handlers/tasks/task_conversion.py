from source.core.log import headless_logger


# Target megapixel count for auto-scaling img2img input images
IMG2IMG_TARGET_MEGAPIXELS = 1024 * 1024

# Default resolution string for image generation tasks
DEFAULT_IMAGE_RESOLUTION = "1024x1024"
from source.models.model_handlers.qwen_handler import QwenHandler
from source.task_handlers.queue.task_queue import GenerationTask
from source.utils.orchestrator_utils import extract_orchestrator_parameters

def db_task_to_generation_task(db_task_params: dict, task_id: str, task_type: str, wan2gp_path: str, debug_mode: bool = False) -> GenerationTask:
    """
    Convert a database task row to a GenerationTask object for the queue system.
    """
    headless_logger.debug(f"Converting DB task to GenerationTask", task_id=task_id)

    prompt = db_task_params.get("prompt", "")

    # For img2img tasks, empty prompt is acceptable (will use minimal changes)
    # Provide a minimal default prompt to avoid errors
    img2img_task_types = {"z_image_turbo_i2i", "qwen_image_edit", "qwen_image_style", "image_inpaint"}
    if not prompt:
        if task_type in img2img_task_types:
            prompt = " "  # Minimal prompt for img2img
            headless_logger.debug(f"Task {task_id}: Using minimal prompt for img2img task", task_id=task_id)
        else:
            raise ValueError(f"Task {task_id}: prompt is required")
    
    model = db_task_params.get("model")
    if not model:
        from source.task_handlers.tasks.task_types import get_default_model
        model = get_default_model(task_type)
    
    generation_params = {}
    
    param_whitelist = {
        "negative_prompt", "resolution", "video_length", "num_inference_steps",
        "guidance_scale", "seed", "embedded_guidance_scale", "flow_shift",
        "audio_guidance_scale", "repeat_generation", "multi_images_gen_type",
        "guidance2_scale", "guidance3_scale", "guidance_phases", "switch_threshold", "switch_threshold2", "model_switch_phase",
        "video_guide", "video_mask",
        "video_prompt_type", "control_net_weight", "control_net_weight2",
        "keep_frames_video_guide", "video_guide_outpainting", "mask_expand",
        "image_prompt_type", "image_start", "image_end", "image_refs",
        "frames_positions", "image_guide", "image_mask",
        "input_video_strength", "image_refs_strengths",
        "model_mode", "video_source", "keep_frames_video_source",
        "audio_guide", "audio_guide2", "audio_source", "audio_prompt_type", "speakers_locations", "audio_scale",
        "activated_loras", "loras_multipliers", "additional_loras", "loras",
        "tea_cache_setting", "tea_cache_start_step_perc", "RIFLEx_setting",
        "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc",
        "cfg_star_switch", "cfg_zero_step", "prompt_enhancer",
        # Hires fix parameters
        "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_method", "lightning_lora_strength",
        "sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise",
        "sliding_window_discard_last_frames", "latent_noise_mask_strength",
        "vid2vid_init_video", "vid2vid_init_strength",
        "remove_background_images_ref", "temporal_upsampling", "spatial_upsampling",
        "film_grain_intensity", "film_grain_saturation",
        "image_refs_relative_size",
        "output_dir", "custom_output_dir",
        "override_profile",
        "image", "image_url", "mask_url",
        "style_reference_image", "subject_reference_image",
        "style_reference_strength", "subject_strength",
        "subject_description", "in_this_scene",
        "output_format", "enable_base64_output", "enable_sync_mode",
        # Uni3C motion guidance parameters
        "use_uni3c", "uni3c_guide_video", "uni3c_strength",
        "uni3c_start_percent", "uni3c_end_percent",
        "uni3c_keep_on_gpu", "uni3c_frame_policy",
        "uni3c_zero_empty_frames", "uni3c_blackout_last_frame",
        # Image-to-image parameters
        "denoising_strength",
    }
    
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # Layer 1 Uni3C logging - detect whitelist failures early
    if "use_uni3c" in generation_params:
        headless_logger.info(
            f"[UNI3C] Task {task_id}: use_uni3c={generation_params.get('use_uni3c')}, "
            f"guide_video={generation_params.get('uni3c_guide_video', 'NOT_SET')}, "
            f"strength={generation_params.get('uni3c_strength', 'NOT_SET')}"
        )
    elif db_task_params.get("use_uni3c"):
        # CRITICAL: Detect when whitelist is missing the param
        headless_logger.warning(
            f"[UNI3C] Task {task_id}: ⚠️ use_uni3c was in db_task_params but NOT in generation_params! "
            f"Check param_whitelist in task_conversion.py"
        )
    
    # Extract orchestrator parameters
    extracted_params = extract_orchestrator_parameters(db_task_params, task_id)

    if "phase_config" in extracted_params:
        db_task_params["phase_config"] = extracted_params["phase_config"]

    # Copy additional_loras from extracted params if not already in generation_params
    if "additional_loras" in extracted_params and "additional_loras" not in generation_params:
        generation_params["additional_loras"] = extracted_params["additional_loras"]
    
    if "steps" in db_task_params and "num_inference_steps" not in generation_params:
        generation_params["num_inference_steps"] = db_task_params["steps"]
    
    # Note: LoRA resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
    # via TaskConfig/LoRAConfig which detects URLs and downloads them automatically

    # Qwen Task Handlers
    qwen_handler = QwenHandler(
        wan_root=wan2gp_path,
        task_id=task_id)

    if task_type == "qwen_image_edit":
        qwen_handler.handle_qwen_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_hires":
        qwen_handler.handle_qwen_image_hires(db_task_params, generation_params)
    elif task_type == "image_inpaint":
        qwen_handler.handle_image_inpaint(db_task_params, generation_params)
    elif task_type == "annotated_image_edit":
        qwen_handler.handle_annotated_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_style":
        qwen_handler.handle_qwen_image_style(db_task_params, generation_params)
        prompt = generation_params.get("prompt", prompt)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image":
        qwen_handler.handle_qwen_image(db_task_params, generation_params)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image_2512":
        qwen_handler.handle_qwen_image_2512(db_task_params, generation_params)
        model = "qwen_image_2512_20B"
    elif task_type == "z_image_turbo":
        # Z-Image turbo - fast text-to-image generation
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 8)))

        # Resolution handling
        if "resolution" in db_task_params:
            generation_params["resolution"] = db_task_params["resolution"]

        # Override model to use Z-Image (user might pass "z-image" with hyphen)
        model = "z_image"

    elif task_type == "z_image_turbo_i2i":
        # Z-Image turbo img2img - fast image-to-image generation
        import tempfile
        from PIL import Image

        # Get image URL
        image_url = db_task_params.get("image") or db_task_params.get("image_url")
        if not image_url:
            raise ValueError(f"Task {task_id}: 'image' or 'image_url' required for z_image_turbo_i2i")

        # Download image to local file (required for WGP)
        local_image_path = None
        try:
            import requests
            import os
            headless_logger.debug(f"Downloading image for img2img: {image_url}", task_id=task_id)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Save to temp file and keep it for WGP to use
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(response.content)
                local_image_path = tmp_file.name

            headless_logger.info(f"[Z_IMAGE_I2I] Downloaded image to: {local_image_path}", task_id=task_id)

            # Handle resolution - auto-detect from image or use provided
            image_size = db_task_params.get("image_size", "auto")
            if image_size == "auto" or not db_task_params.get("resolution"):
                with Image.open(local_image_path) as img:
                    width, height = img.size

                    # Scale to approximately 1 megapixel (1024x1024) while preserving aspect ratio
                    target_pixels = IMG2IMG_TARGET_MEGAPIXELS
                    current_pixels = width * height

                    if current_pixels > 0:
                        # Calculate scale factor to reach target pixels
                        import math
                        scale = math.sqrt(target_pixels / current_pixels)

                        # Apply scale to both dimensions
                        scaled_width = int(round(width * scale))
                        scaled_height = int(round(height * scale))

                        # Round to nearest multiple of 8 (Z-Image requirement)
                        scaled_width = (scaled_width // 8) * 8
                        scaled_height = (scaled_height // 8) * 8

                        # Ensure minimum size
                        scaled_width = max(8, scaled_width)
                        scaled_height = max(8, scaled_height)

                        generation_params["resolution"] = f"{scaled_width}x{scaled_height}"
                        headless_logger.info(
                            f"Scaled resolution: {width}x{height} ({current_pixels:,} px) → "
                            f"{scaled_width}x{scaled_height} ({scaled_width*scaled_height:,} px, scale={scale:.3f})",
                            task_id=task_id
                        )
                    else:
                        generation_params["resolution"] = DEFAULT_IMAGE_RESOLUTION
            elif "resolution" in db_task_params:
                generation_params["resolution"] = db_task_params["resolution"]
            else:
                generation_params["resolution"] = DEFAULT_IMAGE_RESOLUTION

        except (OSError, ValueError, RuntimeError) as e:
            # If download fails, clean up and raise error
            if local_image_path:
                import os
                try:
                    os.unlink(local_image_path)
                except OSError as e_cleanup:
                    headless_logger.debug(f"Failed to clean up temp image file after download error: {e_cleanup}", task_id=task_id)
            raise ValueError(f"Task {task_id}: Failed to download image for img2img: {e}") from e

        # CRITICAL: Pass local file path (not URL) to WGP
        generation_params["image_start"] = local_image_path

        # Set img2img parameters
        generation_params.setdefault("video_prompt_type", "")  # Image input handled via image_start
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 12)))

        # CRITICAL: Add denoising_strength to generation_params so it gets passed through
        actual_strength = db_task_params.get("denoising_strength") or db_task_params.get("denoise_strength") or db_task_params.get("strength", 0.7)
        generation_params["denoising_strength"] = actual_strength

        headless_logger.info(
            f"[Z_IMAGE_I2I] Setup complete - local_image={local_image_path}, resolution={generation_params['resolution']}, "
            f"strength={actual_strength}, steps={generation_params['num_inference_steps']}",
            task_id=task_id
        )

        # Override model to use Z-Image img2img
        model = "z_image_img2img"

    # Defaults
    essential_defaults = {
        "seed": -1,
        "negative_prompt": "",
    }
    for param, default_value in essential_defaults.items():
        if param not in generation_params:
            generation_params[param] = default_value
    
    # Phase Config Override
    if "phase_config" in db_task_params:
        try:
            steps_per_phase = db_task_params["phase_config"].get("steps_per_phase", [2, 2, 2])
            phase_config_steps = sum(steps_per_phase)

            parsed_phase_config = parse_phase_config(
                phase_config=db_task_params["phase_config"],
                num_inference_steps=phase_config_steps,
                task_id=task_id,
                model_name=model,
                debug_mode=debug_mode
            )

            generation_params["num_inference_steps"] = phase_config_steps

            for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                       "guidance_scale", "guidance2_scale", "guidance3_scale",
                       "flow_shift", "sample_solver", "model_switch_phase",
                       "lora_names", "lora_multipliers", "additional_loras"]:
                if key in parsed_phase_config and parsed_phase_config[key] is not None:
                    generation_params[key] = parsed_phase_config[key]

            if "lora_names" in parsed_phase_config:
                generation_params["activated_loras"] = parsed_phase_config["lora_names"]
            if "lora_multipliers" in parsed_phase_config:
                generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])

            if "_patch_config" in parsed_phase_config:
                generation_params["_parsed_phase_config"] = parsed_phase_config
                generation_params["_phase_config_model_name"] = model
            
            # Note: LoRA URL resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
            # URLs in activated_loras are detected by LoRAConfig.from_params() and downloaded there
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Task {task_id}: Invalid phase_config: {e}") from e

    priority = db_task_params.get("priority", 0)
    if task_type.endswith("_orchestrator"):
        priority = max(priority, 10)
    
    generation_task = GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=generation_params,
        priority=priority
    )
    
    return generation_task
