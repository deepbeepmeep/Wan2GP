"""
Qwen Image Edit Model Handler

Handles all Qwen-specific task types with proper LoRA configuration,
system prompts, and preprocessing logic.

Compositing utilities live in qwen_compositor.py.
Prompt/system-instruction helpers live in qwen_prompts.py.

Extracted from worker.py (687 lines of Qwen code).
"""

from pathlib import Path
from typing import Dict, Any, Optional

from source.core.log import model_logger
from source.core.params.phase_multiplier_utils import format_phase_multipliers, extract_phase_values
from source.utils.download_utils import download_image_if_url

from source.models.model_handlers.qwen_compositor import (
    cap_qwen_resolution,
    create_qwen_masked_composite,
)
from source.models.model_handlers.qwen_prompts import (
    SYSTEM_PROMPT_IMAGE_EDIT,
    SYSTEM_PROMPT_INPAINT,
    SYSTEM_PROMPT_ANNOTATED_EDIT,
    SYSTEM_PROMPT_IMAGE_GEN,
    SYSTEM_PROMPT_IMAGE_HIRES,
    SYSTEM_PROMPT_IMAGE_2512,
    SYSTEM_PROMPT_TURBO,
    apply_system_prompt,
    build_style_prompt,
    select_style_system_prompt,
)

# Model configs for Qwen image edit variants.
# Each variant has its own base model and dedicated Lightning LoRA.
QWEN_EDIT_MODEL_CONFIG = {
    "qwen-edit": {
        "model_name": "qwen_image_edit_20B",
        "lightning_fname": "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        "lightning_repo": "lightx2v/Qwen-Image-Lightning",
        "hf_subfolder": None,
    },
    "qwen-edit-2509": {
        "model_name": "qwen_image_edit_plus_20B",
        "lightning_fname": "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
        "lightning_repo": "lightx2v/Qwen-Image-Lightning",
        "hf_subfolder": "Qwen-Image-Edit-2509",
    },
    "qwen-edit-2511": {
        "model_name": "qwen_image_edit_plus2_20B",
        "lightning_fname": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        "lightning_repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "hf_subfolder": None,
    },
}


class LoRADownloadError(RuntimeError):
    """Raised when a required Qwen LoRA download fails."""


class QwenHandler:
    """Handles all Qwen image editing task types."""

    def __init__(self, wan_root: str, task_id: str):
        """
        Initialize Qwen handler.

        Args:
            wan_root: Path to Wan2GP root directory
            task_id: Task ID for logging
        """
        self.wan_root = Path(wan_root).resolve()
        self.task_id = task_id
        self.qwen_lora_dir = self.wan_root / "loras_qwen"
        self.qwen_lora_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging helpers ─────────────────────────────────────────────────

    def _log_debug(self, message: str):
        """Log debug message."""
        model_logger.debug(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def _log_info(self, message: str):
        """Log info message."""
        model_logger.info(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def _log_warning(self, message: str):
        """Log warning message."""
        model_logger.warning(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    def _log_error(self, message: str):
        """Log error message."""
        model_logger.error(f"[QWEN_HANDLER] Task {self.task_id}: {message}", task_id=self.task_id)

    # ── Model config helpers ────────────────────────────────────────────

    def _get_edit_model_config(self, db_task_params: Dict[str, Any]) -> dict:
        """Get the config for the selected edit model variant."""
        variant = db_task_params.get("qwen_edit_model", "qwen-edit")
        config = QWEN_EDIT_MODEL_CONFIG.get(variant)
        if config is None:
            self._log_warning(f"Unknown qwen_edit_model '{variant}', falling back to 'qwen-edit'")
            config = QWEN_EDIT_MODEL_CONFIG["qwen-edit"]
        return config

    def get_edit_model_name(self, db_task_params: Dict[str, Any]) -> str:
        """Return the WGP model name for the selected edit model variant."""
        return self._get_edit_model_config(db_task_params)["model_name"]

    # ── Compositor / resolution delegates ───────────────────────────────

    def cap_qwen_resolution(self, resolution_str: str) -> Optional[str]:
        """Cap resolution to 1200px max dimension while maintaining aspect ratio."""
        return cap_qwen_resolution(resolution_str, task_id=self.task_id)

    def create_qwen_masked_composite(
        self,
        image_url: str,
        mask_url: str,
        output_dir: Path,
    ) -> str:
        """Create composite image with green overlay for Qwen inpainting/annotation."""
        return create_qwen_masked_composite(
            image_url,
            mask_url,
            output_dir,
            task_id=self.task_id,
        )

    # ── LoRA management ─────────────────────────────────────────────────

    def _download_lora_if_missing(self, repo_id: str, filename: str, hf_subfolder: str = None) -> Optional[Path]:
        """Helper to download a LoRA if it doesn't exist locally."""
        target_path = self.qwen_lora_dir / filename
        if target_path.exists():
            self._log_debug(f"LoRA already present: {target_path}")
            return target_path.resolve()

        self._log_info(f"Downloading LoRA '{filename}' from '{repo_id}'" +
                       (f" (subfolder: {hf_subfolder})" if hf_subfolder else ""))
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            dl_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=hf_subfolder,
                revision="main",
                local_dir=str(self.qwen_lora_dir)
            )
            # If downloaded to a subdirectory, move to flat location
            actual_path = Path(dl_path)
            if actual_path.exists() and actual_path.resolve() != target_path.resolve():
                actual_path.replace(target_path)
            self._log_info(f"Successfully downloaded {filename}")
            return target_path.resolve()
        except (OSError, ValueError, RuntimeError) as e:
            self._log_warning(f"LoRA download failed for {filename}: {e}")
            return None

    def _ensure_lora_lists(self, generation_params: Dict[str, Any]):
        """Ensure lora_names and lora_multipliers exist as lists."""
        generation_params.setdefault("lora_names", [])
        generation_params.setdefault("lora_multipliers", [])

    def _add_lightning_lora(
        self,
        db_task_params: Dict[str, Any],
        generation_params: Dict[str, Any],
        lightning_fname: str = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        lightning_repo: str = "lightx2v/Qwen-Image-Lightning",
        hf_subfolder: str = None,
        default_phase1: float = 0.85,
        default_phase2: float = 0.0,
    ):
        """Download Lightning LoRA if needed and add to generation params with phase support."""
        self._ensure_lora_lists(generation_params)

        if not (self.qwen_lora_dir / lightning_fname).exists():
            self._download_lora_if_missing(lightning_repo, lightning_fname, hf_subfolder=hf_subfolder)

        if lightning_fname in generation_params["lora_names"]:
            return

        phase1 = float(db_task_params.get("lightning_lora_strength_phase_1", default_phase1))
        phase2 = float(db_task_params.get("lightning_lora_strength_phase_2", default_phase2))

        generation_params["lora_names"].append(lightning_fname)
        if db_task_params.get("hires_scale") is not None:
            generation_params["lora_multipliers"].append(f"{phase1};{phase2}")
            self._log_info(f"Added Lightning LoRA with strength Phase1={phase1}, Phase2={phase2}")
        else:
            generation_params["lora_multipliers"].append(phase1)
            self._log_info(f"Added Lightning LoRA with strength {phase1}")

    def _add_lightning_lora_simple(
        self,
        db_task_params: Dict[str, Any],
        generation_params: Dict[str, Any],
        lightning_fname: str = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
        lightning_repo: str = "lightx2v/Qwen-Image-Lightning",
        strength_param: str = "lightning_lora_strength",
        default_strength: float = 0.45,
    ):
        """Download Lightning LoRA if needed and add with a single strength value (no phases)."""
        self._ensure_lora_lists(generation_params)

        if not (self.qwen_lora_dir / lightning_fname).exists():
            downloaded = self._download_lora_if_missing(lightning_repo, lightning_fname)
            if not downloaded:
                raise LoRADownloadError(f"Failed to download LoRA: {lightning_fname}")

        if lightning_fname in generation_params["lora_names"]:
            return

        strength = float(db_task_params.get(strength_param, default_strength))
        generation_params["lora_names"].append(lightning_fname)
        generation_params["lora_multipliers"].append(strength)
        self._log_info(f"Added Lightning LoRA '{lightning_fname}' @ {strength}")

    def _add_task_lora(self, generation_params: Dict[str, Any], repo_id: str, filename: str, multiplier: float = 1.0):
        """Download and add a task-specific LoRA."""
        downloaded = self._download_lora_if_missing(repo_id, filename)
        if not downloaded:
            raise LoRADownloadError(f"Failed to download LoRA: {filename}")
        self._ensure_lora_lists(generation_params)
        if filename not in generation_params["lora_names"]:
            generation_params["lora_names"].append(filename)
            generation_params["lora_multipliers"].append(multiplier)

    def _apply_additional_loras(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Apply additional LoRAs from task params (loras array or additional_loras dict)."""
        # Handle array format: [{"path": "url", "scale": 0.8}]
        loras = db_task_params.get("loras", [])
        for lora in loras:
            if isinstance(lora, dict):
                path = lora.get("path", "")
                scale = float(lora.get("scale", 1.0))
                if path:
                    if "additional_loras" not in generation_params:
                        generation_params["additional_loras"] = {}
                    generation_params["additional_loras"][path] = scale
                    self._log_debug(f"Added LoRA from array: {path} @ {scale}")

        # Handle dict format: {"url": scale}
        additional_loras = db_task_params.get("additional_loras", {})
        if additional_loras:
            if "additional_loras" not in generation_params:
                generation_params["additional_loras"] = {}
            generation_params["additional_loras"].update(additional_loras)
            self._log_debug(f"Added {len(additional_loras)} additional LoRAs from dict")

    # ── Hires fix ───────────────────────────────────────────────────────

    def _maybe_add_hires_config(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Add hires_config if hires params are present in task params."""
        hires_scale = db_task_params.get("hires_scale")
        if hires_scale is None:
            return  # No hires fix requested

        hires_config = {
            "enabled": True,
            "scale": float(hires_scale),
            "hires_steps": int(db_task_params.get("hires_steps", 6)),
            "denoising_strength": float(db_task_params.get("hires_denoise", 0.5)),
            "upscale_method": db_task_params.get("hires_upscale_method", "bicubic"),
        }
        generation_params["hires_config"] = hires_config

        # Convert LoRA multipliers to phase-based format (pass1;pass2)
        # This allows fine control over which LoRAs are active in each pass
        self._convert_to_phase_multipliers(generation_params)

        self._log_info(
            f"Hires fix enabled: {hires_config['scale']}x scale, "
            f"{hires_config['hires_steps']} steps @ {hires_config['denoising_strength']} denoise"
        )

    def _convert_to_phase_multipliers(self, generation_params: Dict[str, Any]):
        """
        Convert simple LoRA multipliers to phase-based format for hires fix.

        IMPORTANT: This stores the full phase format in hires_config but sends
        only pass 1 values to WGP to avoid conflicts with WGP's phase system.

        Converts:
          ["1.1", "0.5"] -> ["1.1;1.1", "0.5;0.5"]

        Where the format is "pass1_strength;pass2_strength".

        Lightning LoRA phases are manually controlled via:
        - lightning_lora_strength_phase_1 (default: 0.85)
        - lightning_lora_strength_phase_2 (default: 0.0)

        All other LoRAs -> "X;X" (same strength in both passes unless phase format specified).
        """
        if "lora_multipliers" not in generation_params:
            return

        lora_names = generation_params.get("lora_names", [])
        multipliers = generation_params["lora_multipliers"]

        # Ensure multipliers is a list
        if not isinstance(multipliers, list):
            multipliers = [multipliers]

        # Convert to phase-based format
        # Note: auto_detect_lightning=False because Lightning LoRA phases are now manually specified
        phase_multipliers = format_phase_multipliers(
            lora_names=lora_names,
            multipliers=multipliers,
            num_phases=2,  # Hires fix = 2 passes
            auto_detect_lightning=False  # Manual control via lightning_lora_strength_phase_1/2
        )

        # Store full phase format AND LoRA names in hires_config for qwen_main.py to use
        if "hires_config" in generation_params:
            generation_params["hires_config"]["lora_names"] = lora_names
            generation_params["hires_config"]["phase_lora_multipliers"] = phase_multipliers
            self._log_info(f"Stored {len(lora_names)} LoRA names and phase multipliers in hires_config")
            for i, (name, mult) in enumerate(zip(lora_names, phase_multipliers)):
                self._log_info(f"   LoRA {i+1}: {name} @ {mult}")

        # Extract ONLY pass 1 values to send to WGP (avoids conflict with WGP's phase system)
        pass1_multipliers = extract_phase_values(
            phase_multipliers,
            phase_index=0,  # Pass 1
            num_phases=2
        )

        generation_params["lora_multipliers"] = pass1_multipliers
        self._log_info(f"Sending pass 1 multipliers to WGP: {pass1_multipliers}")

    # ── Task handlers ───────────────────────────────────────────────────

    def handle_qwen_image_edit(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle qwen_image_edit task type."""
        self._log_info("Processing qwen_image_edit task")

        image_url = db_task_params.get("image") or db_task_params.get("image_url")
        if not image_url:
            raise ValueError(f"Task {self.task_id}: 'image' or 'image_url' required for qwen_image_edit")

        downloads_dir = Path("outputs/qwen_edit_images")
        downloads_dir.mkdir(parents=True, exist_ok=True)
        local_image_path = download_image_if_url(
            image_url, downloads_dir,
            task_id_for_logging=self.task_id,
            debug_mode=False,
            descriptive_name="edit_image"
        )
        generation_params["image_guide"] = str(local_image_path)
        self._log_info(f"Using image_guide: {local_image_path}")

        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res

        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        apply_system_prompt(db_task_params, generation_params, SYSTEM_PROMPT_IMAGE_EDIT)

        edit_config = self._get_edit_model_config(db_task_params)
        self._add_lightning_lora(
            db_task_params, generation_params,
            lightning_fname=edit_config["lightning_fname"],
            lightning_repo=edit_config["lightning_repo"],
            hf_subfolder=edit_config["hf_subfolder"],
            default_phase1=0.85, default_phase2=0.0,
        )

        # Optional hires fix - can be enabled on any qwen_image_edit task
        self._maybe_add_hires_config(db_task_params, generation_params)

    def _handle_qwen_image_task(
        self,
        db_task_params: Dict[str, Any],
        generation_params: Dict[str, Any],
        task_label: str,
        composite_subdir: str,
        default_system_prompt: str,
        task_lora_repo: str,
        task_lora_fname: str,
    ):
        """Shared implementation for mask-based Qwen image tasks (inpaint, annotated edit)."""
        self._log_info(f"Processing {task_label} task")

        image_url = db_task_params.get("image_url") or db_task_params.get("image")
        mask_url = db_task_params.get("mask_url")

        if not image_url or not mask_url:
            raise ValueError(f"Task {self.task_id}: 'image_url' and 'mask_url' required for {task_label}")

        composite_dir = Path(f"outputs/{composite_subdir}")
        composite_path = self.create_qwen_masked_composite(image_url, mask_url, composite_dir)

        generation_params["image_guide"] = str(composite_path)

        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res

        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        apply_system_prompt(db_task_params, generation_params, default_system_prompt)

        edit_config = self._get_edit_model_config(db_task_params)
        self._add_lightning_lora(
            db_task_params, generation_params,
            lightning_fname=edit_config["lightning_fname"],
            lightning_repo=edit_config["lightning_repo"],
            hf_subfolder=edit_config["hf_subfolder"],
            default_phase1=0.75, default_phase2=0.0,
        )
        self._add_task_lora(generation_params, task_lora_repo, task_lora_fname)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

    def handle_image_inpaint(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle image_inpaint task type."""
        self._handle_qwen_image_task(
            db_task_params, generation_params,
            task_label="image_inpaint",
            composite_subdir="qwen_inpaint_composites",
            default_system_prompt=SYSTEM_PROMPT_INPAINT,
            task_lora_repo="ostris/qwen_image_edit_inpainting",
            task_lora_fname="qwen_image_edit_inpainting.safetensors",
        )

    def handle_annotated_image_edit(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle annotated_image_edit task type."""
        self._handle_qwen_image_task(
            db_task_params, generation_params,
            task_label="annotated_image_edit",
            composite_subdir="qwen_annotate_composites",
            default_system_prompt=SYSTEM_PROMPT_ANNOTATED_EDIT,
            task_lora_repo="peteromallet/random_junk",
            task_lora_fname="in_scene_pure_squares_flipped_450_lr_000006700.safetensors",
        )

    def handle_qwen_image_style(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """Handle qwen_image_style task type."""
        self._log_info("Processing qwen_image_style task")

        original_prompt = generation_params.get("prompt", db_task_params.get("prompt", ""))

        style_strength = float(db_task_params.get("style_reference_strength", 0.0) or 0.0)
        subject_strength = float(db_task_params.get("subject_strength", 0.0) or 0.0)
        scene_strength = float(db_task_params.get("scene_reference_strength", 0.0) or 0.0)
        subject_description = db_task_params.get("subject_description", "")
        in_this_scene = db_task_params.get("in_this_scene", False)

        # Build modified prompt using extracted helper
        modified_prompt = build_style_prompt(
            original_prompt, style_strength, subject_strength,
            subject_description, in_this_scene,
        )
        if modified_prompt != original_prompt:
            generation_params["prompt"] = modified_prompt
            self._log_info(f"Modified prompt to: {modified_prompt}")

        reference_image = db_task_params.get("style_reference_image") or db_task_params.get("subject_reference_image")
        if reference_image:
            try:
                downloads_dir = Path("outputs/style_refs")
                downloads_dir.mkdir(parents=True, exist_ok=True)
                local_ref_path = download_image_if_url(
                    reference_image, downloads_dir,
                    task_id_for_logging=self.task_id,
                    debug_mode=False,
                    descriptive_name="reference_image"
                )
                generation_params["image_guide"] = str(local_ref_path)
            except (OSError, ValueError, RuntimeError) as e:
                self._log_warning(f"Failed to download reference image: {e}")

        if "resolution" in db_task_params:
            capped_res = self.cap_qwen_resolution(db_task_params["resolution"])
            if capped_res:
                generation_params["resolution"] = capped_res

        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", 12)
        generation_params.setdefault("video_length", 1)

        # Set system prompt based on parameters
        custom_prompt = db_task_params.get("system_prompt")
        if custom_prompt:
            generation_params["system_prompt"] = custom_prompt
        else:
            generation_params["system_prompt"] = select_style_system_prompt(
                has_subject=subject_strength > 0,
                has_style=style_strength > 0,
                has_scene=scene_strength > 0,
            )

        # Lightning LoRA (variant-aware)
        edit_config = self._get_edit_model_config(db_task_params)
        self._add_lightning_lora(
            db_task_params, generation_params,
            lightning_fname=edit_config["lightning_fname"],
            lightning_repo=edit_config["lightning_repo"],
            hf_subfolder=edit_config["hf_subfolder"],
            default_phase1=0.85, default_phase2=0.0,
        )

        # Conditional style/subject/scene LoRAs
        style_fname = "style_transfer_qwen_edit_2_000011250.safetensors"
        subject_fname = "in_subject_qwen_edit_2_000006750.safetensors"
        scene_fname = "in_scene_different_object_000010500.safetensors"

        if style_strength > 0.0:
            self._add_task_lora(generation_params, "peteromallet/ad_motion_loras", style_fname, style_strength)
        if subject_strength > 0.0:
            self._add_task_lora(generation_params, "peteromallet/mystery_models", subject_fname, subject_strength)
        if scene_strength > 0.0:
            self._add_task_lora(generation_params, "peteromallet/random_junk", scene_fname, scene_strength)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

    def handle_qwen_image_hires(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """
        Handle qwen_image_hires task type - two-pass generation with latent upscaling.

        Workflow:
        - Pass 1: Generate at base resolution (default 1328x1328)
        - Latent upscale: Bicubic interpolation in latent space
        - Pass 2: Refine at higher resolution with partial denoising

        This replicates the ComfyUI two-pass hires fix workflow.
        """
        self._log_info("Processing qwen_image_hires task (two-pass generation)")

        # Default base resolutions matching ComfyUI workflow
        BASE_RESOLUTIONS = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        # Determine base resolution
        resolution_input = db_task_params.get("resolution", "1328x1328")
        if resolution_input in BASE_RESOLUTIONS:
            # Aspect ratio key provided
            w, h = BASE_RESOLUTIONS[resolution_input]
            resolution_str = f"{w}x{h}"
        else:
            resolution_str = resolution_input

        generation_params["resolution"] = resolution_str

        # Build hires config
        hires_config = {
            "enabled": True,
            "scale": float(db_task_params.get("hires_scale", 2.0)),
            "hires_steps": int(db_task_params.get("hires_steps", 6)),
            "denoising_strength": float(db_task_params.get("hires_denoise", 0.5)),
            "upscale_method": db_task_params.get("hires_upscale_method", "bicubic"),
        }
        generation_params["hires_config"] = hires_config

        # Base generation params
        generation_params.setdefault("video_prompt_type", "KI")
        generation_params.setdefault("guidance_scale", 1)
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 10)))
        generation_params.setdefault("video_length", 1)

        # System prompt
        apply_system_prompt(db_task_params, generation_params, SYSTEM_PROMPT_IMAGE_HIRES)

        self._add_lightning_lora_simple(db_task_params, generation_params, default_strength=0.45)

        # Log final config
        base_w, base_h = map(int, resolution_str.split("x"))
        final_w = int(base_w * hires_config["scale"])
        final_h = int(base_h * hires_config["scale"])
        self._log_info(
            f"Hires workflow: {resolution_str} -> {final_w}x{final_h}, "
            f"base {generation_params['num_inference_steps']} steps, "
            f"hires {hires_config['hires_steps']} steps @ {hires_config['denoising_strength']} denoise"
        )

    def handle_qwen_image(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """
        Handle qwen_image task type - general text-to-image generation.

        This is the standard Qwen text-to-image generation without any input image.
        Supports optional LoRAs and standard generation parameters.
        """
        self._log_info("Processing qwen_image task (text-to-image)")

        # Resolution handling
        resolution = db_task_params.get("resolution", "1024x1024")
        capped_res = self.cap_qwen_resolution(resolution)
        if capped_res:
            generation_params["resolution"] = capped_res

        # Base generation params - optimized for 4-step Lightning V2.0
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("guidance_scale", 3.5)
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 4)))
        generation_params.setdefault("video_length", 1)

        # System prompt
        apply_system_prompt(db_task_params, generation_params, SYSTEM_PROMPT_IMAGE_GEN)

        self._add_lightning_lora(
            db_task_params, generation_params,
            lightning_fname="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
            lightning_repo="lightx2v/Qwen-Image-Lightning",
            default_phase1=1.0, default_phase2=1.0,
        )

        # Handle additional LoRAs from task params
        self._apply_additional_loras(db_task_params, generation_params)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

        self._log_info(f"qwen_image: {generation_params.get('resolution')}, {generation_params['num_inference_steps']} steps")

    def handle_qwen_image_2512(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """
        Handle qwen_image_2512 task type - enhanced text-to-image with better text rendering.

        This variant is optimized for:
        - Better text/typography rendering in images
        - More realistic human generation
        - Higher quality output overall
        """
        self._log_info("Processing qwen_image_2512 task (enhanced text-to-image)")

        # Resolution handling - this model supports up to 2512px
        resolution = db_task_params.get("resolution", "1024x1024")
        # Allow higher resolution for this model variant
        capped_res = cap_qwen_resolution(resolution, max_dimension=2512, task_id=self.task_id)
        if capped_res:
            generation_params["resolution"] = capped_res

        # Base generation params - optimized for 4-step Lightning LoRA
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("guidance_scale", 4.0)  # Slightly higher for better text
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 4)))
        generation_params.setdefault("video_length", 1)

        # System prompt optimized for text rendering
        apply_system_prompt(db_task_params, generation_params, SYSTEM_PROMPT_IMAGE_2512)

        self._add_lightning_lora(
            db_task_params, generation_params,
            lightning_fname="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            lightning_repo="lightx2v/Qwen-Image-2512-Lightning",
            default_phase1=1.0, default_phase2=1.0,
        )

        # Handle additional LoRAs
        self._apply_additional_loras(db_task_params, generation_params)

        # Optional hires fix
        self._maybe_add_hires_config(db_task_params, generation_params)

        self._log_info(f"qwen_image_2512: {generation_params.get('resolution')}, {generation_params['num_inference_steps']} steps")

    def handle_z_image_turbo(self, db_task_params: Dict[str, Any], generation_params: Dict[str, Any]):
        """
        Handle z_image_turbo task type - ultra-fast image generation.

        Optimized for:
        - Rapid prototyping (~0.2s generation)
        - High-volume generation
        - Cost-sensitive workflows

        Uses 8-step inference pipeline with aggressive Lightning LoRA.
        """
        self._log_info("Processing z_image_turbo task (fast text-to-image)")

        # Resolution handling
        resolution = db_task_params.get("resolution", "512x512")  # Default smaller for speed
        capped_res = self.cap_qwen_resolution(resolution)
        if capped_res:
            generation_params["resolution"] = capped_res

        # Base generation params - optimized for SPEED
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("guidance_scale", 2.5)  # Lower for faster convergence
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 8)))  # Fast!
        generation_params.setdefault("video_length", 1)

        # System prompt
        apply_system_prompt(db_task_params, generation_params, SYSTEM_PROMPT_TURBO)

        self._add_lightning_lora_simple(db_task_params, generation_params, default_strength=1.0)

        # Handle additional LoRAs
        self._apply_additional_loras(db_task_params, generation_params)

        # Note: hires fix generally not recommended for turbo mode (defeats the speed purpose)
        # but still allow it if explicitly requested
        if db_task_params.get("hires_scale"):
            self._maybe_add_hires_config(db_task_params, generation_params)
            self._log_warning("Hires fix enabled on turbo mode - this will significantly increase generation time")

        self._log_info(f"z_image_turbo: {generation_params.get('resolution')}, {generation_params['num_inference_steps']} steps (fast mode)")
