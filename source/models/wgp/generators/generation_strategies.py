"""Generation strategy convenience methods for WanOrchestrator.

Provides typed wrappers (T2V, VACE, Flux, config-based) around the core
:meth:`WanOrchestrator.generate` method.  Each function is bound to the
orchestrator at class-definition time via ``orchestrator.py``.
"""

from typing import TYPE_CHECKING, Optional

from source.core.log import generation_logger

if TYPE_CHECKING:
    from source.core.params import TaskConfig


# ---------------------------------------------------------------------------
# Text-to-Video
# ---------------------------------------------------------------------------

def generate_t2v(self, prompt: str, model_type: str = None, **kwargs) -> str:
    """Generate text-to-video content.

    Thin wrapper around :meth:`WanOrchestrator.generate` that adds a
    warning if the currently-loaded model is not a T2V model.

    Args:
        self: WanOrchestrator instance.
        prompt: Text prompt for generation.
        model_type: Optional model override.
        **kwargs: Additional parameters forwarded to generate().

    Returns:
        Path to generated video file.
    """
    if not self._is_t2v() and not self._is_z_image():
        generation_logger.warning(f"Current model {self.current_model} may not be optimized for T2V")
    return self.generate(prompt=prompt, model_type=model_type, **kwargs)


# ---------------------------------------------------------------------------
# VACE (Video Any Controllable Edit)
# ---------------------------------------------------------------------------

def generate_vace(
    self,
    prompt: str,
    video_guide: str,
    model_type: str = None,
    video_mask: Optional[str] = None,
    video_prompt_type: str = "VP",
    control_net_weight: float = 1.0,
    control_net_weight2: float = 1.0,
    **kwargs,
) -> str:
    """Generate VACE controlled video content.

    Thin wrapper around :meth:`WanOrchestrator.generate` that adds
    a warning if the currently-loaded model is not a VACE model.

    Args:
        self: WanOrchestrator instance.
        prompt: Text prompt for generation.
        video_guide: Path to primary control video (required).
        model_type: Optional model override.
        video_mask: Path to primary mask video (optional).
        video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA").
        control_net_weight: Strength for first VACE encoding.
        control_net_weight2: Strength for second VACE encoding.
        **kwargs: Additional parameters forwarded to generate().

    Returns:
        Path to generated video file.
    """
    if not self._is_vace():
        generation_logger.warning(f"Current model {self.current_model} may not be a VACE model")

    # Log LoRA parameters at VACE level
    generation_logger.debug_block(
        "GENERATE",
        {
            "path": "vace",
            "kwargs_keys": sorted(kwargs.keys()),
            "lora_names": kwargs.get("lora_names"),
            "lora_multipliers": kwargs.get("lora_multipliers"),
        },
    )

    return self.generate(
        prompt=prompt,
        model_type=model_type,
        video_guide=video_guide,
        video_mask=video_mask,
        video_prompt_type=video_prompt_type,
        control_net_weight=control_net_weight,
        control_net_weight2=control_net_weight2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Flux image generation
# ---------------------------------------------------------------------------

def generate_flux(self, prompt: str, images: int = 4, model_type: str = None, **kwargs) -> str:
    """Generate Flux images.

    Thin wrapper around :meth:`WanOrchestrator.generate` that adds a
    warning if the currently-loaded model is not a Flux model.

    Args:
        self: WanOrchestrator instance.
        prompt: Text prompt for generation.
        images: Number of images to generate (maps to video_length).
        model_type: Optional model override.
        **kwargs: Additional parameters forwarded to generate().

    Returns:
        Path to generated image(s).
    """
    if not self._is_flux():
        generation_logger.warning(f"Current model {self.current_model} may not be a Flux model")

    return self.generate(
        prompt=prompt,
        model_type=model_type,
        video_length=images,  # For Flux, video_length = number of images
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config-based generation (Qwen / universal typed API)
# ---------------------------------------------------------------------------

def generate_with_config(self, config: 'TaskConfig') -> str:
    """Generate content using a typed TaskConfig object.

    This is the new typed API that provides cleaner parameter handling:
    1. All parameters are already parsed and validated in TaskConfig
    2. LoRAs are already resolved (downloaded if needed)
    3. Phase config is already applied

    Works for all model types (T2V, VACE, Flux, Qwen) but is the
    primary entry-point for Qwen-style image generation.

    Args:
        self: WanOrchestrator instance.
        config: TaskConfig object with all generation parameters.

    Returns:
        Path to generated output file(s).

    Raises:
        TypeError: If config is not a TaskConfig instance.
        RuntimeError: If no model is loaded.
        ValueError: If no prompt is provided.

    Example::

        from source.core.params import TaskConfig
        config = TaskConfig.from_db_task(task.parameters, task_id=task.id)
        result = orchestrator.generate_with_config(config)
    """
    # Import here to avoid circular import
    from source.core.params import TaskConfig

    if not isinstance(config, TaskConfig):
        raise TypeError(f"Expected TaskConfig, got {type(config)}")

    if not self.current_model:
        raise RuntimeError("No model loaded. Call load_model() first.")

    # Log what we're generating
    generation_logger.debug_block(
        "GENERATE",
        {
            "task_id": config.task_id,
            "path": "typed_config",
            "model": config.model or self.current_model,
            "lora_files": [] if config.lora.is_empty() else config.lora.filenames,
            "phase_count": 0 if config.phase.is_empty() else config.phase.num_phases,
        },
    )

    # Convert to WGP format
    wgp_params = config.to_wgp_format()

    # Ensure prompt is set
    prompt = wgp_params.pop('prompt', config.generation.prompt)
    if not prompt:
        raise ValueError("No prompt provided in TaskConfig")

    # Determine model type
    model_type = wgp_params.pop('model', None) or config.model or self.current_model

    # Call the standard generate method with unpacked params
    return self.generate(
        prompt=prompt,
        model_type=model_type,
        **wgp_params,
    )
