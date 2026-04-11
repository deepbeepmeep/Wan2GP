"""
WanGP Content Generation Orchestrator

Initializes WGP runtime state (sys.path, config hygiene, Matplotlib backend),
manages output-path configuration, traces LoRA application, runs preflight
checks (directory verification, model-definition loading), and delegates
generation to the extracted helpers in source/models/wgp/.
"""

import os
import sys
import traceback
from typing import Optional, List, TYPE_CHECKING

# Import structured logging
from source.core.log import (
    orchestrator_logger, model_logger, generation_logger, safe_dict_repr,
)

# Type hints for TaskConfig (avoid circular import)
if TYPE_CHECKING:
    from source.core.params import TaskConfig

# ---------------------------------------------------------------------------
# Delegated modules (source/models/wgp/ package)
# ---------------------------------------------------------------------------
from source.models.wgp.param_resolution import resolve_parameters as _resolve_parameters_impl
from source.models.wgp.lora_setup import setup_loras_for_model as _setup_loras_for_model_impl
from source.models.wgp.model_ops import (
    load_missing_model_definition as _load_missing_model_definition_impl,
    load_model_impl as _load_model_impl,
    unload_model_impl as _unload_model_impl,
    get_or_load_uni3c_controlnet as _get_or_load_uni3c_controlnet_impl,
)
from source.models.wgp.generation_helpers import (
    verify_wgp_directory as _verify_wgp_directory,
    create_vace_fixed_generate_video as _create_vace_fixed_generate_video_impl,
    load_image as _load_image_impl,
    resolve_media_path as _resolve_media_path_impl,
    is_vace as _is_vace_impl,
    is_model_vace as _is_model_vace_impl,
    is_flux as _is_flux_impl,
    is_t2v as _is_t2v_impl,
    is_qwen as _is_qwen_impl,
)

# ---------------------------------------------------------------------------
# Generation pipeline helpers (source/models/wgp/generators/ package)
# ---------------------------------------------------------------------------
from source.models.wgp.generators.capture import run_with_capture
from source.models.wgp.generators.wgp_params import (
    build_passthrough_params,
    build_normal_params,
    apply_kwargs_overrides,
)
from source.models.wgp.generators.output import (
    extract_output_path,
    log_captured_output,
    log_memory_stats,
)
from source.models.wgp.generators.preflight import (
    prepare_svi_image_refs,
    configure_model_specific_params,
    prepare_image_inputs,
)
from source.models.wgp.generators.generation_strategies import (
    generate_t2v as _generate_t2v_impl,
    generate_vace as _generate_vace_impl,
    generate_flux as _generate_flux_impl,
    generate_with_config as _generate_with_config_impl,
)
from source.runtime.process_globals import temporary_process_globals
from source.runtime.wgp_bridge import (
    get_wgp_runtime_module,
    get_wgp_runtime_module_mutable,
    reset_wgp_runtime_module,
)

from source.core.log import is_debug_enabled


def _set_wgp_output_paths(wgp_module, output_dir: str) -> None:
    """Set output paths on WGP module-level state and server_config dict.

    Must be called both BEFORE and AFTER apply_changes() because
    apply_changes() resets certain internal state. The WGP vendored
    code reads these paths at initialization AND at generation time,
    so both calls are required to ensure correct output locations.
    """
    if not hasattr(wgp_module, 'server_config'):
        wgp_module.server_config = {}
    wgp_module.server_config['save_path'] = output_dir
    wgp_module.server_config['image_save_path'] = output_dir
    wgp_module.save_path = output_dir
    wgp_module.image_save_path = output_dir


class WanOrchestrator:
    """Thin adapter around `wgp.generate_video` for easier programmatic use."""

    # Typed runtime collaborator contract markers:
    # model_runtime: ModelRuntime | None = None
    # media_resolver: MediaResolver | None = None
    # parameter_resolver: ParameterResolver | None = None
    # self.model_runtime = model_runtime or DefaultModelRuntime(self)
    # runtime_context.prepare(...)

    def __init__(self, wan_root: str, main_output_dir: Optional[str] = None):
        """Initialize orchestrator with WanGP directory.

        Args:
            wan_root: Path to WanGP repository root directory (MUST be absolute path to Wan2GP/)
            main_output_dir: Optional path for output directory. If not provided, defaults to
                           'outputs' directory next to wan_root (preserves backwards compatibility)

        IMPORTANT: Caller MUST have already changed to wan_root directory before calling this.
        wgp.py uses relative paths and expects to run from Wan2GP/.
        """
        import logging
        _init_logger = logging.getLogger('HeadlessQueue')

        # Store the wan_root (should match current directory)
        self.wan_root = os.path.abspath(wan_root)
        current_dir = os.getcwd()

        # CRITICAL CHECK: Verify caller already changed to the correct directory
        # wgp.py will import and execute module-level code that uses relative paths like "defaults/*.json"
        # If we're in the wrong directory, wgp will load 0 models and fail mysteriously
        if current_dir != self.wan_root:
            error_msg = (
                f"CRITICAL: WanOrchestrator must be initialized from Wan2GP directory!\n"
                f"  Current directory: {current_dir}\n"
                f"  Expected directory: {self.wan_root}\n"
                f"  Caller must chdir() before creating WanOrchestrator instance."
            )
            if is_debug_enabled():
                _init_logger.error(f"[INIT_DEBUG] {error_msg}")
            raise RuntimeError(error_msg)

        # Verify Wan2GP structure
        if not os.path.isdir("defaults"):
            raise RuntimeError(
                f"defaults/ directory not found in {current_dir}. "
                f"This doesn't appear to be a valid Wan2GP directory!"
            )
        if not os.path.isdir("models"):
            if is_debug_enabled():
                _init_logger.warning(f"[INIT_DEBUG] models/ directory not found in {current_dir}")

        # Ensure Wan2GP is first in sys.path so wgp.py imports correctly
        if self.wan_root in sys.path:
            sys.path.remove(self.wan_root)
        sys.path.insert(0, self.wan_root)
        # Optional smoke/CPU-only modes
        self.smoke_mode = bool(os.environ.get("HEADLESS_WAN2GP_SMOKE", ""))
        force_cpu = os.environ.get("HEADLESS_WAN2GP_FORCE_CPU", "0") == "1"
        init_summary = {
            "wan_root": self.wan_root,
            "cwd": current_dir,
            "smoke_mode": self.smoke_mode,
            "force_cpu": force_cpu,
        }
        if is_debug_enabled():
            _init_logger.debug("INIT_DEBUG %s", safe_dict_repr(init_summary))

        # Force CPU if requested and guard CUDA capability queries before importing WGP
        if force_cpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import torch  # type: ignore
            # If CUDA isn't available, stub capability query used by upstream on import
            if force_cpu or not torch.cuda.is_available():
                try:
                    def _safe_get_device_capability(device=None):
                        return (8, 0)
                    torch.cuda.get_device_capability = _safe_get_device_capability  # type: ignore
                except (RuntimeError, AttributeError) as e:
                    _init_logger.debug("Failed to stub torch.cuda.get_device_capability: %s", e)
        except ImportError as e:
            _init_logger.debug("Failed to import torch for CPU/CUDA guards: %s", e)

        # Force a headless Matplotlib backend to avoid Tkinter requirements during upstream imports
        if not os.environ.get("MPLBACKEND"):
            os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib  # type: ignore
            _orig_use = matplotlib.use  # type: ignore
            def _force_agg(_backend=None, *args, **kwargs):
                try:
                    return _orig_use('Agg', force=True)
                except (RuntimeError, ValueError):
                    return None
            matplotlib.use = _force_agg  # type: ignore
        except ImportError as e:
            _init_logger.debug("Failed to force matplotlib Agg backend: %s", e)

        # Pre-import config hygiene for upstream wgp.py
        # Some older configs may store preload_model_policy as an int, but upstream now expects a list
        try:
            cfg_path = os.path.join(self.wan_root, "wgp_config.json")
            if os.path.isfile(cfg_path):
                import json as _json
                with open(cfg_path, "r", encoding="utf-8") as _r:
                    _cfg = _json.load(_r)
                changed = False
                pmp = _cfg.get("preload_model_policy", [])
                if isinstance(pmp, int):
                    _cfg["preload_model_policy"] = []  # disable preloading and fix type
                    changed = True
                # Ensure save paths exist and are strings
                if not isinstance(_cfg.get("save_path", ""), str):
                    _cfg["save_path"] = "outputs"
                    changed = True
                if not isinstance(_cfg.get("image_save_path", ""), str):
                    _cfg["image_save_path"] = "outputs"
                    changed = True
                if changed:
                    with open(cfg_path, "w", encoding="utf-8") as _w:
                        _json.dump(_cfg, _w, indent=4)
        except (OSError, ValueError, KeyError, TypeError):
            # Config hygiene should not block import
            pass

        # Import WGP components after path setup (skip entirely in smoke mode)
        if not self.smoke_mode:
            try:
                current_dir = os.getcwd()
                if is_debug_enabled():
                    _init_logger.debug(
                        f"[INIT_DEBUG] {safe_dict_repr({'phase': 'import_wgp', 'cwd': current_dir, 'sys_path_0': sys.path[0] if sys.path else 'empty', 'wgp_in_modules': 'wgp' in sys.modules})}"
                    )
                # Double-check we're in the right directory before importing
                if current_dir != self.wan_root:
                    if is_debug_enabled():
                        _init_logger.error(f"[INIT_DEBUG] CRITICAL: Current directory {current_dir} != expected {self.wan_root}")
                    raise RuntimeError(f"Directory changed unexpectedly before wgp import: {current_dir} != {self.wan_root}")

                # If wgp was previously imported from wrong directory, remove it so it reimports
                if 'wgp' in sys.modules:
                    if is_debug_enabled():
                        _init_logger.warning("[INIT_DEBUG] resetting existing wgp module before reimport")
                    reset_wgp_runtime_module()

                with temporary_process_globals(
                    argv=["headless_wgp.py"],
                    prepend_sys_path=self.wan_root,
                ):
                    wgp = get_wgp_runtime_module_mutable()
                    generate_video = wgp.generate_video
                    get_base_model_type = wgp.get_base_model_type
                    get_model_family = wgp.get_model_family
                    test_vace_module = wgp.test_vace_module

                # Verify directory didn't change during wgp import
                _verify_wgp_directory(_init_logger, "after importing wgp module")

                # Apply VACE fix wrapper to generate_video
                self._generate_video = _create_vace_fixed_generate_video_impl(generate_video)
                self._get_base_model_type = get_base_model_type
                self._get_model_family = get_model_family
                self._test_vace_module = test_vace_module

                # Apply WGP monkeypatches for headless operation (Qwen support, LoRA fixes, etc.)
                from source.models.wgp.wgp_patches import apply_all_wgp_patches
                apply_all_wgp_patches(wgp, self.wan_root)

                # Initialize WGP global state (normally done by UI)
                # Set absolute output path to avoid issues when working directory changes
                # Use main_output_dir if provided, otherwise default to 'outputs' next to wan_root
                if main_output_dir is not None:
                    absolute_outputs_path = os.path.abspath(main_output_dir)
                    output_dir_source = "provided"
                else:
                    absolute_outputs_path = os.path.abspath(os.path.join(os.path.dirname(self.wan_root), 'outputs'))
                    output_dir_source = "default"

                # Initialize attributes that don't exist yet
                for attr, default in {
                    'wan_model': None, 'offloadobj': None, 'reload_needed': True,
                    'transformer_type': None
                }.items():
                    if not hasattr(wgp, attr):
                        setattr(wgp, attr, default)

                # Set output paths on WGP (first call — before apply_changes)
                _set_wgp_output_paths(wgp, absolute_outputs_path)
                model_logger.debug_anomaly("OUTPUT_DIR", f"Initialized output paths to {absolute_outputs_path} (source={output_dir_source})")

                # Debug: Check if model definitions are loaded
                if is_debug_enabled():
                    init_summary["available_models"] = len(wgp.models_def)
                    _init_logger.debug_anomaly("INIT_DEBUG", f"{safe_dict_repr(init_summary)}")
                _verify_wgp_directory(model_logger, "after wgp setup and monkeypatching")

                if not wgp.models_def:
                    error_msg = (
                        f"CRITICAL: No model definitions found after importing wgp! "
                        f"Current directory: {os.getcwd()}, "
                        f"Expected: {self.wan_root}, "
                        f"defaults/ exists: {os.path.exists('defaults')}, "
                        f"finetunes/ exists: {os.path.exists('finetunes')}"
                    )
                    model_logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except ImportError as e:
                raise ImportError(f"Failed to import wgp module. Ensure {wan_root} contains wgp.py: {e}") from e

        # Initialize state object (mimics UI state)
        self.state = {
            "generation": {},
            "model_type": None,
            "model_filename": "",
            "advanced": False,
            "last_model_per_family": {},
            "last_resolution_per_group": {},
            "gen": {
                "queue": [],
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "prompt_no": 1,
                "prompts_max": 1
            },
            "loras": [],
            "loras_names": [],
            "loras_presets": {},
            # Additional state properties to prevent future KeyErrors
            "validate_success": 1,      # Required for validation checks
            "apply_success": 1,         # Required for settings application
            "refresh": None,            # Required for UI refresh operations
            "all_settings": {},        # Required for settings persistence
            "image_mode_tab": 0,       # Required for image/video mode switching
            "prompt": ""               # Required for prompt handling
        }

        # Apply sensible defaults (mirrors typical UI defaults)
        if not self.smoke_mode:
            # Ensure model_type is set before applying config, as upstream generate_header/get_overridden_attention
            # will look up the current model's definition via state["model_type"].
            try:
                _w = get_wgp_runtime_module()
                default_model_type = getattr(_w, "transformer_type", None) or "t2v"
            except (ImportError, AttributeError):
                default_model_type = "t2v"
            self.state["model_type"] = default_model_type

            # Apply headless defaults directly to wgp.server_config and module globals.
            # (apply_changes was removed upstream; we set the same values it used to.)
            outputs_dir = "outputs/"
            try:
                _wgp_cfg = get_wgp_runtime_module_mutable()
                _cfg = _wgp_cfg.server_config
                _cfg["transformer_types"] = ["t2v"]
                _cfg["transformer_dtype_policy"] = "auto"
                _cfg["text_encoder_quantization"] = "bf16"
                _cfg["vae_precision"] = "fp32"
                _cfg["mixed_precision"] = 0
                _cfg["save_path"] = outputs_dir
                _cfg["image_save_path"] = outputs_dir
                _cfg["attention_mode"] = "auto"
                _cfg["compile"] = ""
                _cfg["profile"] = 1  # Profile 1 for 24GB+ VRAM (4090/3090)
                _cfg["vae_config"] = 0
                _cfg["metadata_type"] = "none"
                _cfg["transformer_quantization"] = "int8"
                _cfg["preload_model_policy"] = []
                # Sync module-level globals that other wgp code reads directly
                for attr, val in [
                    ("transformer_types", ["t2v"]),
                    ("transformer_dtype_policy", "auto"),
                    ("text_encoder_quantization", "bf16"),
                    ("save_path", outputs_dir),
                    ("image_save_path", outputs_dir),
                    ("attention_mode", "auto"),
                    ("compile", ""),
                    ("default_profile", 1),
                    ("vae_config", 0),
                    ("transformer_quantization", "int8"),
                    ("preload_model_policy", []),
                    ("boost", 1),
                ]:
                    if hasattr(_wgp_cfg, attr):
                        setattr(_wgp_cfg, attr, val)
                orchestrator_logger.debug("Headless WGP defaults applied to server_config")
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                orchestrator_logger.error(f"FATAL: Failed to apply WGP defaults: {e}\n{traceback.format_exc()}")
                raise RuntimeError(f"Failed to apply WGP defaults during orchestrator init: {e}") from e

            # Verify directory after config update
            _verify_wgp_directory(orchestrator_logger, "after applying headless defaults")

            # Set output paths again (second call — after config update)
            _set_wgp_output_paths(wgp, absolute_outputs_path)
            orchestrator_logger.debug_anomaly("OUTPUT_DIR", f"Re-applied output paths after config update: {absolute_outputs_path}")

        else:
            # Provide stubbed helpers for smoke mode
            self._get_base_model_type = lambda model_key: ("t2v" if "flux" not in (model_key or "") else "flux")
            self._get_model_family = lambda model_key, for_ui=False: ("VACE" if "vace" in (model_key or "") else ("Flux" if "flux" in (model_key or "") else "T2V"))
            self._test_vace_module = lambda model_name: ("vace" in (model_name or ""))
        self.current_model = None
        self.offloadobj = None  # Store WGP's offload object
        self.passthrough_mode = False  # Flag for explicit passthrough mode
        self._cached_uni3c_controlnet = None  # Cached Uni3C ControlNet to avoid reloading from disk

        orchestrator_logger.success(f"WanOrchestrator initialized with WGP at {wan_root}")

        # Final notice for smoke mode
        if self.smoke_mode:
            orchestrator_logger.warning("HEADLESS_WAN2GP_SMOKE enabled: generation will return sample outputs only")

    # ------------------------------------------------------------------
    # Delegate wrappers to already-extracted modules
    # ------------------------------------------------------------------

    def _load_missing_model_definition(self, model_key: str, json_path: str):
        """Delegate to source.models.wgp.model_ops."""
        runtime = getattr(self, "model_runtime", None)
        if runtime is not None:
            return runtime.load_missing_model_definition(model_key, json_path)
        return _load_missing_model_definition_impl(self, model_key, json_path)

    def load_model(self, model_key: str) -> bool:
        """Load and validate a model type using WGP's exact generation-time pattern.

        Delegates to source.models.wgp.model_ops.load_model_impl.
        """
        runtime = getattr(self, "model_runtime", None)
        if runtime is not None:
            return runtime.load_model(model_key)
        return _load_model_impl(self, model_key)

    def unload_model(self):
        """Unload the current model. Delegates to source.models.wgp.model_ops."""
        runtime = getattr(self, "model_runtime", None)
        if runtime is not None:
            return runtime.unload_model()
        return _unload_model_impl(self)

    def _setup_loras_for_model(self, model_type: str):
        """Initialize LoRA discovery. Delegates to source.models.wgp.lora_setup."""
        runtime = getattr(self, "model_runtime", None)
        if runtime is not None:
            return runtime.setup_loras_for_model(model_type)
        return _setup_loras_for_model_impl(self, model_type)

    def _create_vace_fixed_generate_video(self, original_generate_video):
        """Delegate to source.models.wgp.generation_helpers."""
        return _create_vace_fixed_generate_video_impl(original_generate_video)

    def _is_vace(self) -> bool:
        """Check if current model is a VACE model. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.is_current_model_vace()
        return _is_vace_impl(self)

    def is_model_vace(self, model_name: str) -> bool:
        """Check if a given model name is a VACE model (model-agnostic). Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.is_model_vace(model_name)
        return _is_model_vace_impl(self, model_name)

    def _is_flux(self) -> bool:
        """Check if current model is a Flux model. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.is_current_model_flux()
        return _is_flux_impl(self)

    def _is_t2v(self) -> bool:
        """Check if current model is a T2V model. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.is_current_model_t2v()
        return _is_t2v_impl(self)

    def _is_qwen(self) -> bool:
        """Check if current model is a Qwen image model. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.is_current_model_qwen()
        return _is_qwen_impl(self)

    def _get_or_load_uni3c_controlnet(self):
        """Get cached Uni3C controlnet. Delegates to source.models.wgp.model_ops."""
        runtime = getattr(self, "model_runtime", None)
        if runtime is not None:
            return runtime.get_or_load_uni3c_controlnet()
        return _get_or_load_uni3c_controlnet_impl(self)

    def _load_image(self, path: Optional[str], mask: bool = False):
        """Load image from path. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.load_image(path, mask=mask)
        return _load_image_impl(self, path, mask=mask)

    def _resolve_media_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve media paths. Delegates to source.models.wgp.generation_helpers."""
        resolver = getattr(self, "media_resolver", None)
        if resolver is not None:
            return resolver.resolve_media_path(path)
        return _resolve_media_path_impl(self, path)

    def _resolve_parameters(self, model_type: str, task_params: dict) -> dict:
        """Resolve generation parameters. Delegates to source.models.wgp.param_resolution."""
        resolver = getattr(self, "parameter_resolver", None)
        if resolver is not None:
            return resolver.resolve_parameters(model_type, task_params)
        return _resolve_parameters_impl(self, model_type, task_params)

    # ------------------------------------------------------------------
    # Core generation method
    # ------------------------------------------------------------------

    def generate(self,
                prompt: str,
                model_type: str = None,
                # Common parameters - None means "use model/system defaults"
                resolution: Optional[str] = None,
                video_length: Optional[int] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                seed: Optional[int] = None,
                # VACE parameters
                video_guide: Optional[str] = None,
                video_mask: Optional[str] = None,
                video_prompt_type: Optional[str] = None,
                control_net_weight: Optional[float] = None,
                control_net_weight2: Optional[float] = None,
                # Flux parameters
                embedded_guidance_scale: Optional[float] = None,
                # LoRA parameters
                lora_names: Optional[List[str]] = None,
                lora_multipliers: Optional[List] = None,  # Can be List[float] or List[str] for phase-config
                # Other parameters
                negative_prompt: Optional[str] = None,
                batch_size: Optional[int] = None,
                **kwargs) -> str:
        """Generate content using the loaded model.

        Args:
            prompt: Text prompt for generation
            resolution: Output resolution (e.g., "1280x720", "1024x1024")
            video_length: Number of frames for video or images for Flux
            num_inference_steps: Denoising steps
            guidance_scale: CFG guidance strength
            seed: Random seed for reproducibility
            video_guide: Path to control video (required for VACE)
            video_mask: Path to mask video (optional)
            video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA")
            control_net_weight: Strength for first VACE encoding
            control_net_weight2: Strength for second VACE encoding
            embedded_guidance_scale: Flux-specific guidance
            lora_names: List of LoRA filenames
            lora_multipliers: List of LoRA strength multipliers
            negative_prompt: Negative prompt text
            batch_size: Batch size for generation
            **kwargs: Additional parameters passed to generate_video

        Returns:
            Path to generated output file(s)

        Raises:
            RuntimeError: If no model is loaded
            ValueError: If required parameters are missing for model type
        """
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # SVI / Image-Refs Path Bridging
        prepare_svi_image_refs(kwargs)

        # Smoke-mode short-circuit
        if self.smoke_mode:
            return self._generate_smoke(prompt)

        # Use provided model_type or current loaded model
        effective_model_type = model_type or self.current_model

        # Build task explicit parameters: only non-None values are considered "explicit"
        task_explicit_params = self._build_task_params(
            prompt=prompt, resolution=resolution, video_length=video_length,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            seed=seed, video_guide=video_guide, video_mask=video_mask,
            video_prompt_type=video_prompt_type, control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            embedded_guidance_scale=embedded_guidance_scale,
            lora_names=lora_names, lora_multipliers=lora_multipliers,
            negative_prompt=negative_prompt, batch_size=batch_size, **kwargs,
        )
        generation_logger.debug_block(
            "SETUP",
            {
                "model": self.current_model,
                "effective_model": effective_model_type,
                "prompt_len": len(prompt or ""),
                "resolution": resolution,
                "video_length": video_length,
                "video_guide": bool(video_guide),
                "video_mask": bool(video_mask),
                "kwargs_keys": sorted(kwargs.keys()),
            },
        )

        # Initialize LoRA variables (needed for both modes)
        activated_loras = []
        loras_multipliers_str = ""

        # Resolve final parameters with proper precedence (skip in passthrough mode)
        if self.passthrough_mode:
            resolved_params = task_explicit_params.copy()
            generation_logger.debug_anomaly("PASSTHROUGH", f"Using task parameters directly without resolution: {len(resolved_params)} params")
        else:
            try:
                resolved_params = self._resolve_parameters(effective_model_type, task_explicit_params)
            except Exception as e:
                generation_logger.critical(f"[PARAM_RESOLUTION] FAILED with {type(e).__name__}: {e}")
                import traceback
                generation_logger.critical(f"[PARAM_RESOLUTION] Traceback:\n{traceback.format_exc()}")
                raise
        generation_logger.debug_block(
            "PARAMS",
            {
                "passthrough": self.passthrough_mode,
                "explicit_keys": sorted(task_explicit_params.keys()),
                "resolved_keys": sorted(resolved_params.keys()),
                "resolved_resolution": resolved_params.get("resolution"),
                "resolved_video_length": resolved_params.get("video_length"),
                "resolved_steps": resolved_params.get("num_inference_steps"),
            },
        )

        # Determine model types for generation
        _base_model_type = self._get_base_model_type(self.current_model)
        _vace_test_result = self._test_vace_module(self.current_model)

        is_vace = self._is_vace()
        is_flux = self._is_flux()
        is_qwen = self._is_qwen()
        is_t2v = self._is_t2v()

        generation_logger.debug_block(
            "MODEL_DETECT",
            {
                "model": self.current_model,
                "effective_model": effective_model_type,
                "is_vace": is_vace,
                "is_flux": is_flux,
                "is_qwen": is_qwen,
                "is_t2v": is_t2v,
                "resolution": resolution,
                "video_length": video_length,
            },
        )

        if is_vace:
            if not video_prompt_type:
                video_prompt_type = "VP"
            if control_net_weight is None:
                control_net_weight = 1.0
            if control_net_weight2 is None:
                control_net_weight2 = 1.0
        elif video_guide or video_mask:
            # Non-VACE model with guide/mask: default control weights to avoid NoneType crash in wgp.py
            if control_net_weight is None:
                control_net_weight = 1.0
            if control_net_weight2 is None:
                control_net_weight2 = 1.0

        # Resolve media paths before validation
        video_guide = self._resolve_media_path(video_guide)
        video_mask = self._resolve_media_path(video_mask)

        if is_vace and not video_guide:
            raise ValueError("VACE models require video_guide parameter")

        # Extract resolved parameters
        final_video_length = resolved_params.get("video_length", 49)
        final_batch_size = resolved_params.get("batch_size", 1)
        final_guidance_scale = resolved_params.get("guidance_scale", 7.5)
        final_embedded_guidance = resolved_params.get("embedded_guidance_scale", 3.0)

        # Configure model-specific parameters (delegated to preflight module)
        model_params = configure_model_specific_params(
            is_flux=is_flux, is_qwen=is_qwen, is_vace=is_vace,
            resolved_params=resolved_params,
            final_video_length=final_video_length,
            final_batch_size=final_batch_size,
            final_guidance_scale=final_guidance_scale,
            final_embedded_guidance=final_embedded_guidance,
            video_guide=video_guide, video_mask=video_mask,
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            model_type=effective_model_type,
        )
        image_mode = model_params["image_mode"]
        actual_video_length = model_params["actual_video_length"]
        actual_batch_size = model_params["actual_batch_size"]
        actual_guidance = model_params["actual_guidance"]
        video_guide = model_params["video_guide"]
        video_mask = model_params["video_mask"]
        video_prompt_type = model_params["video_prompt_type"]
        control_net_weight = model_params["control_net_weight"]
        control_net_weight2 = model_params["control_net_weight2"]

        is_passthrough_mode = self.passthrough_mode

        if not is_passthrough_mode:
            # Normal mode: format LoRAs for WGP
            activated_loras = lora_names or kwargs.get('activated_loras') or []
            eff_multipliers = lora_multipliers or kwargs.get('lora_multipliers') or kwargs.get('loras_multipliers')
            if eff_multipliers:
                if isinstance(eff_multipliers, str):
                    loras_multipliers_str = eff_multipliers.replace(',', ' ')
                else:
                    loras_multipliers_str = " ".join(str(m) for m in eff_multipliers)
            else:
                loras_multipliers_str = ""

        # Build WGP parameter dictionary (delegated to wgp_params module)
        if is_passthrough_mode:
            wgp_params = build_passthrough_params(
                state=self.state, current_model=self.current_model,
                image_mode=image_mode, resolved_params=resolved_params,
                video_guide=video_guide, video_mask=video_mask,
                video_prompt_type=video_prompt_type,
                control_net_weight=control_net_weight,
                control_net_weight2=control_net_weight2,
            )
        else:
            wgp_params = build_normal_params(
                state=self.state, current_model=self.current_model,
                image_mode=image_mode, resolved_params=resolved_params,
                prompt=prompt, actual_video_length=actual_video_length,
                actual_batch_size=actual_batch_size,
                actual_guidance=actual_guidance,
                final_embedded_guidance=final_embedded_guidance,
                is_flux=is_flux, video_guide=video_guide,
                video_mask=video_mask, video_prompt_type=video_prompt_type,
                control_net_weight=control_net_weight,
                control_net_weight2=control_net_weight2,
                activated_loras=activated_loras,
                loras_multipliers_str=loras_multipliers_str,
            )

        # Override ANY parameter provided in kwargs
        apply_kwargs_overrides(wgp_params, kwargs)

        # Generate content type description
        content_type = "images" if (is_flux or is_qwen) else "video"
        model_type_desc = (
            "Flux" if is_flux else ("Qwen" if is_qwen else ("VACE" if is_vace else "T2V"))
        )
        count_desc = f"{video_length} {'images' if is_flux else 'frames'}"

        generation_logger.debug(f"Generating {model_type_desc} {content_type}: {resolution}, {count_desc}")
        # Initialize capture variables at outer scope for exception handling
        captured_stdout = None
        captured_stderr = None
        captured_logs = []

        try:
            encodings = [c for c in video_prompt_type if c in "VPDSLCMUA"] if is_vace and video_prompt_type else []
            generation_request_summary = {
                "model": self.current_model,
                "path": "vace" if is_vace else ("flux" if is_flux else "t2v"),
                "passthrough": is_passthrough_mode,
                "actual_video_length": actual_video_length,
                "actual_batch_size": actual_batch_size,
                "guidance_scale": actual_guidance,
                "guidance2_scale": wgp_params.get("guidance2_scale", "NOT_SET"),
                "video_prompt_type": video_prompt_type,
                "vace_encodings": encodings,
                "video_guide": bool(video_guide),
                "video_mask": bool(video_mask),
                "lora_count": len(activated_loras),
                "lora_source": "lora_names" if lora_names else ("kwargs.activated_loras" if activated_loras else "none"),
            }
            generation_logger.debug_block("GENERATE", generation_request_summary)

            # Pre-populate WGP UI state for LoRA compatibility
            original_loras = self.state.get("loras", [])
            if activated_loras and len(activated_loras) > 0:
                self._log_lora_application_trace(activated_loras, loras_multipliers_str, lora_names)
                self.state["loras"] = activated_loras.copy()

            try:
                # Pre-initialize WGP process status
                try:
                    if isinstance(self.state.get("gen"), dict):
                        self.state["gen"]["process_status"] = "process:main"
                except (KeyError, TypeError):
                    pass

                # Prepare image inputs (delegated to preflight module)
                prepare_image_inputs(
                    wgp_params,
                    is_qwen=is_qwen,
                    image_mode=image_mode,
                    load_image_fn=self._load_image,
                )

                # Inject cached Uni3C controlnet if use_uni3c is enabled
                if wgp_params.get('use_uni3c') and not wgp_params.get('uni3c_controlnet'):
                    cached_controlnet = self._get_or_load_uni3c_controlnet()
                    if cached_controlnet is not None:
                        wgp_params['uni3c_controlnet'] = cached_controlnet

                # Filter out unsupported parameters to match upstream signature
                _filtered_params = self._filter_wgp_params(wgp_params)

                # COMPREHENSIVE LOGGING: Show all final parameters
                self._log_final_params(_filtered_params)

                # Execute WGP with capture (delegated to capture module)
                # Extract task_id from kwargs for logging context
                task_id_for_logging = kwargs.get("task_id") or _filtered_params.get("task_id")
                _, captured_stdout, captured_stderr, captured_logs = run_with_capture(
                    self._generate_video,
                    task_id=task_id_for_logging,
                    log_func=generation_logger.debug,
                    **_filtered_params,
                )

                # Verify directory after generation
                _verify_wgp_directory(generation_logger, "after wgp.generate_video()")

            finally:
                # Restore original UI state after generation
                if activated_loras and len(activated_loras) > 0:
                    self.state["loras"] = original_loras

            # Extract output path (delegated to output module)
            output_path = extract_output_path(
                self.state, model_type_desc,
                captured_stdout, captured_stderr, captured_logs,
            )
            generation_logger.debug_block(
                "OUTPUT",
                {
                    "model": self.current_model,
                    "content_type": content_type,
                    "output_path": output_path,
                },
            )

            # Memory monitoring (delegated to output module)
            log_memory_stats()

            return output_path

        except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
            generation_logger.error(f"Generation failed: {e}")
            try:
                # If run_with_capture raised, captured output is attached to the exception
                exc_stdout = getattr(e, '__captured_stdout__', None) or captured_stdout
                exc_stderr = getattr(e, '__captured_stderr__', None) or captured_stderr
                exc_logs = getattr(e, '__captured_logs__', None) or captured_logs
                log_captured_output(exc_stdout, exc_stderr, exc_logs)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass  # Don't let logging errors mask the original exception
            try:
                from source.core.log import flush_log_buffer
                flush_log_buffer()
            except (ImportError, AttributeError, OSError):
                pass  # Don't let flush errors mask the original exception
            raise

    # ------------------------------------------------------------------
    # Private helpers (keep generate() readable)
    # ------------------------------------------------------------------

    def _build_task_params(self, *, prompt, resolution, video_length,
                           num_inference_steps, guidance_scale, seed,
                           video_guide, video_mask, video_prompt_type,
                           control_net_weight, control_net_weight2,
                           embedded_guidance_scale, lora_names, lora_multipliers,
                           negative_prompt, batch_size, **kwargs) -> dict:
        """Collect non-None explicit parameters into a task dict."""
        task_explicit_params = {"prompt": prompt}
        param_values = {
            "resolution": resolution,
            "video_length": video_length,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "video_guide": self._resolve_media_path(video_guide),
            "video_mask": self._resolve_media_path(video_mask),
            "video_prompt_type": video_prompt_type,
            "control_net_weight": control_net_weight,
            "control_net_weight2": control_net_weight2,
            "embedded_guidance_scale": embedded_guidance_scale,
            "lora_names": lora_names,
            "lora_multipliers": lora_multipliers,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
        }
        for param, value in param_values.items():
            if value is not None:
                task_explicit_params[param] = value
        task_explicit_params.update(kwargs)
        return task_explicit_params

    def _generate_smoke(self, prompt: str) -> str:
        """Smoke-mode short-circuit: return a sample output path."""
        from pathlib import Path
        import shutil
        out_dir = Path(os.getcwd()) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        _project_root = Path(__file__).parent.parent.parent.parent
        sample_src = Path(os.path.abspath(os.path.join(_project_root, "samples", "test.mp4")))
        if not sample_src.exists():
            sample_src = Path(os.path.abspath(os.path.join(_project_root, "samples", "video.mp4")))
        ts = __import__("time").strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"smoke_{self.current_model}_{ts}.mp4"
        try:
            shutil.copyfile(str(sample_src), str(out_path))
        except OSError as e:
            generation_logger.debug_anomaly(
                "SMOKE",
                f"Failed to copy sample file {sample_src}, creating empty placeholder: {e}",
            )
            out_path.write_bytes(b"")
        try:
            self.state["gen"]["file_list"].append(str(out_path))
        except (KeyError, TypeError) as e:
            generation_logger.debug_anomaly("SMOKE", f"Failed to update state gen file_list: {e}")
        generation_logger.debug_anomaly("SMOKE", f"Generated placeholder output at: {out_path}")
        return str(out_path)

    def _filter_wgp_params(self, wgp_params: dict) -> dict:
        """Filter WGP params to only those accepted by generate_video()."""
        try:
            import inspect as _inspect
            _wgp = get_wgp_runtime_module()
            _sig = _inspect.signature(_wgp.generate_video)
            _allowed = set(_sig.parameters.keys())
            _filtered = {k: v for k, v in wgp_params.items() if k in _allowed}
            _dropped = sorted(set(wgp_params.keys()) - _allowed)
            if _dropped:
                generation_logger.debug_anomaly("PARAM_SANITIZE", f"Dropping unsupported params: {_dropped}")
            return _filtered
        except (ValueError, TypeError, RuntimeError) as _e:
            return wgp_params

    def _log_final_params(self, filtered_params: dict) -> None:
        """Log all final parameters being sent to WGP generate_video()."""
        state = filtered_params.get("state") if isinstance(filtered_params.get("state"), dict) else {}
        final_param_summary = {
            "keys": sorted(filtered_params.keys()),
            "resolution": filtered_params.get("resolution"),
            "video_length": filtered_params.get("video_length"),
            "num_inference_steps": filtered_params.get("num_inference_steps"),
            "guidance_scale": filtered_params.get("guidance_scale"),
            "guidance2_scale": filtered_params.get("guidance2_scale"),
            "video_prompt_type": filtered_params.get("video_prompt_type"),
            "use_uni3c": filtered_params.get("use_uni3c"),
            "state": {
                "model_type": state.get("model_type"),
                "gen_file_count": len((state.get("gen") or {}).get("file_list", [])) if isinstance(state.get("gen"), dict) else 0,
                "loras_count": len(state.get("loras") or []),
            },
        }
        generation_logger.debug_block("GENERATE", final_param_summary)

    def _log_lora_application_trace(self, activated_loras: list, loras_multipliers_str: str, lora_names) -> None:
        """Log detailed LoRA application breakdown."""
        multiplier_list = loras_multipliers_str.split() if loras_multipliers_str else []
        lora_summary = {
            "count": len(activated_loras),
            "filenames": [os.path.basename(str(path)) for path in activated_loras],
            "multipliers": multiplier_list,
            "amount_of_motion_lora": bool(lora_names and len(lora_names) > 0),
        }
        generation_logger.debug_anomaly("LORA_APPLICATION_TRACE", f"{safe_dict_repr(lora_summary)}")

    # ------------------------------------------------------------------
    # Convenience methods for specific generation types
    # (delegated to source/models/wgp/generators/)
    # ------------------------------------------------------------------

    def generate_t2v(self, prompt: str, model_type: str = None, **kwargs) -> str:
        """Generate text-to-video content. Delegates to generators.t2v."""
        return _generate_t2v_impl(self, prompt=prompt, model_type=model_type, **kwargs)

    def generate_with_config(self, config: 'TaskConfig') -> str:
        """Generate content using a typed TaskConfig object. Delegates to generators.qwen."""
        return _generate_with_config_impl(self, config)

    def generate_vace(self,
                     prompt: str,
                     video_guide: str,
                     model_type: str = None,
                     video_mask: Optional[str] = None,
                     video_prompt_type: str = "VP",
                     control_net_weight: float = 1.0,
                     control_net_weight2: float = 1.0,
                     **kwargs) -> str:
        """Generate VACE controlled video content. Delegates to generators.vace."""
        return _generate_vace_impl(
            self, prompt=prompt, video_guide=video_guide,
            model_type=model_type, video_mask=video_mask,
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            **kwargs,
        )

    def generate_flux(self, prompt: str, images: int = 4, model_type: str = None, **kwargs) -> str:
        """Generate Flux images. Delegates to generators.flux."""
        return _generate_flux_impl(self, prompt=prompt, images=images, model_type=model_type, **kwargs)


# Backward compatibility
WanContentOrchestrator = WanOrchestrator
