"""
WGP monkeypatches for headless operation.

These patches adapt WGP (Wan2GP) for headless/programmatic use without
modifying the upstream Wan2GP repository files directly.

Each patch is documented with:
- Why it's needed
- What it modifies
- When it should be applied
"""

import os
from typing import TYPE_CHECKING

from source.core.log import model_logger
from source.runtime.wgp_bridge import (
    get_qwen_family_handler,
    get_qwen_main_module,
    get_shared_lora_utils_module,
)

if TYPE_CHECKING:
    import types


_DEFAULT_PATCH_CONTEXT_ID = "default"
_PATCH_CONTEXT_STATE: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
_PATCH_CONTEXT_ROLLBACKS: dict[str, dict[str, dict[str, object]]] = {}


def _normalize_patch_context_id(context_id: str | None = None) -> str:
    normalized = context_id or _DEFAULT_PATCH_CONTEXT_ID
    return str(normalized)


def _context_patch_state(context_id: str | None = None) -> dict[str, dict[str, dict[str, object]]]:
    return _PATCH_CONTEXT_STATE.setdefault(_normalize_patch_context_id(context_id), {})


def _context_patch_rollbacks(context_id: str | None = None) -> dict[str, dict[str, object]]:
    return _PATCH_CONTEXT_ROLLBACKS.setdefault(_normalize_patch_context_id(context_id), {})


def _register_patch_application(
    patch_name: str,
    target_key: str,
    *,
    applied: bool,
    context_id: str | None = None,
    rollback=None,
) -> None:
    state_bucket = _context_patch_state(context_id).setdefault(patch_name, {})
    state_bucket[target_key] = {"applied": bool(applied)}
    if rollback is not None:
        rollback_bucket = _context_patch_rollbacks(context_id).setdefault(patch_name, {})
        rollback_bucket[target_key] = rollback


def get_wgp_patch_state(*, context_id: str | None = None) -> dict[str, dict[str, dict[str, object]]]:
    state = _PATCH_CONTEXT_STATE.get(_normalize_patch_context_id(context_id), {})
    return {
        patch_name: {
            target_key: dict(metadata)
            for target_key, metadata in patch_entries.items()
        }
        for patch_name, patch_entries in state.items()
    }


def clear_wgp_patch_context(context_id: str | None = None) -> None:
    normalized = _normalize_patch_context_id(context_id)
    _PATCH_CONTEXT_STATE.pop(normalized, None)
    _PATCH_CONTEXT_ROLLBACKS.pop(normalized, None)


def rollback_wgp_patches(
    *,
    patch_name: str | None = None,
    target_key: str | None = None,
    context_id: str | None = None,
) -> int:
    if context_id is not None:
        contexts = [_normalize_patch_context_id(context_id)]
    else:
        contexts = sorted(set(_PATCH_CONTEXT_ROLLBACKS.keys()) | set(_PATCH_CONTEXT_STATE.keys()))
    restored = 0

    for normalized_context in contexts:
        rollback_state = _PATCH_CONTEXT_ROLLBACKS.get(normalized_context, {})
        state = _PATCH_CONTEXT_STATE.get(normalized_context, {})

        patch_names = [patch_name] if patch_name is not None else list(rollback_state.keys())
        for current_patch_name in patch_names:
            patch_rollbacks = rollback_state.get(current_patch_name, {})
            target_keys = [target_key] if target_key is not None else list(patch_rollbacks.keys())
            for current_target_key in target_keys:
                rollback = patch_rollbacks.pop(current_target_key, None)
                if callable(rollback):
                    restored += int(rollback())
                patch_state = state.get(current_patch_name, {})
                patch_state.pop(current_target_key, None)
                if not patch_state:
                    state.pop(current_patch_name, None)
            if not patch_rollbacks:
                rollback_state.pop(current_patch_name, None)

        if not rollback_state:
            _PATCH_CONTEXT_ROLLBACKS.pop(normalized_context, None)
        if not state:
            _PATCH_CONTEXT_STATE.pop(normalized_context, None)

    return restored


def apply_qwen_model_routing_patch(wgp_module: "types.ModuleType", wan_root: str) -> bool:
    """
    Patch WGP to route Qwen-family models to their dedicated handler.

    Why: WGP's load_wan_model doesn't natively handle Qwen models.
    This routes Qwen-family models (qwen_image_edit, etc.) to the
    dedicated Qwen handler in models/qwen/qwen_handler.py.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _orig_load_wan_model = wgp_module.load_wan_model

        def _patched_load_wan_model(
            model_filename,
            model_type,
            base_model_type,
            model_def,
            quantizeTransformer=False,
            dtype=None,
            VAE_dtype=None,
            mixed_precision_transformer=False,
            save_quantized=False
        ):
            # Check if this is a Qwen-family model
            try:
                base = wgp_module.get_base_model_type(base_model_type)
            except (AttributeError, ValueError, TypeError):
                base = base_model_type

            if isinstance(base, str) and "qwen" in base.lower():
                model_logger.debug_anomaly("QWEN_LOAD", "Routing to Qwen family loader via monkeypatch")
                _qwen_handler = get_qwen_family_handler()
                pipe_processor, pipe = _qwen_handler.load_model(
                    model_filename=model_filename,
                    model_type=model_type,
                    base_model_type=base_model_type,
                    model_def=model_def,
                    quantizeTransformer=quantizeTransformer,
                    text_encoder_quantization=wgp_module.text_encoder_quantization,
                    dtype=dtype,
                    VAE_dtype=VAE_dtype,
                    mixed_precision_transformer=mixed_precision_transformer,
                    save_quantized=save_quantized,
                )
                return pipe_processor, pipe

            # Fallback to original WAN loader
            return _orig_load_wan_model(
                model_filename, model_type, base_model_type, model_def,
                quantizeTransformer=quantizeTransformer, dtype=dtype, VAE_dtype=VAE_dtype,
                mixed_precision_transformer=mixed_precision_transformer, save_quantized=save_quantized
            )

        wgp_module.load_wan_model = _patched_load_wan_model
        return True

    except (RuntimeError, AttributeError, ImportError) as e:
        model_logger.debug_anomaly("QWEN_LOAD", f"Failed to apply load_wan_model patch: {e}")
        return False


def apply_qwen_lora_directory_patch(wgp_module: "types.ModuleType", wan_root: str) -> bool:
    """
    Patch get_lora_dir to redirect Qwen models to loras_qwen/ directory.

    Why: Qwen models need their own LoRA directory (loras_qwen/) since they
    use a different LoRA format than WAN models.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        target_key = f"wgp_module:{id(wgp_module)}:get_lora_dir"
        existing = get_wgp_patch_state().get("qwen_lora_directory", {})
        if target_key in existing and getattr(wgp_module.get_lora_dir, "_headless_patch_target_key", None) == target_key:
            return True

        _orig_get_lora_dir = wgp_module.get_lora_dir

        def _patched_get_lora_dir(model_type: str):
            try:
                mt = (model_type or "").lower()
                if "qwen" in mt:
                    qwen_dir = os.path.join(wan_root, "loras_qwen")
                    if os.path.isdir(qwen_dir):
                        return qwen_dir
            except (ValueError, TypeError, OSError) as e:
                model_logger.debug_anomaly("QWEN_LOAD", f"Error checking Qwen LoRA directory for model_type={model_type}: {e}")
            return _orig_get_lora_dir(model_type)

        _patched_get_lora_dir._headless_patch_name = "qwen_lora_directory"
        _patched_get_lora_dir._headless_patch_target_key = target_key
        wgp_module.get_lora_dir = _patched_get_lora_dir
        _register_patch_application(
            "qwen_lora_directory",
            target_key,
            applied=True,
            rollback=lambda: _rollback_qwen_lora_directory_patch(wgp_module, target_key, _orig_get_lora_dir),
        )
        return True

    except (RuntimeError, AttributeError, ImportError) as e:
        model_logger.debug_anomaly("QWEN_LOAD", f"Failed to apply get_lora_dir patch: {e}")
        return False


def apply_lora_multiplier_parser_patch(wgp_module: "types.ModuleType") -> bool:
    """
    Harmonize LoRA multiplier parsing across pipelines.

    Why: Use the 3-phase capable parser so Qwen pipeline (which expects
    phase3/shared) receives a compatible slists_dict. This is backward
    compatible for 2-phase models.

    Args:
        wgp_module: The imported wgp module

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _shared_lora_utils = get_shared_lora_utils_module()
        wgp_module.parse_loras_multipliers = _shared_lora_utils.parse_loras_multipliers
        wgp_module.preparse_loras_multipliers = _shared_lora_utils.preparse_loras_multipliers
        return True

    except (ImportError, AttributeError) as e:
        model_logger.debug_anomaly("QWEN_LOAD", f"Failed to apply lora parser patch: {e}")
        return False


def apply_qwen_inpainting_lora_patch() -> bool:
    """
    Disable Qwen's built-in inpainting LoRA (preload_URLs) in headless mode.

    Why: Qwen models have a built-in inpainting LoRA that gets auto-loaded.
    In headless mode, this can cause issues. Disabled by default unless
    HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA=1 is set.

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _qwen_main = get_qwen_main_module()
        _orig_qwen_get_loras_transformer = _qwen_main.model_factory.get_loras_transformer

        def _patched_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs):
            try:
                if os.environ.get("HEADLESS_WAN2GP_ENABLE_QWEN_INPAINTING_LORA", "0") != "1":
                    return [], []
            except (ValueError, TypeError, OSError):
                # If env check fails, fall back to disabled behavior
                return [], []
            return _orig_qwen_get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, **kwargs)

        _qwen_main.model_factory.get_loras_transformer = _patched_qwen_get_loras_transformer
        return True

    except (ImportError, AttributeError, AssertionError, RuntimeError) as e:
        # AssertionError/RuntimeError: CUDA init fails on non-GPU machines during import
        model_logger.debug_anomaly("QWEN_LOAD", f"Failed to apply Qwen inpainting LoRA patch: {e}")
        return False


def apply_lora_key_tolerance_patch(wgp_module: "types.ModuleType") -> bool:
    """
    Patch LoRA loading to tolerate non-standard keys (e.g., lightx2v format).

    Why: Some LoRA files (like lightx2v/Wan2.2-Distill-Loras) contain keys in
    non-standard formats (diff_m, norm_k_img) that mmgp rejects as hard errors.
    ComfyUI's loader silently skips these unrecognized keys. This patch wraps
    the LoRA preprocessor to strip unrecognized keys before mmgp's validator
    sees them, matching ComfyUI's behavior.

    How: Monkeypatches wgp.get_loras_preprocessor to wrap the returned
    preprocessor with a key-stripping step that:
    1. Removes keys with no recognized LoRA suffix (handles diff_m, etc.)
    2. Removes keys whose module path doesn't exist in the model (handles
       norm_k_img and similar non-standard module references)

    Args:
        wgp_module: The imported wgp module

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        _orig_get_loras_preprocessor = wgp_module.get_loras_preprocessor

        # Valid LoRA key suffixes that mmgp's validator recognizes
        # Order: longest first to avoid partial matches
        VALID_LORA_SUFFIXES = [
            '.lora_down.weight',
            '.lora_up.weight',
            '.lora_A.weight',
            '.lora_B.weight',
            '.dora_scale',
            '.diff_b',
            '.diff',
        ]
        KNOWN_PREFIXES = ("diffusion_model.", "transformer.")

        def _patched_get_loras_preprocessor(transformer, model_type):
            orig_preproc = _orig_get_loras_preprocessor(transformer, model_type)

            # Build set of known module names for unexpected-key detection
            modules_set = set(name for name, _ in transformer.named_modules())

            def _tolerant_preprocessor(sd):
                # Run original preprocessor first (if any)
                if orig_preproc is not None:
                    sd = orig_preproc(sd)

                # Detect prefix from first key (same logic as mmgp)
                first_key = next(iter(sd), None)
                if first_key is None:
                    return sd

                pos = first_key.find(".")
                prefix = first_key[:pos + 1] if pos > 0 else ""
                if prefix not in KNOWN_PREFIXES:
                    prefix = ""

                keys_to_remove = []
                for k in sd:
                    # Skip alpha keys (handled separately by mmgp before validation)
                    if k.endswith('.alpha'):
                        continue

                    # Find matching suffix and extract module name
                    module_name = None
                    for suffix in VALID_LORA_SUFFIXES:
                        if k.endswith(suffix):
                            raw_module = k[:-len(suffix)]
                            if prefix and raw_module.startswith(prefix):
                                module_name = raw_module[len(prefix):]
                            else:
                                module_name = raw_module
                            break

                    if module_name is None:
                        # No valid LoRA suffix found - would be invalid_keys in mmgp
                        keys_to_remove.append(k)
                        continue

                    # Check if module exists in model - would be unexpected_keys in mmgp
                    if module_name not in modules_set:
                        keys_to_remove.append(k)
                        continue

                if keys_to_remove:
                    model_logger.debug_anomaly(
                        "LORA_TOLERANCE",
                        f"Stripping {len(keys_to_remove)} non-standard/incompatible keys from LoRA: "
                        f"{keys_to_remove[:5]}{'...' if len(keys_to_remove) > 5 else ''}"
                    )
                    for k in keys_to_remove:
                        del sd[k]

                return sd

            return _tolerant_preprocessor

        wgp_module.get_loras_preprocessor = _patched_get_loras_preprocessor
        return True

    except (RuntimeError, AttributeError, ImportError) as e:
        model_logger.debug_anomaly("LORA_TOLERANCE", f"Failed to apply LoRA key tolerance patch: {e}")
        return False


def apply_lora_caching_patch() -> bool:
    """
    Cache LoRA loading between generations to avoid redundant reloads.

    Why: WGP unloads LoRAs after every generation (wgp.py:6439) and mmgp's
    load_loras_into_model always unloads before loading. For a 20B Qwen model
    with CPU offloading, LoRA loading takes ~7 minutes per generation. When
    running the same LoRAs repeatedly (batch qwen_image tasks, travel segments),
    this is pure waste — 21 min of LoRA loading for 2.5 min of actual generation.

    How:
    - Patches offload.unload_loras_from_model to be a no-op (keeps LoRAs loaded)
    - Patches offload.load_loras_into_model to skip if same LoRA files are
      already loaded on the model, otherwise does a real unload + load cycle
    - Cache is keyed on (model object identity, LoRA file paths)
    - Safety: checks model._loras_adapters to verify LoRAs are actually loaded
      (catches cases where error handlers cleared state)

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        from mmgp import offload

        target_key = f"mmgp.offload:{id(offload)}"
        existing = get_wgp_patch_state().get("lora_caching", {})
        if target_key in existing and getattr(offload.load_loras_into_model, "_headless_patch_target_key", None) == target_key:
            return True

        _original_load = offload.load_loras_into_model
        _original_unload = offload.unload_loras_from_model

        def _cached_load(model, lora_path, lora_multi=None, **kwargs):
            # check_only mode (used by LoRA validation) — pass through
            if kwargs.get("check_only", False):
                return _original_load(model, lora_path, lora_multi, **kwargs)

            # Normalize to tuple for comparison
            paths = lora_path if isinstance(lora_path, list) else [lora_path]
            requested = tuple(paths)

            # Check cache: same model object + same LoRA files + LoRAs actually loaded
            cached = getattr(model, "_headless_cached_lora_paths", None)
            if (
                cached is not None
                and cached == requested
                and getattr(model, "_loras_adapters", None) is not None
            ):
                model_logger.debug(
                    f"[LORA_CACHE] Skipping LoRA reload — same {len(requested)} LoRAs already loaded"
                )
                return

            # Different LoRAs or first load — do full cycle
            if cached is not None:
                model_logger.debug(
                    f"[LORA_CACHE] LoRAs changed ({len(cached)} -> {len(requested)}), reloading"
                )
            else:
                model_logger.debug_anomaly("LORA_CACHE", f"First LoRA load: {len(requested)} LoRAs")

            # Explicitly unload via original (our global patch is a no-op, and
            # the internal unload inside _original_load also hits our no-op,
            # so we must call the real one here)
            _original_unload(model)

            result = _original_load(model, lora_path, lora_multi, **kwargs)

            # Tag model with what we loaded
            model._headless_cached_lora_paths = requested
            return result

        def _deferred_unload(model):
            """No-op — keeps LoRAs loaded for reuse by next generation."""
            if model is None:
                return
            model_logger.debug_anomaly("LORA_CACHE", "Deferring LoRA unload (cached for reuse)")

        _cached_load._headless_patch_name = "lora_caching"
        _cached_load._headless_patch_target_key = target_key
        _deferred_unload._headless_patch_name = "lora_caching"
        _deferred_unload._headless_patch_target_key = target_key
        offload.load_loras_into_model = _cached_load
        offload.unload_loras_from_model = _deferred_unload
        _register_patch_application(
            "lora_caching",
            target_key,
            applied=True,
            rollback=lambda: _rollback_lora_caching_patch(offload, target_key, _original_load, _original_unload),
        )
        return True

    except (ImportError, AttributeError, RuntimeError) as e:
        model_logger.debug_anomaly("LORA_CACHE", f"Failed to apply LoRA caching patch: {e}")
        return False


def apply_headless_app_stub(wgp_module: "types.ModuleType") -> bool:
    """
    Stub the 'app' global in wgp so plugin_manager checks are skipped.

    Why: Upstream wgp.py references a global 'app' (WAN2GPApplication instance)
    in prepare_inputs_dict() for plugin hooks. 'app' is only created when wgp.py
    runs as __main__ (Gradio UI), not when imported. Without this stub,
    headless generation crashes with 'NameError: name app is not defined'.

    The stub is a SimpleNamespace with no plugin_manager attribute, so
    hasattr(app, 'plugin_manager') returns False and the plugin path is skipped.
    """
    try:
        from types import SimpleNamespace

        if not hasattr(wgp_module, "app"):
            wgp_module.app = SimpleNamespace()
            model_logger.debug_anomaly("WGP_PATCHES", "Stubbed wgp.app for headless mode (no plugin_manager)")
        return True
    except Exception as e:
        model_logger.warning(f"[WGP_PATCHES] Failed to stub wgp.app: {e}")
        return False


def _rollback_qwen_lora_directory_patch(wgp_module, target_key: str, original_get_lora_dir) -> int:
    if getattr(wgp_module.get_lora_dir, "_headless_patch_target_key", None) != target_key:
        return 0
    wgp_module.get_lora_dir = original_get_lora_dir
    return 1


def _rollback_lora_caching_patch(offload, target_key: str, original_load, original_unload) -> int:
    restored = 0
    if getattr(offload.load_loras_into_model, "_headless_patch_target_key", None) == target_key:
        offload.load_loras_into_model = original_load
        restored += 1
    if getattr(offload.unload_loras_from_model, "_headless_patch_target_key", None) == target_key:
        offload.unload_loras_from_model = original_unload
        restored += 1
    return restored


def apply_all_wgp_patches(
    wgp_module: "types.ModuleType",
    wan_root: str,
    *,
    context_id: str | None = None,
) -> dict:
    """
    Apply all WGP patches for headless operation.

    Args:
        wgp_module: The imported wgp module
        wan_root: Path to Wan2GP root directory

    Returns:
        Dict with patch names as keys and success status as values
    """
    results = {}

    results["qwen_model_routing"] = apply_qwen_model_routing_patch(wgp_module, wan_root)
    results["qwen_lora_directory"] = apply_qwen_lora_directory_patch(wgp_module, wan_root)
    results["lora_multiplier_parser"] = apply_lora_multiplier_parser_patch(wgp_module)
    results["qwen_inpainting_lora"] = apply_qwen_inpainting_lora_patch()
    results["lora_key_tolerance"] = apply_lora_key_tolerance_patch(wgp_module)
    results["lora_caching"] = apply_lora_caching_patch()
    results["headless_app_stub"] = apply_headless_app_stub(wgp_module)

    # Log summary
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    if successful:
        model_logger.debug_anomaly("WGP_PATCHES", f"Applied: {', '.join(successful)}")
    if failed:
        model_logger.debug_anomaly("WGP_PATCHES", f"Failed (non-fatal): {', '.join(failed)}")

    context_target_key = f"wgp_module:{id(wgp_module)}"
    for patch_name, applied in results.items():
        _register_patch_application(
            patch_name,
            context_target_key,
            applied=applied,
            context_id=context_id,
        )

    return results
