"""
WGP Generation Helpers

Miscellaneous helper functions used during generation:
- VACE wrapper for generate_video
- Image loading and media path resolution
- Model type checker functions
- Worker notification and directory verification
"""

import os
from typing import Optional

from source.core.log import model_logger, generation_logger


# ---------------------------------------------------------------------------
# Worker / directory helpers (module-level functions)
# ---------------------------------------------------------------------------

def notify_worker_model_switch(old_model: Optional[str], new_model: str):
    """
    Notify Supabase edge function that a model switch is about to happen.

    Called BEFORE the actual model load so the worker status can be updated
    to show it's switching models (which can take 30-60+ seconds).

    Only fires if WORKER_ID is set (i.e., running in cloud worker context).
    """
    worker_id = os.getenv("WORKER_ID")
    if not worker_id:
        return

    # Prefer service role; fall back to older env var name or PAT.
    from source.core.db import config as _db_config
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or _db_config.SUPABASE_ACCESS_TOKEN
    )
    if not supabase_key:
        model_logger.debug("No SUPABASE_SERVICE_ROLE_KEY or access token, skipping model switch notification")
        return

    try:
        import httpx
    except ImportError as e:
        model_logger.warning(f"Failed to notify model switch: {e}")
        return

    try:
        supabase_url = os.getenv("SUPABASE_URL", "https://wczysqzxlwdndgxitrvc.supabase.co")
        response = httpx.post(
            f"{supabase_url.rstrip('/')}/functions/v1/update-worker-model",
            headers={
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            },
            json={
                "worker_id": worker_id,
                # Keep payload minimal to avoid schema mismatches server-side.
                "current_model": new_model,
            },
            timeout=5.0  # Don't block model loading on this
        )

        if response.status_code == 200:
            model_logger.debug(f"Notified edge function of model switch: {old_model} → {new_model}")
        else:
            model_logger.warning(f"Edge function returned {response.status_code}: {response.text[:200]}")

    except (httpx.HTTPError, OSError, ValueError) as e:
        # Don't let notification failures block model loading
        model_logger.warning(f"Failed to notify model switch: {e}")


def verify_wgp_directory(logger, context: str = ""):
    """
    Verify we're still in the Wan2GP directory after any operation.

    Wan2GP expects to run from its directory and uses relative paths.
    Call this after any wgp operation to catch directory changes early.

    Args:
        logger: Logger instance to use
        context: Description of what just happened (e.g., "after wgp.generate_video()")
    """
    current_dir = os.getcwd()
    expected_substr = "Wan2GP"

    if expected_substr not in current_dir:
        logger.warning(
            f"[PATH_CHECK] {context}: Current directory may be wrong!\n"
            f"  Current: {current_dir}\n"
            f"  Expected: Path containing 'Wan2GP'\n"
            f"  This could cause issues with wgp.py's relative paths!"
        )
    else:
        logger.debug(f"[PATH_CHECK] {context}: Still in Wan2GP directory ✓ ({current_dir})")

    # Also verify critical dirs still accessible
    if not os.path.exists("defaults"):
        logger.error(
            f"[PATH_CHECK] {context}: CRITICAL - defaults/ no longer accessible!\n"
            f"  Current directory: {current_dir}"
        )

    return current_dir


# ---------------------------------------------------------------------------
# VACE wrapper
# ---------------------------------------------------------------------------

def create_vace_fixed_generate_video(original_generate_video):
    """Create a wrapper around generate_video for VACE models.

    VACE models are properly loaded via load_models() now, so no special handling needed.
    Also handles parameter name mapping for compatibility.

    Args:
        original_generate_video: The upstream wgp.generate_video function

    Returns:
        Wrapped generate_video function
    """
    def vace_fixed_generate_video(*args, **kwargs):
        # Map parameter names for compatibility
        if "denoise_strength" in kwargs:
            kwargs["denoising_strength"] = kwargs.pop("denoise_strength")

        # VACE modules are now properly loaded via load_models() - no patching needed
        return original_generate_video(*args, **kwargs)

    return vace_fixed_generate_video


# ---------------------------------------------------------------------------
# Image loading / media path resolution
# ---------------------------------------------------------------------------

def load_image(orchestrator, path: Optional[str], mask: bool = False):
    """Load an image from a path, resolving it via the orchestrator's media path logic.

    Args:
        orchestrator: WanOrchestrator instance (provides _resolve_media_path)
        path: Path to the image file
        mask: If True, convert to grayscale ("L") mode

    Returns:
        PIL.Image or None
    """
    if not path:
        return None
    try:
        from PIL import Image  # type: ignore
        p = resolve_media_path(orchestrator, path)
        img = Image.open(p)
        if mask:
            try:
                return img.convert("L")
            except (OSError, ValueError):
                return img
        else:
            try:
                return img.convert("RGB")
            except (OSError, ValueError):
                return img
    except (OSError, ValueError, AttributeError) as e:
        generation_logger.warning(f"Could not load image from {path}: {e}")
        return None


def resolve_media_path(orchestrator, path: Optional[str]) -> Optional[str]:
    """Resolve media paths relative to the local Headless-Wan2GP repo.

    - If the path exists as-is (absolute), return it.
    - If relative, prefer repo root (agent_tasks/Headless-Wan2GP), then Wan2GP.

    Args:
        orchestrator: WanOrchestrator instance (provides wan_root)
        path: Path to resolve

    Returns:
        Resolved path string or original path
    """
    if not path:
        return path
    try:
        from pathlib import Path
        p = Path(path)
        if p.exists():
            return str(p.resolve())
        wan_root = Path(orchestrator.wan_root)
        repo_root = wan_root.parent
        if not p.is_absolute():
            candidate = repo_root / p
            if candidate.exists():
                return str(candidate.resolve())
            candidate = wan_root / p
            if candidate.exists():
                return str(candidate.resolve())
    except (OSError, ValueError):
        pass  # Path resolution failed; fall back to returning original path
    return path


# ---------------------------------------------------------------------------
# Model type checkers
# ---------------------------------------------------------------------------

def is_vace(orchestrator) -> bool:
    """Check if current model is a VACE model."""
    return orchestrator._test_vace_module(orchestrator.current_model)


def is_model_vace(orchestrator, model_name: str) -> bool:
    """Check if a given model name is a VACE model (model-agnostic).

    This method doesn't require the model to be loaded, making it suitable
    for VACE detection during task processing when the orchestrator may not
    have the model loaded yet.

    Args:
        orchestrator: WanOrchestrator instance
        model_name: The model identifier to check (e.g., "vace_14B", "t2v")

    Returns:
        True if the model is a VACE model, False otherwise
    """
    return orchestrator._test_vace_module(model_name)


def is_flux(orchestrator) -> bool:
    """Check if current model is a Flux model."""
    return orchestrator._get_base_model_type(orchestrator.current_model) == "flux"


def is_t2v(orchestrator) -> bool:
    """Check if current model is a T2V model."""
    base_type = orchestrator._get_base_model_type(orchestrator.current_model)
    return base_type in ["t2v", "t2v_1.3B", "hunyuan", "ltxv_13B", "ltx2_19B", "ltx2_22B"]


def is_qwen(orchestrator) -> bool:
    """Check if current model is a Qwen image model."""
    try:
        if orchestrator._get_model_family(orchestrator.current_model) == "qwen":
            return True
    except (ValueError, KeyError, AttributeError) as e:
        model_logger.debug("Failed to check model family for qwen detection: %s", e)
    base_type = (orchestrator._get_base_model_type(orchestrator.current_model) or "").lower()
    return base_type.startswith("qwen")
