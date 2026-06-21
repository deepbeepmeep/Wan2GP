"""
Service for prompt enhancement using the shared prompt enhancer runtime.

Lazy-loads the runtime on first use and caches it for subsequent requests.

NOTE: The Qwen3.5-based prompt enhancer requires CUDA. On Mac (MPS) or CPU-only
systems, enhancement is unavailable and the service will raise a clear error.
"""

from __future__ import annotations

# Force single GPU to avoid multi-GPU device-mismatch issues
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import io
import secrets
import threading
from pathlib import Path
from typing import Optional

import torch

from PIL import Image

from wgp_fastapi.models.prompt_enhance import PromptEnhancerModel

# Force legacy runner (no vLLM CUDA graphs) — the mmgp offload module can
# leave GGUF embedding weights on CPU, causing device-mismatch with vLLM's
# CUDA-graph warmup. The legacy runner still uses the vLLM infrastructure
# but skips graph capture and uses eager mode.
os.environ.setdefault("WGP_QWEN35_PROMPT_ENHANCER_VLLM", "0")

# Lazy-loaded runtime
_enhancer_lock = threading.Lock()
_enhancer_runtime = None  # PromptEnhancerRuntime
_enhancer_model_loaded: Optional[int] = None  # tracks which enhancer_enabled is loaded
_download_in_progress: bool = False


def _resolve_project_root() -> Path:
    """Resolve the project root from the current file location."""
    return Path(__file__).resolve().parents[2]


def _ensure_enhancer_assets(enhancer_enabled: int):
    """Download prompt enhancer model assets if missing."""
    global _download_in_progress

    from shared.prompt_enhancer import download_prompt_enhancer_assets

    _download_in_progress = True
    try:
        download_prompt_enhancer_assets(enhancer_enabled=enhancer_enabled)
    finally:
        _download_in_progress = False


def _fix_qwen_device_placement(runtime):
    """Materialise all GGUF source tensors to CUDA for the Qwen LLM model.

    The mmgp offload module loads GGUF tensors with writable_tensors=False
    and skips materialize_module_source_tensors for the GGUF backend,
    leaving all weights on CPU. The vLLM runner expects everything on CUDA
    and fails with device-mismatch errors on the first forward pass.
    """
    llm = getattr(runtime, "llm_model", None)
    if llm is None:
        return

    from shared.qtypes.gguf import materialize_module_source_tensors

    try:
        materialize_module_source_tensors(llm)
        print("[PROMPT_ENHANCE] Materialised GGUF source tensors to CUDA")
    except Exception as exc:
        print(f"[PROMPT_ENHANCE] Failed to materialise GGUF tensors: {exc}")
        # Fallback: try whole-model .to('cuda') which creates new tensor copies
        try:
            runtime.llm_model = llm.to("cuda")
            print("[PROMPT_ENHANCE] Fallback: moved entire LLM model to CUDA")
        except Exception as exc2:
            print(f"[PROMPT_ENHANCE] Fallback also failed: {exc2}")


def _load_enhancer_runtime(enhancer_enabled: int):
    """Load (or reload) the prompt enhancer runtime for the given model."""
    global _enhancer_runtime, _enhancer_model_loaded

    # Import here to avoid circular imports at module level
    from shared.prompt_enhancer.loader import load_prompt_enhancer_runtime
    from shared.utils.download import process_files_def

    print(f"[PROMPT_ENHANCE] Loading enhancer runtime (enabled={enhancer_enabled})...")

    runtime = load_prompt_enhancer_runtime(
        process_files_def,
        enhancer_enabled=enhancer_enabled,
    )

    # Move all models to CUDA to prevent mmgp offload deadlocks on Windows.
    # The loader sets writable_tensors=False + _offload_hooks=["generate"],
    # which makes mmgp page weights to GPU during the first model.generate()
    # call — this can deadlock on Windows. Pre-moving avoids it entirely.
    if enhancer_enabled in (1, 2):
        for obj_name in ("llm_model", "image_caption_model"):
            obj = getattr(runtime, obj_name, None)
            if obj is not None:
                try:
                    obj = obj.to("cuda")
                    setattr(runtime, obj_name, obj)
                    print(f"[PROMPT_ENHANCE] Moved {obj_name} to CUDA")
                except Exception as exc:
                    print(f"[PROMPT_ENHANCE] Failed to move {obj_name}: {exc}")
    elif enhancer_enabled in (3, 4):
        _fix_qwen_device_placement(runtime)
        # Also materialise vision tower if present (QwenVL caption model)
        vl = getattr(runtime, "image_caption_model", None)
        if vl is not None:
            try:
                from shared.qtypes.gguf import materialize_module_source_tensors
                materialize_module_source_tensors(vl)
            except Exception:
                pass  # vision model may not use GGUF tensors

    _enhancer_runtime = runtime
    _enhancer_model_loaded = enhancer_enabled
    print(f"[PROMPT_ENHANCE] Enhancer runtime loaded successfully")


def get_enhancer(enhancer_enabled: int):
    """Get the prompt enhancer runtime, loading it if necessary."""
    global _enhancer_runtime, _enhancer_model_loaded

    if _enhancer_runtime is None or _enhancer_model_loaded != enhancer_enabled:
        with _enhancer_lock:
            if _enhancer_runtime is None or _enhancer_model_loaded != enhancer_enabled:
                # Ensure assets are downloaded
                _ensure_enhancer_assets(enhancer_enabled)
                _load_enhancer_runtime(enhancer_enabled)

    return _enhancer_runtime


def check_hardware(model: PromptEnhancerModel):
    """Raise RuntimeError if the model's required hardware is not available.

    Qwen3.5 models rely on CUDA-specific vLLM kernels and embedding tensor
    layouts that do not work on MPS or CPU. Florence2 + Llama32/JoyCaption
    use standard PyTorch SDPA and may work on other devices (experimental).
    """
    if model.requires_cuda and not torch.cuda.is_available():
        device_hint = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU-only"
        raise RuntimeError(
            f"{model.display_name} requires CUDA (NVIDIA GPU). "
            f"Detected: {device_hint}. "
            f"Try model='florence2_llama32' or model='florence2_joycaption' "
            f"for non-CUDA hardware (experimental)."
        )


def enhance_prompt(
    prompt: str,
    model: PromptEnhancerModel,
    image_bytes: Optional[bytes] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> str:
    """Enhance a text prompt using the prompt enhancer.

    Args:
        prompt: The text prompt to enhance.
        model: The prompt enhancer model to use.
        image_bytes: Optional raw image bytes for image-guided enhancement.
        max_new_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        seed: Random seed (None for random).

    Returns:
        The enhanced prompt string.

    Raises:
        RuntimeError: If the selected model's required hardware is missing.
    """
    check_hardware(model)

    enhancer_enabled = model.enhancer_enabled
    runtime = get_enhancer(enhancer_enabled)

    # Ensure the project root is in sys.path for imports
    import sys

    project_root = _resolve_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Resolve seed
    if seed is None or seed < 0:
        seed = secrets.randbits(32)

    # Open image if provided
    images = None
    if image_bytes is not None:
        images = [Image.open(io.BytesIO(image_bytes))]

    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt

    enhanced = generate_cinematic_prompt(
        runtime.image_caption_model,
        runtime.image_caption_processor,
        runtime.llm_model,
        runtime.llm_tokenizer,
        prompt,
        images=images,
        video_prompt=True,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    return enhanced[0] if enhanced else prompt
