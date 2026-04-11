"""VLM model download utilities."""

from pathlib import Path

from source.core.log import headless_logger


def download_qwen_vlm_if_needed(model_dir: Path) -> Path:
    """
    Download Qwen2.5-VL-7B-Instruct model if not already present.
    Uses the same download pattern as WanGP's other models.

    Args:
        model_dir: Directory to download the model to (should be ckpts/Qwen2.5-VL-7B-Instruct)

    Returns:
        Path to the downloaded model directory
    """
    # Check if we have the standard HuggingFace format (multiple model files)
    model_files = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "config.json",
        "tokenizer_config.json"
    ]

    has_all_files = all((model_dir / f).exists() for f in model_files)

    if not has_all_files:
        headless_logger.essential(f"Downloading Qwen2.5-VL-7B-Instruct to {model_dir}...")
        headless_logger.essential("This is a one-time download (~16GB). Future runs will use the cached model.")

        try:
            from huggingface_hub import snapshot_download

            # Download the model in standard HuggingFace format
            snapshot_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            headless_logger.essential(f"Download complete: {model_dir}")
        except (OSError, RuntimeError, ValueError) as e:
            headless_logger.error(f"[VLM_DOWNLOAD] Download failed: {e}")
            raise

    return model_dir
