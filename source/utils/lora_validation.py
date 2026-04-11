"""LoRA file validation, normalization, and special settings helpers."""

from pathlib import Path

from source.core.log import headless_logger

__all__ = [
    "LORA_FULL_MODEL_TENSOR_THRESHOLD",
    "LORA_EXCESSIVE_TENSOR_THRESHOLD",
    "HTML_SNIFF_BYTES",
    "validate_lora_file",
    "check_loras_in_directory",
]

# --- LoRA validation thresholds ---
# A safetensors file with more tensor keys than this is likely a full model checkpoint, not a LoRA.
LORA_FULL_MODEL_TENSOR_THRESHOLD = 100
# A safetensors file with more tensor keys than this is suspiciously large even for a full model.
LORA_EXCESSIVE_TENSOR_THRESHOLD = 10_000
# Number of bytes to read when sniffing a file header for HTML error pages.
HTML_SNIFF_BYTES = 1024

def validate_lora_file(file_path: Path, filename: str) -> tuple[bool, str]:
    """
    Validates a LoRA file for size and format integrity.

    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    file_size = file_path.stat().st_size

    # Known LoRA size ranges (in bytes)
    # These are based on common LoRA architectures and rank sizes
    LORA_SIZE_RANGES = {
        # Very small LoRAs (rank 4-8)
        'tiny': (1_000_000, 50_000_000),      # 1MB - 50MB
        # Standard LoRAs (rank 16-32)
        'standard': (50_000_000, 500_000_000),  # 50MB - 500MB
        # Large LoRAs (rank 64+) or full model fine-tunes
        'large': (500_000_000, 5_000_000_000), # 500MB - 5GB
        # Extremely large (full model weights)
        'xlarge': (5_000_000_000, 50_000_000_000)  # 5GB - 50GB
    }

    # Check if file size is within any reasonable range
    in_valid_range = any(
        min_size <= file_size <= max_size
        for min_size, max_size in LORA_SIZE_RANGES.values()
    )

    if not in_valid_range:
        if file_size < 1_000_000:  # Less than 1MB
            return False, f"File too small ({file_size:,} bytes) - likely corrupted or incomplete download"
        elif file_size > 50_000_000_000:  # More than 50GB
            return False, f"File too large ({file_size:,} bytes) - likely not a LoRA file"

    # For safetensors files, try to open and inspect
    if filename.endswith('.safetensors'):
        try:
            import safetensors.torch as st
            with st.safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())

                # LoRAs typically have keys like "lora_down.weight", "lora_up.weight", etc.
                lora_indicators = ['lora_down', 'lora_up', 'lora.down', 'lora.up', 'lora_A', 'lora_B']
                has_lora_keys = any(indicator in key for key in keys for indicator in lora_indicators)

                if not has_lora_keys and len(keys) > LORA_FULL_MODEL_TENSOR_THRESHOLD:
                    # Might be a full model checkpoint rather than a LoRA
                    headless_logger.warning(f"{filename} appears to be a full model checkpoint ({len(keys)} tensors) rather than a LoRA")
                elif not has_lora_keys:
                    headless_logger.warning(f"{filename} doesn't appear to contain LoRA weights (no lora_* keys found)")

                # Check for reasonable number of parameters
                if len(keys) == 0:
                    return False, "Safetensors file contains no tensors"
                elif len(keys) > LORA_EXCESSIVE_TENSOR_THRESHOLD:
                    headless_logger.warning(f"{filename} contains many tensors ({len(keys)}) - might be a full model")

        except ImportError:
            headless_logger.debug_anomaly("WARNING", f"safetensors not available for detailed validation of {filename}")
        except (OSError, ValueError, RuntimeError) as e:
            return False, f"Safetensors file appears corrupted: {e}"

    # Additional checks for common corruption patterns
    if file_size == 0:
        return False, "File is empty"

    # For binary files, check they don't start with common error HTML patterns
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(HTML_SNIFF_BYTES)
            if first_bytes.startswith(b'<!DOCTYPE html') or first_bytes.startswith(b'<html'):
                return False, "File appears to be an HTML error page rather than a LoRA file"
    except OSError as e:
        headless_logger.warning(f"Could not read first bytes of {filename} for HTML check: {e}")

    return True, f"File validated successfully ({file_size:,} bytes)"

def check_loras_in_directory(lora_dir: Path | str, fix_issues: bool = False) -> dict:
    """
    Checks all LoRA files in a directory for integrity issues.

    Args:
        lora_dir: Directory containing LoRA files
        fix_issues: If True, removes corrupted files

    Returns:
        Dictionary with validation results
    """
    lora_dir = Path(lora_dir)
    if not lora_dir.exists():
        return {"error": f"Directory does not exist: {lora_dir}"}

    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "issues": [],
        "summary": []
    }

    # Look for LoRA-like files
    lora_extensions = ['.safetensors', '.bin', '.pt', '.pth']
    lora_files = []

    for ext in lora_extensions:
        lora_files.extend(lora_dir.glob(f"*{ext}"))
        lora_files.extend(lora_dir.glob(f"**/*{ext}"))  # Include subdirectories

    # Filter to likely LoRA files
    lora_files = [f for f in lora_files if 'lora' in f.name.lower() or f.suffix == '.safetensors']

    results["total_files"] = len(lora_files)

    for lora_file in lora_files:
        is_valid, validation_msg = validate_lora_file(lora_file, lora_file.name)

        if is_valid:
            results["valid_files"] += 1
            results["summary"].append(f"OK {lora_file.name}: {validation_msg}")
        else:
            results["invalid_files"] += 1
            issue_msg = f"FAIL {lora_file.name}: {validation_msg}"
            results["issues"].append(issue_msg)
            results["summary"].append(issue_msg)

            if fix_issues:
                try:
                    lora_file.unlink()
                    results["summary"].append(f"  -> Removed corrupted file: {lora_file.name}")
                except OSError as e:
                    results["summary"].append(f"  -> Failed to remove {lora_file.name}: {e}")

    return results

def _normalize_activated_loras_list(current_activated) -> list:
    """Helper to ensure activated_loras is a proper list."""
    if not isinstance(current_activated, list):
        try:
            return [str(item).strip() for item in str(current_activated).split(',') if item.strip()]
        except (ValueError, TypeError) as e:
            headless_logger.debug_anomaly("WARNING", f"Failed to normalize activated_loras value '{current_activated}': {e}")
            return []
    return current_activated

def _apply_special_lora_settings(task_id: str, lora_type: str, lora_basename: str, default_steps: int,
                                 guidance_scale: float, flow_shift: float, ui_defaults: dict,
                                 task_params_dict: dict, tea_cache_setting: float = None):
    """
    Shared helper to apply special LoRA settings (CausVid, LightI2X, etc.) to ui_defaults.
    """
    headless_logger.essential(f"Applying {lora_type} LoRA settings.", task_id=task_id)

    # [STEPS DEBUG] Add detailed debug for steps logic
    headless_logger.debug_anomaly("STEPS DEBUG", f"{lora_type}: task_params_dict keys: {list(task_params_dict.keys())}")
    if "steps" in task_params_dict:
        headless_logger.debug_anomaly("STEPS DEBUG", f"{lora_type}: Found 'steps' = {task_params_dict['steps']}")
    if "num_inference_steps" in task_params_dict:
        headless_logger.debug_anomaly("STEPS DEBUG", f"{lora_type}: Found 'num_inference_steps' = {task_params_dict['num_inference_steps']}")
    if "video_length" in task_params_dict:
        headless_logger.debug_anomaly("STEPS DEBUG", f"{lora_type}: Found 'video_length' = {task_params_dict['video_length']}")

    # Handle steps logic
    if "steps" in task_params_dict:
        ui_defaults["num_inference_steps"] = task_params_dict["steps"]
        headless_logger.essential(f"{lora_type} task using specified steps: {ui_defaults['num_inference_steps']}", task_id=task_id)
    elif "num_inference_steps" in task_params_dict:
        ui_defaults["num_inference_steps"] = task_params_dict["num_inference_steps"]
        headless_logger.essential(f"{lora_type} task using specified num_inference_steps: {ui_defaults['num_inference_steps']}", task_id=task_id)
    else:
        ui_defaults["num_inference_steps"] = default_steps
        headless_logger.essential(f"{lora_type} task defaulting to steps: {ui_defaults['num_inference_steps']}", task_id=task_id)

    # Set guidance and flow shift
    ui_defaults["guidance_scale"] = guidance_scale
    ui_defaults["flow_shift"] = flow_shift

    # Set tea cache if specified
    if tea_cache_setting is not None:
        ui_defaults["tea_cache_setting"] = tea_cache_setting

    # Handle LoRA activation
    current_activated = _normalize_activated_loras_list(ui_defaults.get("activated_loras", []))

    if lora_basename not in current_activated:
        current_activated.append(lora_basename)
    ui_defaults["activated_loras"] = current_activated

    # Handle multipliers - simple approach for build_task_state
    current_multipliers_str = ui_defaults.get("loras_multipliers", "")
    multipliers_list = [m.strip() for m in current_multipliers_str.split(" ") if m.strip()] if current_multipliers_str else []
    while len(multipliers_list) < len(current_activated):
        multipliers_list.insert(0, "1.0")
    ui_defaults["loras_multipliers"] = " ".join(multipliers_list)
