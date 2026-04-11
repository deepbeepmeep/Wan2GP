"""Magic Edit functionality using Replicate API.

This module provides the magic_edit task handler that processes images through
the flux-kontext-apps/in-scene model on Replicate to generate scene variations.
"""

import os
import traceback
import tempfile
import shutil
from pathlib import Path
import requests
from typing import Tuple

_REPLICATE_UNSET = object()
replicate = _REPLICATE_UNSET

from ..utils import (
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    report_orchestrator_failure
)
from source.core.log import task_logger
from source.core.params.task_result import normalize_task_result


def _load_replicate():
    """Import Replicate lazily so optional dependency issues do not break module import."""
    global replicate
    if replicate is None:
        return None
    if replicate is not _REPLICATE_UNSET:
        return replicate
    try:
        import replicate
    except Exception:
        return None
    globals()["replicate"] = replicate
    return replicate

def _handle_magic_edit_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str) -> Tuple[bool, str]:
    """
    Handle a magic_edit task by processing an image through Replicate's black-forest-labs/flux-kontext-dev-lora model.
    
    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base directory for outputs
        task_id: Unique task identifier
        
    Returns:
        Tuple of (success_bool, output_location_or_error_message)
    """
    try:
        # Check if replicate is available
        replicate = _load_replicate()
        if replicate is None:
            msg = "Replicate library not installed. Run: pip install replicate"
            task_logger.error(msg, task_id=task_id)
            report_orchestrator_failure(task_params_from_db, msg)
            return False, msg
            
        # Check for API token
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            msg = "REPLICATE_API_TOKEN environment variable not set"
            task_logger.error(msg, task_id=task_id)
            report_orchestrator_failure(task_params_from_db, msg)
            return False, msg
            
        # Use centralized extraction function for orchestrator_details
        from ..utils import extract_orchestrator_parameters
        extracted_params = extract_orchestrator_parameters(task_params_from_db, task_id)
        
        # Required parameters (now available at top level)
        image_url = extracted_params.get("image_url")
        prompt = extracted_params.get("prompt", "Make a shot in the same scene from a different angle")
        resolution = extracted_params.get("resolution", "768x576")
        seed = extracted_params.get("seed")
        in_scene = extracted_params.get("in_scene", False)  # Default to False
        
        if not image_url:
            msg = "image_url is required in orchestrator_details"
            task_logger.error(msg, task_id=task_id)
            report_orchestrator_failure(task_params_from_db, msg)
            return False, msg
            
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp(prefix=f"magic_edit_{task_id}_"))
        task_logger.debug_block(
            "SETUP",
            {
                "task": task_id,
                "image_url": image_url,
                "prompt": prompt,
                "resolution": resolution,
                "seed": seed,
                "in_scene": in_scene,
                "temp_dir": temp_dir,
            },
            task_id=task_id,
        )
        
        try:
            # Validate that we have a URL (Replicate expects URLs, not local files)
            if not image_url.startswith(("http://", "https://")):
                msg = f"image_url must be a valid HTTP/HTTPS URL, got: {image_url}"
                task_logger.error(msg, task_id=task_id)
                return False, msg
                    
            task_logger.debug(f"Task {task_id}: Using image URL directly with Replicate: {image_url}")
            
            # Prepare Replicate input
            replicate_input = {
                "prompt": prompt,
                "input_image": image_url,
                "aspect_ratio": "match_input_image",  # Let Replicate match the input image aspect ratio
                "lora_strength": 1,
                "output_format": "webp",
                "output_quality": 90
            }
            
            # Only add LoRA weights if in_scene is True
            if in_scene:
                replicate_input["lora_weights"] = "https://huggingface.co/peteromallet/Flux-Kontext-InScene/resolve/main/InScene-v1.0.safetensors"
                task_logger.debug(f"Task {task_id}: Using InScene LoRA for scene-consistent generation")
            else:
                task_logger.debug(f"Task {task_id}: Using base Flux model without InScene LoRA")
            
            # Add seed if specified
            if seed is not None:
                replicate_input["seed"] = int(seed)
                
            task_logger.debug("Running Replicate black-forest-labs/flux-kontext-dev-lora model...", task_id=task_id)
            task_logger.debug(f"Task {task_id}: Replicate input: {replicate_input}")
            
            # Run the model
            output = replicate.run(
                "black-forest-labs/flux-kontext-dev-lora",
                input=replicate_input
            )
            
            task_logger.debug("Replicate processing completed", task_id=task_id)
            task_logger.debug(f"Task {task_id}: Replicate output type: {type(output)}")
            
            # Download the result
            output_image_path = temp_dir / f"{task_id}_edited.webp"
            
            if hasattr(output, 'read'):
                # Output is a file-like object
                with open(output_image_path, 'wb') as f:
                    f.write(output.read())
            elif hasattr(output, 'url'):
                # Output has a URL method
                response = requests.get(output.url(), timeout=60)
                response.raise_for_status()
                with open(output_image_path, 'wb') as f:
                    f.write(response.content)
            elif isinstance(output, str) and output.startswith(("http://", "https://")):
                # Output is a URL string
                response = requests.get(output, timeout=60)
                response.raise_for_status()
                with open(output_image_path, 'wb') as f:
                    f.write(response.content)
            else:
                msg = f"Unexpected output type from Replicate: {type(output)}"
                task_logger.error(msg, task_id=task_id)
                return False, msg
                
            if not output_image_path.exists() or output_image_path.stat().st_size == 0:
                msg = "Failed to download result from Replicate"
                task_logger.error(msg, task_id=task_id)
                return False, msg
                
            task_logger.debug(f"Downloaded result image: {output_image_path.name} ({output_image_path.stat().st_size} bytes)", task_id=task_id)
            
            # Prepare final output path
            final_output_path, initial_db_location = prepare_output_path_with_upload(
                task_id=task_id,
                filename=output_image_path.name,
                main_output_dir_base=main_output_dir_base,
                task_type="magic_edit")
            
            # Move to final location
            shutil.move(str(output_image_path), str(final_output_path))
            task_logger.debug(f"Task {task_id}: Moved result to final location: {final_output_path}")
            
            # Handle upload and get final DB location
            final_db_location = upload_and_get_final_output_location(
                final_output_path,
                initial_db_location)
            
            return True, final_db_location
            
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
                task_logger.debug(f"Task {task_id}: Cleaned up temp directory: {temp_dir}")
            except OSError as e_cleanup:
                task_logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e_cleanup}", task_id=task_id)
                # Cleanup failure is non-fatal; the task result is already determined
                
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        task_logger.error(f"Magic edit task failed: {e}", task_id=task_id)
        task_logger.debug(f"Magic edit traceback: {traceback.format_exc()}", task_id=task_id)
        msg = f"Magic edit exception: {e}"
        report_orchestrator_failure(task_params_from_db, msg)
        return False, msg


def handle_magic_edit_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
):
    """Typed public entrypoint that normalizes the legacy tuple result."""
    return normalize_task_result(
        _handle_magic_edit_task(
            task_params_from_db=task_params_from_db,
            main_output_dir_base=main_output_dir_base,
            task_id=task_id,
        )
    )
