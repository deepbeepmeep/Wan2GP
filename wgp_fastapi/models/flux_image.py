"""
Pydantic models for flux-image generation endpoint.

Supports all parameters for Flux 2 Klein 9B model.
"""

from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, Field


# Mapping from API model IDs to WanGP model_type strings
FLUX_IMAGE_MODEL_TYPE_MAP = {
    "flux_2_klein": "flux2_klein_9b",
    "flux_2_klein_base_9b": "flux2_klein_base_9b",
    "flux_2_klein_4b": "flux2_klein_4b",
    "pi_flux2": "pi_flux2",
    "image_edit_plus_2509_nunchaku_fp4": "qwen_image_edit_plus_20B_nunchaku_r128_fp4",
    "image_edit_plus_2511": "qwen_image_edit_plus2_20B",
}

IMAGE_EDIT_MODELS = {"image_edit_plus_2509_nunchaku_fp4", "image_edit_plus_2511"}

class FluxImageModel(str, Enum):
    """Supported models for flux-image generation."""

    FLUX_2_KLEIN = "flux_2_klein"
    FLUX_2_KLEIN_BASE_9B = "flux_2_klein_base_9b"
    FLUX_2_KLEIN_4B = "flux_2_klein_4b"
    PI_FLUX2 = "pi_flux2"
    IMAGE_EDIT_PLUS_2509_NUNCHAKU_FP4 = "image_edit_plus_2509_nunchaku_fp4"
    IMAGE_EDIT_PLUS_2511 = "image_edit_plus_2511"

    @property
    def model_type(self) -> str:
        """Get the WanGP model_type string for this model."""
        return FLUX_IMAGE_MODEL_TYPE_MAP.get(self.value, self.value)

    @property
    def is_image_edit(self) -> bool:
        """Whether this model is an image edit model (vs flux generation)."""
        return self.value in IMAGE_EDIT_MODELS


class FluxImageTaskResponse(BaseModel):
    """Response for async task submission."""

    task_id: str = Field(..., description="Task ID for polling status")


class FluxImageRequest(BaseModel):
    """Request model for flux-image generation with all Flux 2 Klein 9B parameters."""

    model_config = {"protected_namespaces": ()}

    # Core parameters
    prompt: str = Field(..., description="Text prompt for image generation")
    seed: int = Field(
        default=-1, description="Random seed for reproducibility (-1 for random)"
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Number of inference steps (default: 4 for Klein)",
    )
    width: int = Field(
        default=1024, ge=256, le=4096, description="Image width in pixels"
    )
    height: int = Field(
        default=1024, ge=256, le=4096, description="Image height in pixels"
    )
    batch_size: int = Field(
        default=1, ge=1, le=16, description="Number of images to generate"
    )
    model: FluxImageModel = Field(
        default=FluxImageModel.FLUX_2_KLEIN,
        description="Model to use: flux_2_klein (distilled, default), flux_2_klein_base_9b (non-distilled), flux_2_klein_4b, pi_flux2",
    )

    # Image prompt type
    image_prompt_type: str = Field(
        default="I",
        description="Image prompt type: S (start), E (end), V (video guide), L (continue), I (image refs)",
    )

    # LoRA parameters - comma-separated list of LoRA file paths
    activated_loras: Optional[str] = Field(
        default=None,
        description="List of loras",
    )

    # Mask and inpainting reference
    mask_path: Optional[str] = Field(
        default=None,
        description="Path to mask image for inpainting",
    )
    inpaint_reference_path: Optional[str] = Field(
        default=None,
        description="Path to reference image for mask generation",
    )

    def to_wgp_settings(self, image_start_path: str | None = None) -> dict:
        """Convert to WanGP task settings dict."""

        # Default LoRAs for distilled flux2_klein_9b; not for base/non-distilled
        if self.model == FluxImageModel.FLUX_2_KLEIN:
            loras_list = [
                "improved_klein.safetensors",
                "Flux2-Klein-9B-consistency-V2.safetensors",
            ]
        else:
            loras_list = []


        # flux uses "I" for image inputs
        image_prompt_type = "I"

        # qwen needs "KI" to keep background
        if self.model.is_image_edit:
            image_prompt_type = "KI"

        settings = {
            # Core parameters
            "prompt": self.prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "resolution": f"{self.width}x{self.height}",
            "batch_size": self.batch_size,
            "model_type": self.model.model_type,
            "image_mode": 1,  # Image generation mode
            # Image parameters
            "image_prompt_type": image_prompt_type,
            # Output - auto-generate uuid filename
            "output_filename": f"{uuid.uuid4()}.png",
            # LoRA settings - WanGP expects activated_loras
            "activated_loras": loras_list,
        }

        # Set LoRA multipliers: improved_klein at 1.0, consistency V2 at 0.5
        if loras_list:
            settings["loras_multipliers"] = ["1.0", "0.5"]

        # Handle mask inpainting
        if self.mask_path:
            # For inpainting: V=control guide, I=image refs, A=mask
            settings["video_prompt_type"] = "VIA"
            settings["image_mask"] = [self.mask_path]

            # The input image (or inpaint-reference if provided) serves as the control guide
            guide_path = self.inpaint_reference_path or image_start_path
            if guide_path:
                settings["image_guide"] = [guide_path]

            # Build image_refs: input image still passed in as before
            if image_start_path:
                refs = [image_start_path]
                # Add inpaint-reference as an additional reference for inpainting context
                if self.inpaint_reference_path and self.inpaint_reference_path != image_start_path:
                    refs.append(self.inpaint_reference_path)
                settings["image_refs"] = refs

            # set the masking strength based on whether you gave an image;
            # users tend to want less side effects of non-masked areas if you give an image reference
            if self.inpaint_reference_path:
                settings["masking_strength"] = 0.5
            else:
                settings["masking_strength"] = 0.25
        else:
            # Default behaviour (no mask)
            if image_start_path:
                settings["video_prompt_type"] = image_prompt_type
                settings["image_refs"] = [image_start_path]

        return settings


class TaskStatus(BaseModel):
    """Response model for task status queries."""

    model_config = {"protected_namespaces": ()}

    progress: int = Field(..., description="Progress percentage (0-100)")
    preview_image: Optional[str] = Field(
        default=None, description="Base64 encoded preview image (during generation)"
    )
    finished_image: Optional[str] = Field(
        default=None, description="URL to the finished image (on completion)"
    )
    seed: Optional[int] = Field(default=None, description="Seed used for generation")


class FluxImageResponse(BaseModel):
    """Response model for flux-image generation."""

    model_config = {"protected_namespaces": ()}

    status: str = Field(..., description="Generation status: pending, success, failed")
    task_id: Optional[str] = Field(default=None, description="Task ID for tracking")
    images: Optional[list[str]] = Field(
        default=None, description="URLs to generated images (filled on completion)"
    )
    seed_used: Optional[int] = Field(
        default=None, description="Seed that was used for generation"
    )
    model_used: Optional[str] = Field(
        default=None, description="Model that was used for generation"
    )
    steps: Optional[int] = Field(
        default=None, description="Number of inference steps used"
    )
    resolution: Optional[str] = Field(default=None, description="Output resolution")
    batch_size: Optional[int] = Field(
        default=None, description="Number of images generated"
    )
    progress: Optional[int] = Field(
        default=None, description="Progress percentage (0-100)"
    )
    preview_image: Optional[str] = Field(
        default=None, description="Base64 encoded preview image (during generation)"
    )
    finished_image: Optional[str] = Field(
        default=None, description="URL to the finished image (on completion)"
    )
