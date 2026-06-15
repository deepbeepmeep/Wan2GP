"""
Pydantic models for image-to-video generation endpoints.

Currently only supports: LTX 2 Video
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# Mapping from API model IDs to WanGP model_type strings
I2V_MODEL_TYPE_MAP = {
    "hunyuan_1_5_480_i2v_step_distilled": "hunyuan_1_5_480_i2v_step_distilled",
    "wan2_2_i2v_14b_enhanced_lightning_v2": "i2v_2_2_Enhanced_Lightning_v2",
}


class I2VVideoModel(str, Enum):
    """Supported models for image-to-video generation."""
    HUNYUAN_1_5_480_I2V_STEP_DISTILLED = "hunyuan_1_5_480_i2v_step_distilled"
    WAN2_2_I2V_14B_ENHANCED_LIGHTNING_V2 = "wan2_2_i2v_14b_enhanced_lightning_v2"

    @property
    def model_type(self) -> str:
        """Get the WanGP model_type string for this model."""
        return I2V_MODEL_TYPE_MAP.get(self.value, self.value)


class ImageToVideoRequest(BaseModel):
    """Request model for image-to-video generation."""

    prompt: str
    seed: int
    num_inference_steps: int
    width: int
    height: int
    batch_size: int
    model: I2VVideoModel
    video_length: int
    guidance_scale: Optional[float]
    fps: int

    def to_wgp_settings(
        self,
        image_start_path: str | None = None,
        image_end_path: str | None = None,
    ) -> dict:
        """Convert to WanGP task settings dict."""
        settings = {
            "prompt": self.prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "resolution": f"{self.width}x{self.height}",
            "batch_size": self.batch_size,
            "model_type": self.model.model_type,
            "guidance_scale": self.guidance_scale,
            "video_length": self.video_length,
            "image_mode": 0,  # Video generation mode
            "force_fps": self.fps,
            "flow_shift": 5
        }

        # apply some wan2.2 settings; defaults from the UI
        if self.model == I2VVideoModel.WAN2_2_I2V_14B_ENHANCED_LIGHTNING_V2:
            settings["sample_solver"] = "unipc"
            settings["guidance_scale"] = 1
            settings["guidance2_scale"] = 1
            settings["guidance3_scale"] = 5
            settings["guidance_phases"] = 2
            settings["switch_threshold"] = 900
            settings["overlap_size"] = 5

        # Build image prompt type based on which images are provided
        image_prompt_chars = ""
        if image_start_path:
            settings["image_start"] = image_start_path
            image_prompt_chars += "S"
        if image_end_path:
            settings["image_end"] = image_end_path
            image_prompt_chars += "E"
        if image_prompt_chars:
            settings["image_prompt_type"] = image_prompt_chars

        return settings


class ImageToVideoResponse(BaseModel):
    """Response model for image-to-video generation."""

    model_config = {"protected_namespaces": ()}

    status: str = Field(..., description="Generation status")
    task_id: Optional[str] = Field(default=None, description="Task ID for tracking")
    videos: Optional[list[str]] = Field(
        default=None, description="Base64 encoded generated videos"
    )
    seed_used: int = Field(..., description="Seed that was used for generation")
    model_used: str = Field(..., description="Model that was used for generation")
    steps: int = Field(..., description="Number of inference steps used")
    resolution: str = Field(..., description="Output resolution")
    batch_size: int = Field(..., description="Number of videos generated")
    video_length: int = Field(..., description="Number of frames generated")
