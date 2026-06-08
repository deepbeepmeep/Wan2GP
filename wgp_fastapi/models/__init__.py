from wgp_fastapi.models.i2v import (
    ImageToVideoRequest,
    ImageToVideoResponse,
    I2VVideoModel,
)
from wgp_fastapi.models.flux_image import (
    FluxImageRequest,
    FluxImageResponse,
    FluxImageModel,
    TaskStatus,
    FluxImageTaskResponse,
)
from wgp_fastapi.models.magic_mask import (
    MagicMaskResponse,
)

__all__ = [
    # Flux Image
    "FluxImageRequest",
    "FluxImageResponse",
    "FluxImageModel",
    # Task Status
    "TaskStatus",
    # Image to Video
    "I2VVideoModel",
    # Magic Mask
    "MagicMaskResponse",
]
