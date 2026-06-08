"""
Pydantic models for magic-mask endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field


class MagicMaskResponse(BaseModel):
    """Response model for magic-mask generation."""

    model_config = {"protected_namespaces": ()}

    image_url: str = Field(..., description="URL to the generated mask image")
    keywords: list[str] = Field(
        ..., description="Keywords used for mask generation"
    )
