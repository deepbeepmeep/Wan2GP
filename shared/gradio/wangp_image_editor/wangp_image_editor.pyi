from __future__ import annotations

import hashlib

from gradio.components.image_editor import ImageEditor

from gradio.events import Dependency

class WanGPImageEditor(ImageEditor):
    TEMPLATE_DIR = "templates/"
    FRONTEND_DIR = "frontend/"
    WANGP_FRONTEND_BUILD_ID = "20260701-connection-loss-export-log-34"
    _wangp_magic_mask_patch_enabled = True

    @classmethod
    def get_component_class_id(cls) -> str:
        return hashlib.sha256(f"{cls.__module__}.{cls.__name__}:{cls.WANGP_FRONTEND_BUILD_ID}".encode()).hexdigest()
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component
