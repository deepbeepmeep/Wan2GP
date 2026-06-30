from __future__ import annotations

import hashlib

from gradio.components.image_editor import ImageEditor


class WanGPImageEditor(ImageEditor):
    TEMPLATE_DIR = "templates/"
    FRONTEND_DIR = "frontend/"
    WANGP_FRONTEND_BUILD_ID = "20260629-export-lock-14"
    _wangp_magic_mask_patch_enabled = True

    @classmethod
    def get_component_class_id(cls) -> str:
        return hashlib.sha256(f"{cls.__module__}.{cls.__name__}:{cls.WANGP_FRONTEND_BUILD_ID}".encode()).hexdigest()
