"""Compatibility package for legacy chatterbox s3gen.utils imports."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.utils import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.utils"
