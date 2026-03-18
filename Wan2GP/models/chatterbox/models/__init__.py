"""Compatibility package for legacy chatterbox model imports."""

from Wan2GP.models.TTS.chatterbox.models import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models"
