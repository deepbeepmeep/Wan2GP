"""Compatibility package for legacy chatterbox import paths."""

from Wan2GP.models.TTS.chatterbox import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox"
