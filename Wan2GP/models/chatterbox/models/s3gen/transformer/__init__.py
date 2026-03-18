"""Compatibility package for legacy chatterbox s3gen.transformer imports."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.transformer import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.transformer"
