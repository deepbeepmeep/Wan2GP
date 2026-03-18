"""Compatibility shim for the legacy chatterbox s3gen utils.mask module."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.utils.mask import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.utils.mask"
