"""Compatibility shim for the legacy chatterbox s3gen.matcha decoder module."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.matcha.decoder import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.matcha.decoder"
