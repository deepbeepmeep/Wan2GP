"""Compatibility shim for the legacy chatterbox transformer upsample_encoder module."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.transformer.upsample_encoder import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.transformer.upsample_encoder"
