"""Compatibility shim for the legacy chatterbox transformer encoder_layer module."""

from Wan2GP.models.TTS.chatterbox.models.s3gen.transformer.encoder_layer import *  # noqa: F401,F403


def _compat_target() -> str:
    return "Wan2GP.models.TTS.chatterbox.models.s3gen.transformer.encoder_layer"
