"""
Runtime patches for the transformers library.

These are applied at import time to work around version-specific issues
between pinned dependencies. Imported from ``source/__init__.py``.
"""

from __future__ import annotations

from source.core.log import model_logger


def _patch_generation_config_from_model_config() -> None:
    """Handle dict-valued ``text_config`` on Qwen2.5-VL + transformers 4.51.3.

    ``GenerationConfig.from_model_config`` calls
    ``model_config.get_text_config(decoder=True)`` and then invokes
    ``.to_dict()`` on the result. For Qwen2.5-VL with transformers 4.51.3
    the ``text_config`` attribute on the parent config is stored as a raw
    ``dict`` instead of a ``PretrainedConfig`` instance, so the call fails
    with ``AttributeError: 'dict' object has no attribute 'to_dict'``.

    The fix temporarily strips dict-valued decoder/generator/text_config
    attrs before delegating to the original method, then restores them.
    """
    try:
        import transformers
    except ImportError:
        return

    generation_config_cls = getattr(transformers, "GenerationConfig", None)
    if generation_config_cls is None:
        return

    original = generation_config_cls.from_model_config
    if getattr(original, "_reigh_patched", False):
        return

    _dict_attr_names = ("decoder", "generator", "text_config")

    def _patched_from_model_config(cls, model_config):
        saved = {}
        for name in _dict_attr_names:
            value = getattr(model_config, name, None)
            if isinstance(value, dict):
                saved[name] = value
                try:
                    delattr(model_config, name)
                except (AttributeError, TypeError):
                    saved.pop(name, None)
        try:
            return original.__func__(cls, model_config)
        finally:
            for name, value in saved.items():
                try:
                    setattr(model_config, name, value)
                except (AttributeError, TypeError):
                    pass

    _patched_from_model_config._reigh_patched = True  # type: ignore[attr-defined]
    generation_config_cls.from_model_config = classmethod(_patched_from_model_config)
    model_logger.debug("[PATCH] transformers.GenerationConfig.from_model_config patched for dict text_config")


def apply_all() -> None:
    _patch_generation_config_from_model_config()


apply_all()
