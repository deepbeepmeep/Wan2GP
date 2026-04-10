"""
Runtime patches for mmgp + transformers config loading.

Applied at import time from ``source/__init__.py``.
"""

from __future__ import annotations

from source.core.log import model_logger


def _fixup_nested_sub_configs(config_obj) -> None:
    """Convert dict-valued sub_configs on a PretrainedConfig into their proper
    config classes.

    ``mmgp.fast_load_transformers_model`` loads the config by dumping the dict
    to a tempfile and calling ``AutoConfig.from_pretrained(tempfile)``. For
    some multimodal models (Qwen2.5-VL + transformers 4.51.3) this leaves
    nested sub-configs (``text_config``, ``vision_config``, ...) as raw dicts
    on the parent config. Downstream code such as
    ``GenerationConfig.from_model_config`` then crashes with
    ``'dict' object has no attribute 'to_dict'``.

    This helper:
      1. Walks ``type(config_obj).sub_configs`` (declared by the config class)
         and instantiates the proper class for any dict value.
      2. For undeclared dict attrs matching the ``decoder``/``generator``/
         ``text_config`` names that transformers' ``get_text_config`` probes,
         removes them so downstream code falls through cleanly.
    """
    sub_configs = getattr(type(config_obj), "sub_configs", None) or {}
    for attr_name, sub_cls in sub_configs.items():
        value = getattr(config_obj, attr_name, None)
        if isinstance(value, dict) and sub_cls is not None:
            try:
                setattr(config_obj, attr_name, sub_cls(**value))
                model_logger.debug(
                    f"[PATCH] Converted {type(config_obj).__name__}.{attr_name} dict → {sub_cls.__name__}"
                )
            except (TypeError, ValueError) as exc:
                model_logger.warning(
                    f"[PATCH] Could not convert {type(config_obj).__name__}.{attr_name} dict to {sub_cls.__name__}: {exc}"
                )

    # Strip undeclared dict attrs that transformers probes in from_model_config.
    for probed_name in ("decoder", "generator", "text_config"):
        if probed_name in sub_configs:
            continue  # already handled above
        value = getattr(config_obj, probed_name, None)
        if isinstance(value, dict):
            try:
                delattr(config_obj, probed_name)
                model_logger.debug(
                    f"[PATCH] Removed undeclared dict attr {type(config_obj).__name__}.{probed_name}"
                )
            except (AttributeError, TypeError):
                pass


def _patch_autoconfig_from_pretrained() -> None:
    """Wrap ``transformers.AutoConfig.from_pretrained`` to post-process the
    returned config and coerce dict sub_configs into proper instances.
    """
    try:
        import transformers
    except ImportError:
        return

    auto_config_cls = getattr(transformers, "AutoConfig", None)
    if auto_config_cls is None:
        return

    original = auto_config_cls.from_pretrained
    if getattr(original, "_reigh_patched", False):
        return

    def _patched_from_pretrained(*args, **kwargs):
        config_obj = original(*args, **kwargs)
        try:
            _fixup_nested_sub_configs(config_obj)
        except (AttributeError, TypeError) as exc:
            model_logger.warning(f"[PATCH] sub_config fixup failed: {exc}")
        return config_obj

    _patched_from_pretrained._reigh_patched = True  # type: ignore[attr-defined]
    auto_config_cls.from_pretrained = _patched_from_pretrained
    model_logger.debug("[PATCH] transformers.AutoConfig.from_pretrained patched for nested sub_configs")


def apply_all() -> None:
    _patch_autoconfig_from_pretrained()


apply_all()
