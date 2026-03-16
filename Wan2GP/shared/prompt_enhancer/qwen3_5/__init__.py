import logging

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

logging.getLogger("fla.utils").setLevel(logging.ERROR)

from . import modeling_qwen3_5 as qwen35_modeling


def register_qwen35_config():
    try:
        AutoConfig.register("qwen3_5", Qwen3_5Config)
    except Exception:
        pass
    try:
        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig)
    except Exception:
        pass
    try:
        CONFIG_MAPPING.register("qwen3_5", Qwen3_5Config)
    except Exception:
        pass
    try:
        CONFIG_MAPPING.register("qwen3_5_text", Qwen3_5TextConfig)
    except Exception:
        pass


def load_qwen35_model_class(modeling_path: str | None = None, class_name: str = "Qwen3_5ForConditionalGeneration"):
    register_qwen35_config()
    try:
        return getattr(qwen35_modeling, class_name)
    except AttributeError as exc:
        raise AttributeError(f"Unsupported Qwen3.5 model class: {class_name}") from exc

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
    "load_qwen35_model_class",
    "register_qwen35_config",
]
