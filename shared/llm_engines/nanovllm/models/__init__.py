import importlib
from typing import Iterable
from torch import nn


_ARCH_REGISTRY = {
    "Qwen3ForCausalLM": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
}

_MODEL_TYPE_REGISTRY = {
    "qwen3": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
}


def _load_model_cls(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Model class '{class_name}' was not found in '{module_name}'.")
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        raise RuntimeError(f"Resolved '{module_name}.{class_name}' is not a torch.nn.Module subclass.")
    return model_cls


def _iter_architectures(hf_config) -> Iterable[str]:
    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        for architecture in architectures:
            if isinstance(architecture, str) and len(architecture) > 0:
                yield architecture


def resolve_model_class(hf_config):
    for architecture in _iter_architectures(hf_config):
        entry = _ARCH_REGISTRY.get(architecture, None)
        if entry is not None:
            return _load_model_cls(entry[0], entry[1])

    model_type = str(getattr(hf_config, "model_type", "")).strip().lower()
    if model_type:
        entry = _MODEL_TYPE_REGISTRY.get(model_type, None)
        if entry is not None:
            return _load_model_cls(entry[0], entry[1])

    supported_architectures = ", ".join(sorted(_ARCH_REGISTRY.keys()))
    supported_model_types = ", ".join(sorted(_MODEL_TYPE_REGISTRY.keys()))
    found_architectures = ", ".join(list(_iter_architectures(hf_config))) or "<none>"
    raise RuntimeError(
        "nanovllm does not support this model architecture. "
        f"Found architectures: {found_architectures}; model_type: '{model_type or '<none>'}'. "
        f"Supported architectures: {supported_architectures}; supported model_type values: {supported_model_types}."
    )
