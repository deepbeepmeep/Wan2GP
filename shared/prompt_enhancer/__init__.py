from importlib import import_module


_EXPORTS = {
    "load_florence2": ("shared.prompt_enhancer.loader", "load_florence2"),
    "ensure_prompt_enhancer_assets": ("shared.prompt_enhancer.loader", "ensure_prompt_enhancer_assets"),
    "load_prompt_enhancer_runtime": ("shared.prompt_enhancer.loader", "load_prompt_enhancer_runtime"),
    "unload_prompt_enhancer_models": ("shared.prompt_enhancer.loader", "unload_prompt_enhancer_models"),
    "load_qwen35_prompt_enhancer": ("shared.prompt_enhancer.qwen35_text", "load_qwen35_prompt_enhancer"),
    "load_qwen35_text_prompt_enhancer": ("shared.prompt_enhancer.qwen35_text", "load_qwen35_text_prompt_enhancer"),
    "QWEN35_VARIANT_9B": ("shared.prompt_enhancer.qwen35_vl", "QWEN35_VARIANT_9B"),
    "QWEN35_VARIANT_4B": ("shared.prompt_enhancer.qwen35_vl", "QWEN35_VARIANT_4B"),
    "QWEN35_ABLITERATED_TEXT_REQUIRED_FILES": ("shared.prompt_enhancer.qwen35_vl", "QWEN35_ABLITERATED_TEXT_REQUIRED_FILES"),
    "enhancer_quantization_GGUF": ("shared.prompt_enhancer.qwen35_vl", "enhancer_quantization_GGUF"),
    "enhancer_quantization_QUANTO_INT8": ("shared.prompt_enhancer.qwen35_vl", "enhancer_quantization_QUANTO_INT8"),
    "enhancer_quantization_SAFETENSORS": ("shared.prompt_enhancer.qwen35_vl", "enhancer_quantization_SAFETENSORS"),
    "QWEN35_TEXT_GGUF_FILENAME": ("shared.prompt_enhancer.qwen35_vl", "QWEN35_TEXT_GGUF_FILENAME"),
    "QWEN35_VISION_FILENAME": ("shared.prompt_enhancer.qwen35_vl", "QWEN35_VISION_FILENAME"),
    "UPSTREAM_MODELING_FILENAME": ("shared.prompt_enhancer.qwen35_vl", "UPSTREAM_MODELING_FILENAME"),
    "get_qwen35_prompt_enhancer_variant": ("shared.prompt_enhancer.qwen35_vl", "get_qwen35_prompt_enhancer_variant"),
    "get_qwen35_assets_dir_name": ("shared.prompt_enhancer.qwen35_vl", "get_qwen35_assets_dir_name"),
    "get_qwen35_modeling_path": ("shared.prompt_enhancer.qwen35_vl", "get_qwen35_modeling_path"),
    "get_qwen35_variant_spec": ("shared.prompt_enhancer.qwen35_vl", "get_qwen35_variant_spec"),
    "ensure_qwen35_prompt_enhancer_assets": ("shared.prompt_enhancer.qwen35_vl", "ensure_qwen35_prompt_enhancer_assets"),
    "get_qwen35_text_gguf_path": ("shared.prompt_enhancer.qwen35_vl", "get_qwen35_text_gguf_path"),
    "load_qwen35_vl_prompt_enhancer": ("shared.prompt_enhancer.qwen35_vl", "load_qwen35_vl_prompt_enhancer"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
