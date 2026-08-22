"""Fast, reusable LoRA module-name mapping without tensor transformations."""

from collections.abc import Iterable, Mapping


COMMON_LORA_ALIASES = (
    ("transformer_blocks", "blocks"),
    ("to_q", "q_proj"),
    ("to_k", "k_proj"),
    ("to_v", "v_proj"),
    ("to_out.0", "out_proj"),
)

_FLAT_PREFIX = "lora_unet_"
_NAMESPACES = ("transformer.", "diffusion_model.")
_TERMINAL_SUFFIXES = (".dora_scale", ".lokr_w1", ".lokr_w2", ".diff_b", ".diff", ".alpha")


def _replace_path(path, source, target):
    parts, source_parts = path.split("."), source.split(".")
    width = len(source_parts)
    variants = []
    for index in range(len(parts) - width + 1):
        if parts[index:index + width] == source_parts:
            variants.append(".".join(parts[:index] + target.split(".") + parts[index + width:]))
    return variants


def _path_variants(path, aliases):
    variants, pending = {path}, [path]
    while pending:
        current = pending.pop()
        for left, right in aliases:
            for source, target in ((left, right), (right, left)):
                for variant in _replace_path(current, source, target):
                    if variant not in variants:
                        variants.add(variant)
                        pending.append(variant)
    return variants


def _add_unique(index, key, target):
    if key not in index:
        index[key] = target
    elif index[key] != target:
        index[key] = None


def _split_adapter_key(key):
    if key.startswith(_FLAT_PREFIX):
        module, separator, suffix = key[len(_FLAT_PREFIX):].partition(".")
        return (module, separator + suffix, True) if separator else (None, None, False)
    position = max(key.rfind(".lora_"), key.rfind(".lora."))
    if position > 0:
        return key[:position], key[position:], False
    for suffix in _TERMINAL_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)], suffix, False
    return None, None, False


class LoraKeyMapper:
    """Resolve common and flattened LoRA names against a target module namespace.

    MMGP remains responsible for adapter suffix normalization, wrapper-prefix
    removal, shape validation, and fused-module tensor splitting.
    """

    __slots__ = ("module_names", "aliases", "flattened")

    def __init__(self, module_names: Iterable[str], aliases: Iterable[tuple[str, str]] = ()):
        self.module_names = frozenset(name for name in module_names if name)
        aliases = COMMON_LORA_ALIASES + tuple(aliases)
        self.aliases = {}
        self.flattened = {}
        for target in self.module_names:
            for variant in _path_variants(target, aliases):
                if variant != target:
                    _add_unique(self.aliases, variant, target)
                _add_unique(self.flattened, variant.replace(".", "_"), target)

    def map_key(self, key: str) -> str:
        module, suffix, flattened = _split_adapter_key(key)
        if module is None:
            return key
        if flattened:
            target = self.flattened.get(module)
            return key if target is None else target + suffix
        if module in self.module_names:
            return key
        namespace = next((prefix for prefix in _NAMESPACES if module.startswith(prefix)), "")
        inner = module[len(namespace):]
        if inner in self.module_names:
            return key
        target = self.aliases.get(inner)
        return key if target is None else namespace + target + suffix

    def map_state_dict(self, state_dict: Mapping[str, object]) -> dict[str, object]:
        mapped = {}
        for key, value in state_dict.items():
            target = self.map_key(key)
            if target in mapped:
                raise ValueError(f"LoRA keys collide after mapping to '{target}'")
            mapped[target] = value
        return mapped

    __call__ = map_state_dict


__all__ = ["COMMON_LORA_ALIASES", "LoraKeyMapper"]
