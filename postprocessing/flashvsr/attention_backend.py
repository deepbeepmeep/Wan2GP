from __future__ import annotations

import importlib
import importlib.util
import traceback
from typing import Callable

import torch


_SPARSE_ATTENTION: Callable | None = None
_BACKEND_NAME: str | None = None
_BACKEND_ERROR: str | None = None
_PRINTED_BACKEND = False
_PRINTED_IMPORT_ERRORS: set[str] = set()
_REQUIREMENTS_MESSAGE = "FlashVSR requires both SpargeAttn kernels and Triton."
_INSTALL_MESSAGE = "Install them from docs/INSTALLATION.md and restart WanGP."
_DEPENDENCIES = (("triton", "Triton"), ("spas_sage_attn", "SpargeAttn"))
_ARCH_KERNELS = {
    "sm80": ("SM80_ENABLED", "spas_sage_attn.sm80_compile", "spas_sage_attn._qattn_sm80"),
    "sm86": ("SM80_ENABLED", "spas_sage_attn.sm80_compile", "spas_sage_attn._qattn_sm80"),
    "sm87": ("SM80_ENABLED", "spas_sage_attn.sm80_compile", "spas_sage_attn._qattn_sm80"),
    "sm89": ("SM89_ENABLED", "spas_sage_attn.sm89_compile", "spas_sage_attn._qattn_sm89"),
    "sm90": ("SM90_ENABLED", "spas_sage_attn.sm90_compile", "spas_sage_attn._qattn_sm90"),
    "sm100": ("SM89_ENABLED", "spas_sage_attn.sm89_compile", "spas_sage_attn._qattn_sm89"),
    "sm120": ("SM89_ENABLED", "spas_sage_attn.sm89_compile", "spas_sage_attn._qattn_sm89"),
    "sm121": ("SM89_ENABLED", "spas_sage_attn.sm89_compile", "spas_sage_attn._qattn_sm89"),
}


def _print_import_error(module_name: str, exc: BaseException) -> None:
    key = f"{module_name}:{type(exc).__name__}:{exc}"
    if key in _PRINTED_IMPORT_ERRORS:
        return
    _PRINTED_IMPORT_ERRORS.add(key)
    print(f"[FlashVSR] Importing {module_name} failed:")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _missing_sparse_attention_dependencies() -> list[str]:
    missing = []
    for module_name, display_name in _DEPENDENCIES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(display_name)
    return missing


def _missing_dependencies_message(missing: list[str]) -> str:
    return f"{_REQUIREMENTS_MESSAGE} Missing: {', '.join(missing)}. {_INSTALL_MESSAGE}"


def _dependency_import_message(display_name: str, module_name: str, exc: BaseException) -> str:
    return f"{_REQUIREMENTS_MESSAGE} {display_name} is installed, but importing {module_name} failed. Check the console for the import error, then reinstall from docs/INSTALLATION.md and restart WanGP. Import failed: {type(exc).__name__}: {exc}"


def _kernel_load_message(sparge_error: str | None) -> str:
    return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but its kernels could not be loaded. Reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. SpargeAttn import failed: {sparge_error or 'not installed'}"


def _arch_kernel_load_message(arch: str, module_name: str, exc: BaseException | None) -> str:
    if exc is not None:
        return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but importing its {arch} kernel failed. Check the console for the import error, then reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. Import failed: {type(exc).__name__}: {exc}"
    return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but its {arch} kernel is unavailable. Reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. Missing kernel module: {module_name}"


def _dependency_import_error() -> str | None:
    for module_name, display_name in _DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            _print_import_error(module_name, exc)
            return _dependency_import_message(display_name, module_name, exc)
    return None


def _import_sparge_core():
    try:
        return importlib.import_module("shared.spas_sage_attn_core"), None
    except Exception as exc:
        _print_import_error("shared.spas_sage_attn_core", exc)
        return None, f"{type(exc).__name__}: {exc}"


def _current_cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm{major}{minor}"


def _arch_kernel_error(module, arch: str | None) -> str | None:
    if arch is None or arch not in _ARCH_KERNELS:
        return None
    flag_name, compile_module_name, direct_module_name = _ARCH_KERNELS[arch]
    if getattr(module, flag_name, False):
        return None
    try:
        importlib.import_module(compile_module_name)
    except ModuleNotFoundError as exc:
        if exc.name != compile_module_name:
            _print_import_error(compile_module_name, exc)
            return _arch_kernel_load_message(arch, compile_module_name, exc)
        try:
            importlib.import_module(direct_module_name)
        except Exception as direct_exc:
            _print_import_error(direct_module_name, direct_exc)
            return _arch_kernel_load_message(arch, direct_module_name, direct_exc)
        return _arch_kernel_load_message(arch, direct_module_name, None)
    except Exception as exc:
        _print_import_error(compile_module_name, exc)
        return _arch_kernel_load_message(arch, compile_module_name, exc)
    return _arch_kernel_load_message(arch, compile_module_name, None)


def _sparse_attention_requirement_status() -> tuple[Callable | None, str | None, str | None]:
    missing = _missing_sparse_attention_dependencies()
    if missing:
        return None, None, _missing_dependencies_message(missing)
    dependency_import_error = _dependency_import_error()
    if dependency_import_error is not None:
        return None, None, dependency_import_error
    module, sparge_error = _import_sparge_core()
    if module is None:
        return None, None, _kernel_load_message(sparge_error)
    arch_kernel_error = _arch_kernel_error(module, _current_cuda_arch())
    if arch_kernel_error is not None:
        return None, None, arch_kernel_error
    fn = getattr(module, "block_sparse_sage2_attn_cuda", None)
    if not callable(fn):
        return None, None, _kernel_load_message("WanGP SpargeAttn block_sparse_sage2_attn_cuda not found")
    return fn, "WanGP SpargeAttn block_sparse_sage2_attn_cuda", None


def sparse_attention_requirement_message() -> str | None:
    _, _, message = _sparse_attention_requirement_status()
    return message


def sparge_attention_available() -> bool:
    return sparse_attention_requirement_message() is None


def require_sparge_attention() -> None:
    _, _, message = _sparse_attention_requirement_status()
    if message is not None:
        raise RuntimeError(message)


def _mask_topk(mask_id: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | float:
    if isinstance(mask_id, list):
        mask_id = mask_id[0] if len(mask_id) > 0 else None
    if mask_id is None or not torch.is_tensor(mask_id):
        return 0.5
    density = mask_id.to(device=q.device, dtype=torch.float32).mean(dim=(0, 2, 3))
    return density.clamp(1.0 / max(int(mask_id.shape[-1]), 1), 1.0)


def _int8_mask(mask_id: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    mask = mask_id[0] if isinstance(mask_id, list) else mask_id
    if mask.dtype != torch.int8:
        mask = mask.to(torch.int8)
        if isinstance(mask_id, list):
            mask_id[0] = mask
    return mask_id if isinstance(mask_id, list) else mask


def _take_mask(mask_id: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(mask_id, list):
        mask = mask_id[0]
        mask_id.clear()
        return mask
    return mask_id


def _load_backend() -> tuple[Callable, str]:
    global _BACKEND_ERROR
    sparge_fn, sparge_name_or_error, message = _sparse_attention_requirement_status()
    if message is not None:
        _BACKEND_ERROR = message
        raise RuntimeError(message)
    if sparge_fn is None:
        _BACKEND_ERROR = _kernel_load_message(sparge_name_or_error)
        raise RuntimeError(_BACKEND_ERROR)

    use_qkv_list = sparge_fn.__module__ == "shared.spas_sage_attn_core"

    def sparge_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
        if "mask_id" in sparge_fn.__code__.co_varnames:
            mask_id = _int8_mask(mask_id)
            if use_qkv_list:
                return sparge_fn(qkv_list, mask_id=mask_id, tensor_layout="HND", output_dtype=qkv_list[0].dtype, recycle_q=recycle_q)
            q, k, v = qkv_list
            qkv_list.clear()
            return sparge_fn(q, k, v, mask_id=_take_mask(mask_id), tensor_layout="HND", output_dtype=q.dtype)
        if "topk" in sparge_fn.__code__.co_varnames:
            if use_qkv_list:
                topk = _mask_topk(mask_id, qkv_list[0])
                if isinstance(mask_id, list):
                    mask_id.clear()
                return sparge_fn(qkv_list, is_causal=False, tensor_layout="HND", output_dtype=qkv_list[0].dtype, topk=topk, recycle_q=recycle_q)
            q, k, v = qkv_list
            qkv_list.clear()
            topk = _mask_topk(mask_id, q)
            if isinstance(mask_id, list):
                mask_id.clear()
            return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype, topk=topk)
        q, k, v = qkv_list
        qkv_list.clear()
        return sparge_fn(q, k, v, is_causal=False, tensor_layout="HND", output_dtype=q.dtype)

    return sparge_attention, sparge_name_or_error or "SpargeAttn"


def get_sparse_backend_name() -> str:
    global _SPARSE_ATTENTION, _BACKEND_NAME
    if _SPARSE_ATTENTION is None:
        _SPARSE_ATTENTION, _BACKEND_NAME = _load_backend()
    return _BACKEND_NAME or "unknown"


def sparse_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
    global _PRINTED_BACKEND
    backend_name = get_sparse_backend_name()
    if not _PRINTED_BACKEND:
        print(f"[FlashVSR] Sparse attention backend: {backend_name}")
        _PRINTED_BACKEND = True
    return _SPARSE_ATTENTION(qkv_list, mask_id, recycle_q=recycle_q)
