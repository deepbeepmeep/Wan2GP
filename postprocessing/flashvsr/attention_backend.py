from __future__ import annotations

import importlib
import importlib.util
import traceback
from typing import Callable

import torch

from .sparse_backend_config import (
    SPARSE_BACKEND_AUTO as _AUTO_BACKEND,
    SPARSE_BACKEND_LABELS as _BACKEND_LABELS,
    SPARSE_BACKEND_SPARGE as _SPARGE_BACKEND,
    SPARSE_BACKEND_TRITON_SPARSE as _TRITON_SPARSE_BACKEND,
    normalize_sparse_backend,
)

_SPARSE_ATTENTION: Callable | None = None
_BACKEND_NAME: str | None = None
_BACKEND_ERROR: str | None = None
_PRINTED_BACKEND = False
_PRINTED_IMPORT_ERRORS: set[str] = set()
_PRINTED_AUTO_FALLBACKS: set[str] = set()
_SPARSE_BACKEND = _AUTO_BACKEND
_REQUIREMENTS_MESSAGE = "FlashVSR sparse attention requirements are not satisfied."
_INSTALL_MESSAGE = "Install them from docs/INSTALLATION.md and restart WanGP."
_BACKEND_DEPENDENCIES = {
    _SPARGE_BACKEND: (("triton", "Triton"), ("spas_sage_attn", "SpargeAttn")),
    _TRITON_SPARSE_BACKEND: (("triton", "Triton"),),
}
_BUNDLED_SPARSE_BACKEND_NAME = "bundled Triton Sparse Attention"
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


def set_sparse_backend(backend: object) -> str:
    global _SPARSE_BACKEND, _SPARSE_ATTENTION, _BACKEND_NAME, _BACKEND_ERROR, _PRINTED_BACKEND
    backend = normalize_sparse_backend(backend)
    if backend != _SPARSE_BACKEND:
        _SPARSE_ATTENTION = None
        _BACKEND_NAME = None
        _BACKEND_ERROR = None
        _PRINTED_BACKEND = False
    _SPARSE_BACKEND = backend
    return _SPARSE_BACKEND


def _selected_sparse_backend(backend: object | None = None) -> str:
    return _SPARSE_BACKEND if backend is None else normalize_sparse_backend(backend)


def _print_auto_fallback(message: str) -> None:
    if message in _PRINTED_AUTO_FALLBACKS:
        return
    _PRINTED_AUTO_FALLBACKS.add(message)
    print(f"[FlashVSR] Auto backend cannot use {_BACKEND_LABELS[_SPARGE_BACKEND]}: {message} Install SpargeAttn for better FlashVSR quality.")
    print(f"[FlashVSR] Auto backend trying {_BACKEND_LABELS[_TRITON_SPARSE_BACKEND]}.")


def _missing_sparse_attention_dependencies(backend: str) -> list[str]:
    missing = []
    for module_name, display_name in _BACKEND_DEPENDENCIES[backend]:
        if importlib.util.find_spec(module_name) is None:
            missing.append(display_name)
    return missing


def _missing_dependencies_message(backend: str, missing: list[str]) -> str:
    return f"{_REQUIREMENTS_MESSAGE} Backend: {_BACKEND_LABELS[backend]}. Missing: {', '.join(missing)}. {_INSTALL_MESSAGE}"


def _dependency_import_message(display_name: str, module_name: str, exc: BaseException) -> str:
    return f"{_REQUIREMENTS_MESSAGE} {display_name} is installed, but importing {module_name} failed. Check the console for the import error, then reinstall from docs/INSTALLATION.md and restart WanGP. Import failed: {type(exc).__name__}: {exc}"


def _kernel_load_message(sparge_error: str | None) -> str:
    return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but its kernels could not be loaded. Reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. SpargeAttn import failed: {sparge_error or 'not installed'}"


def _arch_kernel_load_message(arch: str, module_name: str, exc: BaseException | None) -> str:
    if exc is not None:
        return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but importing its {arch} kernel failed. Check the console for the import error, then reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. Import failed: {type(exc).__name__}: {exc}"
    return f"{_REQUIREMENTS_MESSAGE} SpargeAttn is installed, but its {arch} kernel is unavailable. Reinstall SpargeAttn from docs/INSTALLATION.md and restart WanGP. Missing kernel module: {module_name}"


def _dependency_import_error(backend: str) -> str | None:
    for module_name, display_name in _BACKEND_DEPENDENCIES[backend]:
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


def _load_triton_sparse_backend() -> tuple[Callable | None, str | None, str | None]:
    try:
        from .sparse_sage.core import sparse_sageattn
    except Exception as exc:
        _print_import_error("postprocessing.flashvsr.sparse_sage.core", exc)
        return None, None, _dependency_import_message(_BUNDLED_SPARSE_BACKEND_NAME, "postprocessing.flashvsr.sparse_sage.core", exc)

    def bundled_sparse_sage(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
        mask_id = _int8_mask(mask_id)
        return sparse_sageattn(qkv_list, mask_id=_take_mask(mask_id), is_causal=False, tensor_layout="HND")

    return bundled_sparse_sage, _BUNDLED_SPARSE_BACKEND_NAME, None


def _backend_requirement_status(backend: str) -> tuple[Callable | None, str | None, str | None]:
    missing = _missing_sparse_attention_dependencies(backend)
    if missing:
        return None, None, _missing_dependencies_message(backend, missing)
    dependency_import_error = _dependency_import_error(backend)
    if dependency_import_error is not None:
        return None, None, dependency_import_error
    if backend == _TRITON_SPARSE_BACKEND:
        return _load_triton_sparse_backend()
    module, sparge_error = _import_sparge_core()
    if module is None:
        return None, None, _kernel_load_message(sparge_error)
    arch_kernel_error = _arch_kernel_error(module, _current_cuda_arch())
    if arch_kernel_error is not None:
        return None, None, arch_kernel_error
    fn = getattr(module, "block_sparse_attn_cuda", None)
    if not callable(fn):
        return None, None, _kernel_load_message("WanGP SpargeAttn block sparse CUDA function not found")
    return fn, "WanGP SpargeAttn block sparse CUDA", None


def _sparse_attention_requirement_status(backend: object | None = None) -> tuple[Callable | None, str | None, str | None]:
    backend = _selected_sparse_backend(backend)
    if backend != _AUTO_BACKEND:
        return _backend_requirement_status(backend)

    sparge_fn, sparge_name, sparge_message = _backend_requirement_status(_SPARGE_BACKEND)
    if sparge_message is None:
        return sparge_fn, sparge_name, None

    _print_auto_fallback(sparge_message)
    triton_sparse_fn, triton_sparse_name, triton_sparse_message = _backend_requirement_status(_TRITON_SPARSE_BACKEND)
    if triton_sparse_message is None:
        return triton_sparse_fn, triton_sparse_name, None
    return None, None, f"FlashVSR Auto backend could not load any sparse attention backend. Sparge: {sparge_message} {_BACKEND_LABELS[_TRITON_SPARSE_BACKEND]}: {triton_sparse_message}"


def sparse_attention_requirement_message(backend: object | None = None) -> str | None:
    _, _, message = _sparse_attention_requirement_status(backend)
    return message


def sparge_attention_available() -> bool:
    return sparse_attention_requirement_message(_SPARGE_BACKEND) is None


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
    backend = _selected_sparse_backend()
    sparge_fn, sparge_name_or_error, message = _sparse_attention_requirement_status()
    if message is not None:
        _BACKEND_ERROR = message
        raise RuntimeError(message)
    if sparge_fn is None:
        _BACKEND_ERROR = _kernel_load_message(sparge_name_or_error)
        raise RuntimeError(_BACKEND_ERROR)
    if backend == _TRITON_SPARSE_BACKEND or sparge_name_or_error == _BUNDLED_SPARSE_BACKEND_NAME:
        return sparge_fn, sparge_name_or_error or _BUNDLED_SPARSE_BACKEND_NAME

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


def log_sparse_backend() -> None:
    global _PRINTED_BACKEND
    backend_name = get_sparse_backend_name()
    if not _PRINTED_BACKEND:
        print(f"[FlashVSR] Sparse attention backend: {backend_name}")
        _PRINTED_BACKEND = True


def sparse_attention(qkv_list: list[torch.Tensor], mask_id: torch.Tensor | list[torch.Tensor], recycle_q: bool = False) -> torch.Tensor:
    log_sparse_backend()
    return _SPARSE_ATTENTION(qkv_list, mask_id, recycle_q=recycle_q)
