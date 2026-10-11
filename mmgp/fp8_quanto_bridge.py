# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations
import json, re, inspect, os
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union, Iterable, Callable

import torch
from safetensors.torch import safe_open, save_file

# ---------- Constants ----------
DATA_SUFFIX  = "._data"
SCALE_SUFFIX = "._scale"          # per-channel, shape [out, 1, ...]
IN_SCALE    = ".input_scale"      # 1-D placeholder tensor [1]
OUT_SCALE   = ".output_scale"     # 1-D placeholder tensor [1]

_QTYPE_NAME = {
    "e4m3fn": "qfloat8_e4m3fn",
    "e5m2":   "qfloat8_e5m2",
    "auto":   "qfloat8",
}

_SCALE_META_KEYS = (
    "fp8_scale_map", "fp8.scale_map", "scale_map",
    "quant_scale_map", "weights_scales", "scales",
)

_DTYPE_ALIASES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
}

_fp8_weight_debug_map: Dict[int, str] = {}


def register_fp8_weight_debug_name(weight: Optional[torch.Tensor], name: str) -> None:
    if weight is None:
        return
    if isinstance(name, str) and name:
        try:
            _fp8_weight_debug_map[id(weight)] = name
        except Exception:
            pass
        try:
            setattr(weight, "_debug_name", name)
        except Exception:
            pass


def get_fp8_weight_debug_name(weight: Optional[torch.Tensor]) -> str:
    if weight is None:
        return "unknown"
    debug = getattr(weight, "_debug_name", None)
    if isinstance(debug, str) and debug:
        return debug
    try:
        mapped = _fp8_weight_debug_map.get(id(weight))
        if isinstance(mapped, str) and mapped:
            try:
                setattr(weight, "_debug_name", mapped)
            except Exception:
                pass
            return mapped
    except Exception:
        pass
    if os.environ.get("WAN2GP_FP8_TAG", "").lower() in ("1", "true", "yes", "on"):
        try:
            print(f"[WAN2GP][FP8 tag][miss] type={type(weight).__name__} id={id(weight)} device={weight.device}")
        except Exception:
            print(f"[WAN2GP][FP8 tag][miss] type={type(weight).__name__} id={id(weight)}")
    return "unknown"


def _is_weight_key(k: str) -> bool:
    return k.endswith(".weight")

# ---------- Accessors (unify file vs dict) ----------
class Accessor:
    def keys(self) -> Iterable[str]: ...
    def get_tensor(self, key: str) -> torch.Tensor: ...
    def metadata(self) -> Dict[str, str]: ...
    def has(self, key: str) -> bool: ...          # NEW
    def can_delete(self) -> bool: return False
    def delete(self, key: str) -> None: raise NotImplementedError

class FileAccessor(Accessor):
    def __init__(self, path: str):
        self._fh = safe_open(path, framework="pt")
        self._keys = list(self._fh.keys())
        self._keys_set = set(self._keys)          # O(1) membership
        self._meta = self._fh.metadata() or {}
    def keys(self) -> Iterable[str]: return self._keys
    def has(self, key: str) -> bool: return key in self._keys_set
    def get_tensor(self, key: str) -> torch.Tensor: return self._fh.get_tensor(key)
    def metadata(self) -> Dict[str, str]: return self._meta
    def close(self) -> None: self._fh.close()

class DictAccessor(Accessor):
    def __init__(self, sd: Dict[str, torch.Tensor], meta: Optional[Dict[str, str]] = None,
                 in_place: bool = False, free_cuda_cache: bool = False, cuda_cache_interval: int = 32):
        self.sd = sd
        self._meta = meta or {}
        self._in_place = in_place
        self._free = free_cuda_cache
        self._interval = int(cuda_cache_interval)
        self._deletions = 0
    def keys(self) -> Iterable[str]: return list(self.sd.keys())
    def has(self, key: str) -> bool: return key in self.sd          # dict membership = O(1)
    def get_tensor(self, key: str) -> torch.Tensor: return self.sd[key]
    def metadata(self) -> Dict[str, str]: return self._meta
    def can_delete(self) -> bool: return self._in_place
    def delete(self, key: str) -> None:
        if key in self.sd:
            self.sd.pop(key, None)
            self._deletions += 1
            if self._free and (self._deletions % self._interval == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()
def _as_accessor(src: Union[str, Dict[str, torch.Tensor]], **dict_opts) -> Tuple[Accessor, Callable[[], None]]:
    if isinstance(src, str):
        acc = FileAccessor(src)
        return acc, acc.close
    acc = DictAccessor(src, **dict_opts)
    return acc, (lambda: None)

# ---------- Shared helpers ----------
def _normalize_scale_dtype(scale_dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(scale_dtype, torch.dtype):
        return scale_dtype
    key = str(scale_dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"scale_dtype must be one of {list(_DTYPE_ALIASES.keys())} or a torch.dtype")
    return _DTYPE_ALIASES[key]

def _json_to_dict(s: str) -> Optional[Dict]:
    # Strictly catch JSON decoding only
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def _maybe_parse_scale_map(meta: Dict[str, str]) -> Optional[Dict[str, float]]:
    def try_parse(obj) -> Optional[Dict[str, float]]:
        if not isinstance(obj, dict):
            return None
        out: Dict[str, float] = {}
        for wk, v in obj.items():
            if isinstance(v, (int, float)):
                out[wk] = float(v)
            elif isinstance(v, dict) and "scale" in v:
                sc = v["scale"]
                if isinstance(sc, (int, float)):
                    out[wk] = float(sc)
                elif isinstance(sc, (list, tuple)) and len(sc) == 1 and isinstance(sc[0], (int, float)):
                    out[wk] = float(sc[0])
        if out:
            return out
        for sub in ("weights", "tensors", "params", "map"):
            subobj = obj.get(sub)
            if isinstance(subobj, dict):
                got = try_parse(subobj)
                if got:
                    return got
        return None

    # exact keys first
    for k in _SCALE_META_KEYS:
        raw = meta.get(k)
        if isinstance(raw, str) and raw.startswith("{") and raw.endswith("}"):
            parsed = _json_to_dict(raw)
            if parsed:
                got = try_parse(parsed)
                if got:
                    return got

    # loose scan of any JSON-looking value
    for v in meta.values():
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            parsed = _json_to_dict(v)
            if parsed:
                got = try_parse(parsed)
                if got:
                    return got
    return None

def _quick_fp8_variant_from_sentinel(acc: Accessor) -> Optional[str]:
    if "scaled_fp8" in set(acc.keys()):
        dt = acc.get_tensor("scaled_fp8").dtype
        if dt == torch.float8_e4m3fn: return "e4m3fn"
        if dt == torch.float8_e5m2:   return "e5m2"
    return None

def _per_channel_reshape(vec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return vec.view(weight.shape[0], *([1] * (weight.ndim - 1)))

# ---------- Unified converter ----------
class ConvertResult(Dict[str, object]):
    @property
    def state_dict(self) -> Dict[str, torch.Tensor]: return self["state_dict"]  # type: ignore
    @property
    def quant_map(self) -> Dict[str, Dict]: return self["quant_map"]            # type: ignore
    @property
    def fp8_format(self) -> str: return self["fp8_format"]                      # type: ignore
    @property
    def patch_needed(self) -> bool: return self["patch_needed"]                 # type: ignore

def _preferred_fp8_scale_dtype() -> torch.dtype:
    """
    Marlin's FP8 kernels expect fp16 scale tensors. Using bf16 led to
    noticeable drift on Linux, so force fp16 for stability.
    """
    return torch.float16


def convert_scaled_fp8_to_quanto(
    src: Union[str, Dict[str, torch.Tensor]],
    *,
    fp8_format: Optional[str] = None,            # 'e4m3fn' | 'e5m2' | None (auto)
    require_scale: bool = False,
    allow_default_scale: bool = True,
    default_missing_scale: float = 1.0,
    dtype: Union[str, torch.dtype] = "float32",
    add_activation_placeholders: bool = True,
    # dict mode options
    sd_metadata: Optional[Dict[str, str]] = None,
    in_place: bool = False,
    free_cuda_cache: bool = False,
    cuda_cache_interval: int = 32,
) -> ConvertResult:
    sd_scale_dtype = _normalize_scale_dtype(dtype)
    patch_needed = False
    if sd_scale_dtype == torch.float32:
        # Quanto's FP8 kernels expect half-precision scales. Coerce float32
        # requests to a supported dtype so we don't need to enable the slow
        # fp32 patch/fallback path.
        sd_scale_dtype = _preferred_fp8_scale_dtype()

    acc, closer = _as_accessor(
        src,
        meta=sd_metadata,
        in_place=in_place,
        free_cuda_cache=free_cuda_cache,
        cuda_cache_interval=cuda_cache_interval,
    )
    if not acc.can_delete(): in_place = False
    try:
        meta = acc.metadata() or {}
        meta_scale_map = _maybe_parse_scale_map(meta) or {}

        keys = list(acc.keys())

        # FP8 variant: sentinel -> first FP8 weight -> 'auto'
        fmt = fp8_format or _quick_fp8_variant_from_sentinel(acc)
        if fmt is None:
            for wk in keys:
                if not _is_weight_key(wk): continue
                dt = acc.get_tensor(wk).dtype
                if dt == torch.float8_e4m3fn: fmt = "e4m3fn"; break
                if dt == torch.float8_e5m2:   fmt = "e5m2";   break
        if fmt is None: fmt = "auto"

        # Map '<base>.scale_weight' -> '<base>.weight'
        scale_weight_map: Dict[str, str] = {}
        for sk in keys:
            if sk.endswith(".scale_weight"):
                base = sk[: -len(".scale_weight")]
                wk = base + ".weight"
                if wk in keys:
                    scale_weight_map[wk] = sk

        def get_scale_vec_for_weight(wk: str, out_ch: int) -> Optional[torch.Tensor]:
            # 1) explicit tensor
            sk = scale_weight_map.get(wk)
            if sk is not None:
                s_t = acc.get_tensor(sk).to(torch.float32)
                if in_place: acc.delete(sk)
                if s_t.numel() == 1:
                    return torch.full((out_ch,), float(s_t.item()), dtype=torch.float32)
                if s_t.numel() == out_ch:
                    return s_t.reshape(out_ch)
                if torch.numel(s_t.unique()) == 1:
                    return torch.full((out_ch,), float(s_t.view(-1)[0].item()), dtype=torch.float32)
                raise ValueError(f"Unexpected scale length for '{wk}': {s_t.numel()} (out_ch={out_ch})")
            # 2) metadata exact / normalized
            if wk in meta_scale_map:
                return torch.full((out_ch,), float(meta_scale_map[wk]), dtype=torch.float32)
            for alt in (wk.replace("model.", ""), re.sub(r"(^|\.)weight$", "", wk)):
                if alt in meta_scale_map:
                    return torch.full((out_ch,), float(meta_scale_map[alt]), dtype=torch.float32)
            return None

        out_sd: Dict[str, torch.Tensor] = {}
        qmap: Dict[str, Dict] = {}

        # Single pass: rewrite FP8 weights, copy-through others
        for k in keys:
            # Drop source-only artifacts
            if k == "scaled_fp8" or k.endswith(".scale_weight") :
                continue

            t = acc.get_tensor(k)
            if in_place: acc.delete(k)
            if _is_weight_key(k) and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # Quantized: keep original FP8 tensor as _data
                out_sd[k + DATA_SUFFIX] = t

                out_ch = int(t.shape[0])
                s_vec = get_scale_vec_for_weight(k, out_ch)
                if s_vec is None:
                    if require_scale and not allow_default_scale:
                        raise KeyError(f"No scale found for '{k}' (looked for '.scale_weight' and metadata).")
                    s_vec = torch.full((out_ch,), float(default_missing_scale), dtype=torch.float32)

                s_grid = _per_channel_reshape(s_vec, t).to(sd_scale_dtype)
                out_sd[k + SCALE_SUFFIX] = s_grid

                if add_activation_placeholders:
                    base = k[:-len(".weight")]
                    out_sd[base + IN_SCALE]  = torch.tensor([1], dtype=sd_scale_dtype)
                    out_sd[base + OUT_SCALE] = torch.tensor([1], dtype=sd_scale_dtype)

                base = k[:-len(".weight")]
                qmap[base] = {"weights": _QTYPE_NAME[fmt], "activations": "none"}
            else:
                out_sd[k] =  t if t.dtype == dtype or t.dtype == torch.float32 else t.to(dtype)
            t = None
        return ConvertResult(state_dict=out_sd, quant_map=qmap, fp8_format=fmt, patch_needed=patch_needed)
    finally:
        closer()


def detect(state_dict, verboseLevel=1):
    info = detect_safetensors_format(state_dict)
    kind = info.get("kind", "none")
    matched = kind in ("scaled_fp8", "fp8")
    return {"matched": matched, "kind": "fp8" if matched else "none", "details": info}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    conv_result = convert_scaled_fp8_to_quanto(state_dict, dtype=default_dtype, in_place=True)
    return {"state_dict": conv_result["state_dict"], "quant_map": conv_result["quant_map"]}


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    return quantization_map, []

def detect_safetensors_format(
    src: Union[str, Dict[str, torch.Tensor]],
    *,
    sd_metadata: Optional[Dict[str, str]] = None,
    probe_weights: bool = False,   # if True, we may read up to 2 weights total
    with_hints: bool = False,
) -> Dict[str, str]:
    """
    Returns:
      {
        'kind': 'quanto' | 'scaled_fp8' | 'fp8' | 'none',
        'quant_format': 'qfloat8_e4m3fn' | 'qfloat8_e5m2' | 'qfloat8' | 'qint8' | 'qint4' | 'unknown' | '',
        'fp8_format': 'e4m3fn' | 'e5m2' | 'unknown' | '',
        'hint': '...'  # only when with_hints=True
      }
    """
    acc, closer = _as_accessor(src, meta=sd_metadata, in_place=False)
    try:
        # --- O(1) sentinel test up-front (no key scan) ---
        if acc.has("scaled_fp8"):
            dt = acc.get_tensor("scaled_fp8").dtype
            fp8_fmt = "e4m3fn" if dt == torch.float8_e4m3fn else ("e5m2" if dt == torch.float8_e5m2 else "unknown")
            out = {"kind": "scaled_fp8", "quant_format": "", "fp8_format": fp8_fmt}
            if with_hints: out["hint"] = "sentinel"
            return out

        # --- Single pass over keys (no re-scans) ---
        ks = list(acc.keys())
        has_scale_weight = False
        saw_quanto_data = False
        fp8_variant = None
        fp8_probe_budget = 2 if probe_weights else 1

        for k in ks:
            # Quanto pack short-circuit
            if not saw_quanto_data and k.endswith(DATA_SUFFIX):
                saw_quanto_data = True
                # we can break here, but keep minimal state setting uniformity
                break

        if saw_quanto_data:
            out = {"kind": "quanto", "quant_format": "qfloat8", "fp8_format": ""}
            if with_hints: out["hint"] = "keys:*._data"
            return out

        # continue single pass for the rest (scale keys + bounded dtype probe)
        for k in ks:
            if not has_scale_weight and k.endswith(".scale_weight"):
                has_scale_weight = True
                # don't return yet; we may still probe a dtype to grab variant

            if fp8_probe_budget > 0 and _is_weight_key(k):
                dt = acc.get_tensor(k).dtype
                if dt == torch.float8_e4m3fn:
                    fp8_variant = "e4m3fn"; fp8_probe_budget -= 1
                elif dt == torch.float8_e5m2:
                    fp8_variant = "e5m2";   fp8_probe_budget -= 1

        if has_scale_weight:
            out = {"kind": "scaled_fp8", "quant_format": "", "fp8_format": fp8_variant or "unknown"}
            if with_hints: out["hint"] = "scale_weight keys"
            return out

        if fp8_variant is not None:
            out = {"kind": "fp8", "quant_format": "", "fp8_format": fp8_variant}
            if with_hints: out["hint"] = "weight dtype (plain fp8)"
            return out

        # --- Cheap metadata peek only if keys didn't decide it (no JSON parsing) ---
        meta = acc.metadata() or {}
        blob = " ".join(v for v in meta.values() if isinstance(v, str)).lower()

        # scaled-fp8 hinted by metadata only
        has_scale_map = (
            any(k in meta for k in _SCALE_META_KEYS) or
            (("scale" in blob) and (("fp8" in blob) or ("float8" in blob)))
        )
        if has_scale_map:
            fmt = "e4m3fn" if "e4m3" in blob else ("e5m2" if "e5m2" in blob else "unknown")
            out = {"kind": "scaled_fp8", "quant_format": "", "fp8_format": fmt}
            if with_hints: out["hint"] = "metadata"
            return out

        # quanto hinted by metadata only (not decisive without keys)
        qtype_hint = ""
        for tok in ("qfloat8_e4m3fn", "qfloat8_e5m2", "qfloat8", "qint8", "qint4"):
            if tok in blob:
                qtype_hint = tok
                break

        out = {"kind": "none", "quant_format": qtype_hint, "fp8_format": ""}
        if with_hints: out["hint"] = "no decisive keys"
        return out

    finally:
        closer()

# ---------- Optional Quanto runtime patch (FP32-scale support), enable/disable ----------
_patch_state = SimpleNamespace(enabled=False, orig=None, scale_index=None)
_fp8_kernel_patch_state = SimpleNamespace(enabled=False, orig_forward=None)
def enable_fp8_fp32_scale_support():
    """
    Version-robust wrapper for WeightQBytesTensor.create:
      - matches both positional/keyword call styles via *args/**kwargs,
      - for FP8 + FP32 scales, expands scalar/uniform scales with a VIEW to the needed length,
      - leaves bf16/fp16 (classic Quanto) untouched.
    Enable only if you emitted float32 scales.
    """
    if _patch_state.enabled:
        return True

    from optimum.quanto.tensor.weights import qbytes as _qbytes  # late import
    orig = _qbytes.WeightQBytesTensor.create
    sig = inspect.signature(orig)
    params = list(sig.parameters.keys())
    scale_index = params.index("scale") if "scale" in params else 5  # fallback

    def wrapper(*args, **kwargs):
        # Extract fields irrespective of signature
        qtype = kwargs.get("qtype", args[0] if len(args) > 0 else None)
        axis  = kwargs.get("axis",  args[1] if len(args) > 1 else None)
        size  = kwargs.get("size",  args[2] if len(args) > 2 else None)

        if "scale" in kwargs:
            scale = kwargs["scale"]
            def set_scale(new): kwargs.__setitem__("scale", new)
        else:
            scale = args[scale_index] if len(args) > scale_index else None
            def set_scale(new):
                nonlocal args
                args = list(args)
                if len(args) > scale_index:
                    args[scale_index] = new
                else:
                    kwargs["scale"] = new
                args = tuple(args)

        is_fp8 = isinstance(qtype, str) and ("float8" in qtype.lower() or "qfloat8" in qtype.lower()) or \
                 (not isinstance(qtype, str) and "float8" in str(qtype).lower())

        if is_fp8 and isinstance(scale, torch.Tensor) and scale.dtype == torch.float32:
            need = int(size[axis]) if (isinstance(size, (tuple, list)) and axis is not None and axis >= 0) else None
            if need is not None:
                if scale.numel() == 1:
                    scale = scale.view(1).expand(need, *scale.shape[1:])
                elif scale.shape[0] != need:
                    # Expand if uniform; otherwise raise
                    uniform = (scale.numel() == 1) or (torch.numel(scale.unique()) == 1)
                    if uniform:
                        scale = scale.reshape(1, *scale.shape[1:]).expand(need, *scale.shape[1:])
                    else:
                        raise ValueError(f"Scale leading dim {scale.shape[0]} != required {need}")
            set_scale(scale)

        return orig(*args, **kwargs)

    _qbytes.WeightQBytesTensor.create = wrapper
    _patch_state.enabled = True
    _patch_state.orig = orig
    _patch_state.scale_index = scale_index
    return True

def disable_fp8_fp32_scale_support():
    """Restore Quanto's original factory."""
    if not _patch_state.enabled:
        return False
    from optimum.quanto.tensor.weights import qbytes as _qbytes
    _qbytes.WeightQBytesTensor.create = _patch_state.orig
    _patch_state.enabled = False
    _patch_state.orig = None
    _patch_state.scale_index = None
    return True


def _quant_map_has_fp8(quant_map) -> bool:
    if not quant_map:
        return False
    for cfg in quant_map.values():
        if not isinstance(cfg, dict):
            continue
        weights = cfg.get("weights")
        if isinstance(weights, str) and "qfloat8" in weights.lower():
            return True
    return False


def enable_fp8_marlin_fallback():
    """
    Replace Quanto's Marlin FP8 linear kernel with a plain matmul fallback.

    When enabled, FP8 weights are dequantized on-the-fly and multiplied with
    standard PyTorch ops, side-stepping quanto's custom CUDA kernels.
    """
    if _fp8_kernel_patch_state.enabled:
        return True
    try:
        from optimum.quanto.tensor.function import QuantizedLinearFunction as _QLF
        from optimum.quanto.tensor.weights.marlin.fp8 import qbits as marlin_fp8
    except Exception:
        return False

    orig_forward = marlin_fp8.MarlinF8QBytesLinearFunction.forward

    def fallback_forward(ctx, input, other, bias=None):
        weight = other.dequantize()
        if weight.dtype != input.dtype:
            weight = weight.to(input.dtype)
        weight = weight.contiguous()
        return _QLF.forward(ctx, input, weight, bias)

    marlin_fp8.MarlinF8QBytesLinearFunction.forward = staticmethod(fallback_forward)
    _fp8_kernel_patch_state.enabled = True
    _fp8_kernel_patch_state.orig_forward = orig_forward
    return True


def disable_fp8_marlin_fallback():
    """Restore Quanto's original Marlin FP8 kernel."""
    if not _fp8_kernel_patch_state.enabled:
        return False
    from optimum.quanto.tensor.weights.marlin.fp8 import qbits as marlin_fp8
    marlin_fp8.MarlinF8QBytesLinearFunction.forward = staticmethod(_fp8_kernel_patch_state.orig_forward)
    _fp8_kernel_patch_state.enabled = False
    _fp8_kernel_patch_state.orig_forward = None
    return True


def maybe_enable_fp8_marlin_fallback(quantization_map=None):
    """
    Enable the FP8 fallback only when the provided quantization map contains FP8 weights.
    """
    if quantization_map is not None and not _quant_map_has_fp8(quantization_map):
        return False
    return enable_fp8_marlin_fallback()


def enable_fp8_marlin_workspace_fix():
    """No-op placeholder (workspace patch removed)."""
    return False


def disable_fp8_marlin_workspace_fix():
    """No-op placeholder (workspace patch removed)."""
    return False




# ---------- Tiny CLI (optional) ----------
def _cli():
    import argparse, json as _json
    p = argparse.ArgumentParser("fp8_quanto_bridge")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert scaled-FP8 (file) to Quanto artifacts.")
    p_conv.add_argument("in_path")
    p_conv.add_argument("out_weights")
    p_conv.add_argument("out_qmap")
    p_conv.add_argument("--fp8-format", choices=("e4m3fn", "e5m2"), default=None)
    p_conv.add_argument("--scale-dtype", default="float32",
                        choices=("float32","bfloat16","float16","fp32","bf16","fp16","half"))
    p_conv.add_argument("--no-activation-placeholders", action="store_true")
    p_conv.add_argument("--default-missing-scale", type=float, default=1.0)

    p_det = sub.add_parser("detect", help="Detect format quickly (path).")
    p_det.add_argument("path")
    p_det.add_argument("--probe", action="store_true")
    p_det.add_argument("--hints", action="store_true")

    p_patch = sub.add_parser("patch", help="Enable/disable FP32-scale runtime patch.")
    p_patch.add_argument("mode", choices=("enable","disable"))

    args = p.parse_args()

    if args.cmd == "convert":
        res = convert_scaled_fp8_to_quanto(
            args.in_path,
            fp8_format=args.fp8_format,
            dtype=args.scale_dtype,
            add_activation_placeholders=not args.no_activation_placeholders,
            default_missing_scale=args.default_missing_scale,
        )
        save_file(res.state_dict, args.out_weights)
        with open(args.out_qmap, "w") as f:
            _json.dump(res.quant_map, f)
        print(f"Wrote: {args.out_weights} and {args.out_qmap}. Patch needed: {res.patch_needed}")
        return 0

    if args.cmd == "detect":
        info = detect_safetensors_format(args.path, probe_weights=args.probe, with_hints=args.hints)
        print(info); return 0

    if args.cmd == "patch":
        ok = enable_fp8_fp32_scale_support() if args.mode == "enable" else disable_fp8_fp32_scale_support()
        print(f"patch {args.mode}: {'ok' if ok else 'already in that state'}")
        return 0

if __name__ == "__main__":
    raise SystemExit(_cli())
