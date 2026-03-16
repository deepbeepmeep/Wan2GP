import json

import safetensors
import torch

from .primitives import StateDict, StateDictLoader
from .sd_ops import SDOps


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with _safe_open(
                shard_path,
                framework="pt",
                device=str(device),
                writable_tensors=False,
            ) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with _safe_open(path, framework="pt", writable_tensors=False) as f:
            return json.loads(f.metadata()["config"])

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)


def _safe_open(path: str, **kwargs):
    try:
        return safetensors.safe_open(path, **kwargs)
    except TypeError:
        if "writable_tensors" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("writable_tensors", None)
            return safetensors.safe_open(path, **kwargs)
        raise
