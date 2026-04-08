import inspect
import json
import torch
from safetensors.torch import load_file

from .turbo_vaed_module import TurboVAED


def get_turbo_vaed(config_path, ckpt_path, device="cuda", weight_dtype=torch.float32) -> TurboVAED:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config_keys = set(inspect.signature(TurboVAED.__init__).parameters)
    config = {key: value for key, value in config.items() if key in config_keys}
    student = TurboVAED.from_config(config)

    if str(ckpt_path).lower().endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "ema_state_dict" in ckpt, "ckpt must contain ema_state_dict"
        state_dict = ckpt["ema_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    missing, _ = student.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith("decoder.") and not sample_key.startswith("encoder."):
            student.decoder.load_state_dict(state_dict, strict=False)

    student = student.to(device, dtype=weight_dtype)
    student.eval()
    student.requires_grad_(False)
    return student
