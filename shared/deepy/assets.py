# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import sys
from typing import Any


WHISPER_MEDIUM_FOLDER = "whisper_medium"
WHISPER_MEDIUM_REPO = "DeepBeepMeep/Wan2.1"
WHISPER_MEDIUM_CONFIG_FILENAME = "config.json"
WHISPER_MEDIUM_WEIGHTS_FILENAME = "model.safetensors"
WHISPER_MEDIUM_REQUIRED_FILES = (WHISPER_MEDIUM_CONFIG_FILENAME, WHISPER_MEDIUM_WEIGHTS_FILENAME)
WHISPER_LARGE_V3_FOLDER = "whisper-large-v3"
WHISPER_LARGE_V3_REPO = "DeepBeepMeep/LongCat"
WHISPER_MEDIUM_DOWNLOAD_DEF = {"repoId": WHISPER_MEDIUM_REPO, "sourceFolderList": [WHISPER_MEDIUM_FOLDER], "fileList": [list(WHISPER_MEDIUM_REQUIRED_FILES)]}
WHISPER_LARGE_V3_DOWNLOAD_DEF = {"repoId": WHISPER_LARGE_V3_REPO, "sourceFolderList": [WHISPER_LARGE_V3_FOLDER], "fileList": [["config.json", "model.safetensors"]]}
WHISPER_DOWNLOAD_DEFS = {"medium": WHISPER_MEDIUM_DOWNLOAD_DEF, "large-v3": WHISPER_LARGE_V3_DOWNLOAD_DEF}
# wgp_config.json (Configuration > Extensions): the Whisper model of every WanGP feature (Deepy, prompt enhancer, speech models)
WHISPER_MODEL_KEY = "whisper_model"
WHISPER_MODEL_CHOICES = [("Auto - Medium up to 8 GB of VRAM, Large-v3 Above", "auto"), ("Medium (1.6 GB of VRAM)", "medium"), ("Large-v3 (3.1 GB of VRAM, More Accurate)", "large-v3")]


def whisper_model_name(server_config: dict | None = None) -> str:
    """The configured Whisper model: auto picks medium on GPUs with up to 8 GB of VRAM (or without CUDA), large-v3 otherwise."""
    if server_config is None:  # the running WanGP's config; none in standalone tools and tests (auto)
        main = next((module for module in (sys.modules.get("__main__"), sys.modules.get("wgp")) if hasattr(module, "server_config")), None)
        server_config = {} if main is None else main.server_config
    value = server_config.get(WHISPER_MODEL_KEY, "auto")
    if value in WHISPER_DOWNLOAD_DEFS:
        return value
    import torch
    return "large-v3" if torch.cuda.is_available() and round(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30) > 8 else "medium"


def whisper_download_def(model_name: str | None = None) -> dict[str, Any]:
    """The files of the configured (or given) Whisper model, for the download declarations of the features using it."""
    return dict(WHISPER_DOWNLOAD_DEFS[model_name or whisper_model_name()])


def query_deepy_download_defs() -> list[dict[str, Any]]:
    # Either model may be configured (WHISPER_MODEL_KEY).
    return [dict(WHISPER_LARGE_V3_DOWNLOAD_DEF), dict(WHISPER_MEDIUM_DOWNLOAD_DEF)]
