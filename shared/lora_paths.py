"""Shared LoRA directory overrides, keyed by the default model subfolder."""

import argparse
import json
from pathlib import Path


def load_lora_config(filename):
    try:
        config_path = Path(filename).expanduser().resolve()
        with config_path.open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
        if not isinstance(config, dict) or any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip() for key, value in config.items()):
            raise ValueError("expected a JSON object mapping LoRA subfolder names to non-empty directory paths")
        return {key: str((config_path.parent / Path(value).expanduser()).resolve()) for key, value in config.items()}
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Cannot load LoRA config '{filename}': {exc}") from exc


def resolve_lora_dir(key, root, config):
    directory = Path(config.get(key, Path(root) / key))
    if directory.is_file():
        raise ValueError(f"LoRA path '{directory}' exists and is not a directory")
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)
