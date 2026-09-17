from __future__ import annotations

from typing import Any

from .config import LLM_CONFIG_KEY, normalize_llm_config
from shared.utils.config_store import config_lock, update_config


def cached_model_catalog(server_config: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    return normalize_llm_config(server_config)["profiles"][engine]["model_catalog"]


def save_model_catalog(server_config: dict[str, Any], server_config_filename: str, engine: str, catalog: list[dict[str, Any]]) -> None:
    with config_lock:
        llm_config = normalize_llm_config(server_config)
        llm_config["profiles"][engine]["model_catalog"] = catalog
        update_config(server_config, server_config_filename, {LLM_CONFIG_KEY: llm_config})
