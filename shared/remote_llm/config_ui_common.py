from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .config import LLM_CONFIG_KEY, normalize_llm_config


def cached_model_catalog(server_config: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    return normalize_llm_config(server_config)["profiles"][engine]["model_catalog"]


def save_model_catalog(server_config: dict[str, Any], server_config_filename: str, engine: str, catalog: list[dict[str, Any]]) -> None:
    updated_config = deepcopy(server_config)
    llm_config = normalize_llm_config(updated_config)
    llm_config["profiles"][engine]["model_catalog"] = catalog
    updated_config[LLM_CONFIG_KEY] = llm_config
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(updated_config, indent=4))
    server_config[LLM_CONFIG_KEY] = llm_config
