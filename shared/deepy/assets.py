from __future__ import annotations

from typing import Any


WHISPER_MEDIUM_FOLDER = "whisper_medium"
WHISPER_MEDIUM_REPO = "DeepBeepMeep/Wan2.1"
WHISPER_MEDIUM_CONFIG_FILENAME = "config.json"
WHISPER_MEDIUM_WEIGHTS_FILENAME = "model.safetensors"
WHISPER_MEDIUM_REQUIRED_FILES = (WHISPER_MEDIUM_CONFIG_FILENAME, WHISPER_MEDIUM_WEIGHTS_FILENAME)


def query_deepy_download_defs() -> list[dict[str, Any]]:
    return [{
        "repoId": WHISPER_MEDIUM_REPO,
        "sourceFolderList": [WHISPER_MEDIUM_FOLDER],
        "fileList": [list(WHISPER_MEDIUM_REQUIRED_FILES)],
    }]
