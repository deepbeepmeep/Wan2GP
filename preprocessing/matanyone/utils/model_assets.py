import json
import os
import shutil
import sys
from typing import Any

from huggingface_hub import hf_hub_download
from mmgp import offload
from omegaconf import OmegaConf

from shared.utils import files_locator as fl

from .model_signature import MATANYONE_V1, MATANYONE_V2, detect_matanyone_model_version


def _replace_file(src: str, dst: str) -> None:
    try:
        os.replace(src, dst)
    except OSError as err:
        if err.errno != getattr(os, "EXDEV", 18):
            raise
        shutil.copy2(src, dst)
        os.remove(src)

MATANYONE_SETTINGS_KEY = "matanyone_version"
MATANYONE_DEFAULT_VERSION = MATANYONE_V1
MATANYONE_REPO_ID = "DeepBeepMeep/Wan2.1"
MATANYONE_FOLDER = "mask"
MATANYONE_CONFIG_NAME = "config.json"
MATANYONE_SAM_NAME = "sam_vit_h_4b8939_fp16.safetensors"
MATANYONE_LEGACY_NAME = "model.safetensors"
MATANYONE_WEIGHT_FILES = {
    MATANYONE_V1: "matanyone.safetensors",
    MATANYONE_V2: "matanyone2.safetensors",
}
MATANYONE_VERSION_LABELS = {
    MATANYONE_V1: "MatAnyone v1 (original)",
    MATANYONE_V2: "MatAnyone v2",
}


def _mask_relpath(filename: str) -> str:
    return os.path.join(MATANYONE_FOLDER, filename)


def normalize_matanyone_version(value: Any) -> str:
    if value is None:
        return MATANYONE_DEFAULT_VERSION
    text = str(value).strip().lower()
    if text in {"2", MATANYONE_V2, "matanyone2"}:
        return MATANYONE_V2
    return MATANYONE_V1


def _get_runtime_server_config(server_config=None):
    if isinstance(server_config, dict):
        return server_config
    main_module = sys.modules.get("__main__")
    runtime_config = getattr(main_module, "server_config", None)
    return runtime_config if isinstance(runtime_config, dict) else {}


def _get_runtime_server_config_filename() -> str | None:
    main_module = sys.modules.get("__main__")
    filename = getattr(main_module, "server_config_filename", None)
    return filename if isinstance(filename, str) and len(filename) > 0 else None


def _save_runtime_server_config(server_config) -> bool:
    filename = _get_runtime_server_config_filename()
    if not isinstance(server_config, dict) or filename is None:
        return False
    with open(filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))
    return True


def get_selected_matanyone_version(server_config=None) -> str:
    runtime_config = _get_runtime_server_config(server_config)
    return normalize_matanyone_version(runtime_config.get(MATANYONE_SETTINGS_KEY, MATANYONE_DEFAULT_VERSION))


def get_selected_matanyone_label(server_config=None) -> str:
    return MATANYONE_VERSION_LABELS[get_selected_matanyone_version(server_config)]


def get_matanyone_title_html(server_config=None) -> str:
    return f"<B>Mask Edition is provided by {get_selected_matanyone_label(server_config)}, VRAM optimizations & Extended Masks by DeepBeepMeep</B>"


def get_selected_matanyone_weight_name(server_config=None) -> str:
    return MATANYONE_WEIGHT_FILES[get_selected_matanyone_version(server_config)]


def get_selected_matanyone_weights_path(server_config=None):
    selected_name = get_selected_matanyone_weight_name(server_config)
    selected_path = fl.locate_file(_mask_relpath(selected_name), error_if_none=False)
    if selected_path is not None:
        return selected_path

    legacy_path = fl.locate_file(_mask_relpath(MATANYONE_LEGACY_NAME), error_if_none=False)
    if legacy_path is None:
        return None

    if detect_matanyone_model_version(legacy_path) == get_selected_matanyone_version(server_config):
        return legacy_path
    return None


def _download_mask_asset(filename: str) -> str:
    return hf_hub_download(repo_id=MATANYONE_REPO_ID, filename=filename, local_dir=fl.get_download_location(), subfolder=MATANYONE_FOLDER)


def query_matanyone_download_def(server_config=None):
    runtime_config = _get_runtime_server_config(server_config)
    return {
        "repoId": MATANYONE_REPO_ID,
        "sourceFolderList": [MATANYONE_FOLDER],
        "fileList": [[MATANYONE_SAM_NAME, get_selected_matanyone_weight_name(runtime_config), MATANYONE_CONFIG_NAME]],
    }


def migrate_matanyone_install(server_config=None):
    runtime_config = _get_runtime_server_config(server_config)
    runtime_config.setdefault(MATANYONE_SETTINGS_KEY, MATANYONE_DEFAULT_VERSION)

    legacy_path = fl.locate_file(_mask_relpath(MATANYONE_LEGACY_NAME), error_if_none=False)
    if legacy_path is None:
        return None

    legacy_version = detect_matanyone_model_version(legacy_path)
    if legacy_version is None:
        return None

    target_name = MATANYONE_WEIGHT_FILES[legacy_version]
    target_path = fl.get_download_location(_mask_relpath(target_name))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.normcase(os.path.abspath(legacy_path)) != os.path.normcase(os.path.abspath(target_path)):
        if os.path.isfile(target_path):
            os.remove(legacy_path)
        else:
            _replace_file(legacy_path, target_path)

    config_changed = False
    if legacy_version == MATANYONE_V2 and runtime_config.get(MATANYONE_SETTINGS_KEY) != MATANYONE_V2:
        runtime_config[MATANYONE_SETTINGS_KEY] = MATANYONE_V2
        config_changed = True
    elif MATANYONE_SETTINGS_KEY not in runtime_config:
        runtime_config[MATANYONE_SETTINGS_KEY] = legacy_version
        config_changed = True

    if config_changed:
        _save_runtime_server_config(runtime_config)

    if legacy_version == MATANYONE_V2:
        return "Migrated legacy MatAnyone v2 weights to 'mask/matanyone2.safetensors' and selected MatAnyone v2."
    return "Migrated legacy MatAnyone v1 weights to 'mask/matanyone.safetensors'."


def ensure_selected_matanyone_assets(server_config=None):
    runtime_config = _get_runtime_server_config(server_config)
    migrate_matanyone_install(runtime_config)

    for filename in query_matanyone_download_def(runtime_config)["fileList"][0]:
        if filename == get_selected_matanyone_weight_name(runtime_config):
            continue
        if fl.locate_file(_mask_relpath(filename), error_if_none=False) is None:
            _download_mask_asset(filename)

    weights_path = get_selected_matanyone_weights_path(runtime_config)
    if weights_path is None:
        weights_path = _download_mask_asset(get_selected_matanyone_weight_name(runtime_config))

    config_path = fl.locate_file(_mask_relpath(MATANYONE_CONFIG_NAME))
    sam_path = fl.locate_file(_mask_relpath(MATANYONE_SAM_NAME))
    return config_path, weights_path, sam_path


def load_selected_matanyone_model(server_config=None):
    config_path, weights_path, _ = ensure_selected_matanyone_assets(server_config)

    with open(config_path, "r", encoding="utf-8-sig") as reader:
        config_data = json.load(reader)

    from ..matanyone.model.matanyone import MatAnyone

    model = MatAnyone(OmegaConf.create(config_data["cfg"]), single_object=config_data.get("single_object", True)).eval()
    offload.load_model_data(model, weights_path, writable_tensors=False)
    return model, get_selected_matanyone_version(server_config), weights_path
