from __future__ import annotations

import json
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
APP_ROOT_DIR = PLUGIN_DIR.parent.parent
APP_SETTINGS_DIR = APP_ROOT_DIR / "settings"
PROCESS_SETTINGS_DIR = PLUGIN_DIR / "settings"
PROCESS_FULL_VIDEO_SETTINGS_FILE = APP_SETTINGS_DIR / "process_full_video_settings.json"
LAUNCH_DEFAULT_PROCESS_NAME = "Outpaint Video - LTX 2.3 Distilled 1.1"
USER_SETTINGS_STORAGE_KEY = "user_settings"
USER_PROCESS_VALUE_PREFIX = "__user_settings__:"


def load_process_definitions() -> tuple[dict[str, dict], str | None]:
    if not PROCESS_SETTINGS_DIR.is_dir():
        return {}, f"Missing process settings folder: {PROCESS_SETTINGS_DIR}"
    process_definitions: dict[str, dict] = {}
    for settings_path in sorted(PROCESS_SETTINGS_DIR.glob("*.json")):
        try:
            raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"Unable to read process setting file {settings_path.name}: {exc}"
        if not isinstance(raw_settings, dict):
            return {}, f"Process setting file {settings_path.name} must contain a JSON object."
        process_name = str(settings_path.stem).strip()
        model_type = str(raw_settings.get("model_type") or "").strip()
        if len(process_name) == 0:
            return {}, f"Process setting file {settings_path.name} has an empty filename stem."
        if len(model_type) == 0:
            return {}, f"Process setting file {settings_path.name} is missing model_type."
        process_definitions[process_name] = {"settings": raw_settings, "path": str(settings_path)}
    if len(process_definitions) == 0:
        return {}, f"No process setting files were found in: {PROCESS_SETTINGS_DIR}"
    return process_definitions, None


PROCESS_DEFINITIONS, PROCESS_DEFINITIONS_ERROR = load_process_definitions()
DEFAULT_PROCESS_NAME = LAUNCH_DEFAULT_PROCESS_NAME if LAUNCH_DEFAULT_PROCESS_NAME in PROCESS_DEFINITIONS else next(iter(PROCESS_DEFINITIONS), "")
DEFAULT_MODEL_TYPE = str(PROCESS_DEFINITIONS.get(DEFAULT_PROCESS_NAME, {}).get("settings", {}).get("model_type") or "")


def load_saved_process_full_video_settings() -> dict:
    if not PROCESS_FULL_VIDEO_SETTINGS_FILE.is_file():
        return {}
    try:
        raw_settings = json.loads(PROCESS_FULL_VIDEO_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Process Full Video] Warning: unable to read saved UI settings from {PROCESS_FULL_VIDEO_SETTINGS_FILE}: {exc}")
        return {}
    return raw_settings if isinstance(raw_settings, dict) else {}


def save_process_full_video_settings(settings: dict) -> None:
    PROCESS_FULL_VIDEO_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_FULL_VIDEO_SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalize_user_settings_ref(value) -> str:
    text = str(value or "").strip().strip('"').replace("\\", "/")
    if len(text) == 0 or text.startswith(("/", "./", "../")):
        return ""
    if len(text) >= 2 and text[1] == ":":
        return ""
    parts = [part.strip() for part in text.split("/") if len(part.strip()) > 0]
    if len(parts) != 2:
        return ""
    base_model_type, filename = parts
    filename = Path(filename).name
    if not filename.lower().endswith(".json"):
        return ""
    return f"{base_model_type}/{filename}"


def is_user_process_value(value) -> bool:
    return str(value or "").startswith(USER_PROCESS_VALUE_PREFIX)


def user_process_value(ref: str) -> str:
    normalized = normalize_user_settings_ref(ref)
    return f"{USER_PROCESS_VALUE_PREFIX}{normalized}" if len(normalized) > 0 else ""


def user_process_ref_from_value(value) -> str:
    text = str(value or "").strip()
    if not text.startswith(USER_PROCESS_VALUE_PREFIX):
        return ""
    return normalize_user_settings_ref(text[len(USER_PROCESS_VALUE_PREFIX):])


def get_saved_user_settings_refs(settings: dict | None) -> list[str]:
    raw_refs = settings.get(USER_SETTINGS_STORAGE_KEY, []) if isinstance(settings, dict) else []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        ref = normalize_user_settings_ref(raw_ref)
        if len(ref) == 0 or ref.casefold() in seen:
            continue
        refs.append(ref)
        seen.add(ref.casefold())
    return refs


def store_user_settings_refs(refs: list[str]) -> None:
    saved_settings = load_saved_process_full_video_settings()
    normalized_refs = get_saved_user_settings_refs({USER_SETTINGS_STORAGE_KEY: refs})
    saved_settings[USER_SETTINGS_STORAGE_KEY] = normalized_refs
    save_process_full_video_settings(saved_settings)


def save_process_full_video_ui_settings(settings: dict) -> None:
    saved_settings = load_saved_process_full_video_settings()
    user_refs = get_saved_user_settings_refs(saved_settings)
    next_settings = dict(settings)
    next_settings[USER_SETTINGS_STORAGE_KEY] = user_refs
    save_process_full_video_settings(next_settings)
