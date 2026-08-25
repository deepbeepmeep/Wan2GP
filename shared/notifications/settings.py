from typing import Any
from uuid import uuid4

from . import secure_store


APPRISE_URLS_KEY = "notification_apprise_urls"
NOTIFY_GENERATION_KEY = "notification_on_generation"
NOTIFY_QUEUE_COMPLETE_KEY = "notification_on_queue_complete"
NOTIFY_QUEUE_INTERRUPTED_KEY = "notification_on_queue_interrupted"
SECURE_STORAGE_KEY = "notification_apprise_urls_secure"
CREDENTIAL_ID_KEY = "notification_apprise_credential_id"
CONFIG_KEYS = (APPRISE_URLS_KEY, SECURE_STORAGE_KEY, CREDENTIAL_ID_KEY, NOTIFY_GENERATION_KEY, NOTIFY_QUEUE_COMPLETE_KEY, NOTIFY_QUEUE_INTERRUPTED_KEY)


def default_config() -> dict[str, Any]:
    return {APPRISE_URLS_KEY: [], SECURE_STORAGE_KEY: True, CREDENTIAL_ID_KEY: uuid4().hex, NOTIFY_GENERATION_KEY: False, NOTIFY_QUEUE_COMPLETE_KEY: False, NOTIFY_QUEUE_INTERRUPTED_KEY: False}


def apply_defaults(config: dict[str, Any]) -> None:
    legacy_urls = normalize_apprise_urls(config.get(APPRISE_URLS_KEY, []))
    config.setdefault(SECURE_STORAGE_KEY, not bool(legacy_urls))
    config.setdefault(CREDENTIAL_ID_KEY, uuid4().hex)
    config.setdefault(APPRISE_URLS_KEY, [])
    config.setdefault(NOTIFY_GENERATION_KEY, False)
    config.setdefault(NOTIFY_QUEUE_COMPLETE_KEY, False)
    config.setdefault(NOTIFY_QUEUE_INTERRUPTED_KEY, False)


def normalize_apprise_urls(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
    return [url for item in values for url in str(item).split() if url]


def apprise_urls_text(value: Any) -> str:
    return " ".join(normalize_apprise_urls(value))


def configured_urls(config: dict[str, Any]) -> list[str]:
    if config.get(SECURE_STORAGE_KEY, False):
        credential_id = str(config.get(CREDENTIAL_ID_KEY, "")).strip()
        if not credential_id:
            raise secure_store.SecureStorageError("The notification credential identifier is missing.")
        return secure_store.load_urls(credential_id)
    return normalize_apprise_urls(config.get(APPRISE_URLS_KEY, []))


def prepare_config_update(config: dict[str, Any], urls: Any, secure: Any, on_generation: Any, on_queue_complete: Any, on_queue_interrupted: Any) -> dict[str, Any]:
    normalized_urls = normalize_apprise_urls(urls)
    secure = bool(secure)
    credential_id = str(config.get(CREDENTIAL_ID_KEY, "")).strip() or uuid4().hex
    if secure:
        secure_store.save_urls(credential_id, normalized_urls)
    return {
        APPRISE_URLS_KEY: [] if secure else normalized_urls,
        SECURE_STORAGE_KEY: secure,
        CREDENTIAL_ID_KEY: credential_id,
        NOTIFY_GENERATION_KEY: bool(on_generation),
        NOTIFY_QUEUE_COMPLETE_KEY: bool(on_queue_complete),
        NOTIFY_QUEUE_INTERRUPTED_KEY: bool(on_queue_interrupted),
    }


def cleanup_config_update(old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
    if old_config.get(SECURE_STORAGE_KEY, False) and not new_config.get(SECURE_STORAGE_KEY, False):
        credential_id = str(old_config.get(CREDENTIAL_ID_KEY, "")).strip()
        if credential_id:
            secure_store.delete_urls(credential_id)
