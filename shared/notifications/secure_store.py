import json
from typing import Any


SERVICE_NAME = "WanGP Notifications"


class SecureStorageError(RuntimeError):
    pass


def _keyring():
    try:
        import keyring
        backend = keyring.get_keyring()
        if backend.priority <= 0:
            raise RuntimeError("No usable credential-store backend")
        return keyring
    except Exception as error:
        raise SecureStorageError(f"OS credential storage is unavailable ({type(error).__name__}).") from error


def load_urls(credential_id: str) -> list[str]:
    try:
        value = _keyring().get_password(SERVICE_NAME, credential_id)
        if value is None:
            return []
        urls = json.loads(value)
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError("Invalid stored notification destinations")
        return urls
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not read notification destinations from OS credential storage ({type(error).__name__}).") from error


def save_urls(credential_id: str, urls: list[str]) -> None:
    try:
        _keyring().set_password(SERVICE_NAME, credential_id, json.dumps(urls))
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not save notification destinations in OS credential storage ({type(error).__name__}).") from error


def delete_urls(credential_id: str) -> None:
    try:
        keyring = _keyring()
        if keyring.get_password(SERVICE_NAME, credential_id) is not None:
            keyring.delete_password(SERVICE_NAME, credential_id)
    except SecureStorageError:
        raise
    except Exception as error:
        raise SecureStorageError(f"Could not remove notification destinations from OS credential storage ({type(error).__name__}).") from error
