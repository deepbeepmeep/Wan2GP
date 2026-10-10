import queue
import threading
from typing import Any

from .secure_store import SecureStorageError
from .settings import configured_urls


_send_queue: queue.Queue[tuple[dict[str, Any], str, str] | None] = queue.Queue()
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


def send_notification(config: dict[str, Any], title: str, message: str) -> dict[str, Any]:
    try:
        urls = configured_urls(config)
    except SecureStorageError as error:
        return {"status": "error", "sent": False, "error": str(error)}
    if not urls:
        return {"status": "error", "sent": False, "error": "No Apprise notification destination is configured."}
    try:
        import apprise
    except ImportError:
        return {"status": "error", "sent": False, "error": "Apprise is not installed."}

    try:
        app = apprise.Apprise()
        invalid = sum(not app.add(url) for url in urls)
    except Exception as error:
        return {"status": "error", "sent": False, "error": f"Notification configuration failed ({type(error).__name__})."}
    valid = len(urls) - invalid
    if not valid:
        return {"status": "error", "sent": False, "error": "No valid Apprise notification destination is configured."}
    try:
        sent = bool(app.notify(title=str(title or "WanGP"), body=str(message or "")))
    except Exception as error:
        return {"status": "error", "sent": False, "error": f"Notification delivery failed ({type(error).__name__})."}
    if not sent:
        return {"status": "error", "sent": False, "error": "Apprise reported that notification delivery failed."}
    result = {"status": "sent", "sent": True, "destinations": valid}
    if invalid:
        result["warning"] = f"{invalid} invalid destination{'s were' if invalid != 1 else ' was'} skipped."
    return result


def _notification_worker() -> None:
    while True:
        item = _send_queue.get()
        if item is None:
            _send_queue.task_done()
            return
        try:
            config, title, message = item
            result = send_notification(config, title, message)
            if not result["sent"]:
                print(f"Apprise notification failed: {result['error']}")
        except Exception as error:
            print(f"Apprise notification failed ({type(error).__name__}).")
        finally:
            _send_queue.task_done()


def send_notification_async(config: dict[str, Any], title: str, message: str) -> bool:
    try:
        urls = configured_urls(config)
    except SecureStorageError as error:
        print(f"Apprise notification failed: {error}")
        return False
    if not urls:
        return False
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_notification_worker, name="wangp-notifications", daemon=True)
            _worker.start()
    _send_queue.put((dict(config), str(title), str(message)))
    return True
