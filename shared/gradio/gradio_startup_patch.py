"""Runtime patches for Gradio server startup."""

from __future__ import annotations

import time
import warnings

import httpx


def _url_ok_with_retries(url: str) -> bool:
    for attempt in range(5):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                response = httpx.head(url, timeout=3, verify=False)
            if response.status_code in (200, 401, 302, 303, 307):
                return True
        except (ConnectionError, httpx.ConnectError, httpx.TimeoutException):
            pass
        if attempt < 4:
            time.sleep(0.5)
    return False


def install() -> bool:
    import gradio.networking as gradio_networking

    if getattr(gradio_networking.url_ok, "_wangp_startup_patch_installed", False):
        return True
    _url_ok_with_retries._wangp_startup_patch_installed = True
    _url_ok_with_retries._wangp_original_url_ok = gradio_networking.url_ok
    gradio_networking.url_ok = _url_ok_with_retries
    return True


__all__ = ["install"]
