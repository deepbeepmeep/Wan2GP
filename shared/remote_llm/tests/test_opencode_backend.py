from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from shared.remote_llm.opencode_backend import OpenCodeBackend


class OpenCodeBackendTests(unittest.TestCase):
    def test_sse_is_decoded_as_utf8(self):
        class Response:
            encoding = "ISO-8859-1"

            def iter_lines(self, *, decode_unicode=False):
                payload = "data: I’ll generate café images.".encode("utf-8")
                yield payload.decode(self.encoding) if decode_unicode else payload

        response = Response()
        self.assertEqual(list(OpenCodeBackend._utf8_sse_lines(response)), ["data: I’ll generate café images."])
        self.assertEqual(response.encoding, "utf-8")

    def test_stop_aborts_without_reverting_session_history(self):
        backend = OpenCodeBackend({"model_catalog": [{"provider": "openai", "model": "gpt"}]})
        backend._session_id = "session-1"
        backend._ensure_session = lambda *_args: None
        aborted = threading.Event()
        calls = []

        def request(method, path, **kwargs):
            calls.append((method, path, kwargs.get("json")))
            if path.endswith("/message"):
                aborted.wait(timeout=2)
                return {}
            if path.endswith("/abort"):
                aborted.set()
                return True
            raise AssertionError(f"Unexpected OpenCode request: {method} {path}")

        backend._request = request
        with patch("shared.remote_llm.opencode_backend.requests.get", side_effect=RuntimeError("no event stream")):
            backend.run_turn("continue", system_prompt="system", tools=[], images=[], on_event=lambda _event: None, call_tool=lambda *_args: {}, should_stop=lambda: True)

        message_body = next(body for _method, path, body in calls if path.endswith("/message"))
        self.assertNotIn("messageID", message_body)
        self.assertTrue(any(path.endswith("/abort") for _method, path, _body in calls))
        self.assertFalse(any(path.endswith("/revert") for _method, path, _body in calls))
        self.assertEqual(backend._session_id, "session-1")


if __name__ == "__main__":
    unittest.main()
