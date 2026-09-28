"""Provide Gradio with the configured public origin behind an HTTP proxy."""

from urllib.parse import urlsplit


class PublicURLMiddleware:
    def __init__(self, app, public_url):
        self.app = app
        parsed = urlsplit(public_url)
        self.host = parsed.netloc.encode("ascii")
        self.scheme = parsed.scheme.encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        # Gradio uses these headers when constructing its browser API and queue
        # URLs. Do not change ASGI root_path: a URL there breaks mounted routes.
        headers = [(key, value) for key, value in scope["headers"]
                   if key.lower() not in {b"x-forwarded-host", b"x-forwarded-proto"}]
        headers.extend(((b"x-forwarded-host", self.host), (b"x-forwarded-proto", self.scheme)))
        return await self.app({**scope, "headers": headers}, receive, send)
