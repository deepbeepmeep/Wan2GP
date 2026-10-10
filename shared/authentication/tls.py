"""TLS settings and HTTP-to-HTTPS redirects for every network interface."""
import os

from starlette.datastructures import URL
from starlette.responses import RedirectResponse


def tls_options(args, port):
    cert = args.ssl_certfile or os.getenv("WANGP_SSL_CERT")
    key = args.ssl_keyfile or os.getenv("WANGP_SSL_KEY")
    https_port = args.https_port
    if bool(cert) != bool(key) or (https_port is not None and not cert):
        raise ValueError("HTTPS requires both --ssl-certfile and --ssl-keyfile (or WANGP_SSL_CERT and WANGP_SSL_KEY).")
    if https_port is not None and (not 1 <= https_port <= 65535 or https_port == port):
        raise ValueError("--https-port must be between 1 and 65535 and differ from the main port.")
    if cert:
        import ssl
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
    return cert, key, https_port


class HTTPSRedirect:
    def __init__(self, app, port):
        self.app, self.port = app, port

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"} and scope["scheme"] in {"http", "ws"}:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            url = URL(scope=scope).replace(scheme="https", port=self.port)
            return await RedirectResponse(str(url), status_code=307)(scope, receive, send)
        return await self.app(scope, receive, send)


def run_http_and_https(http_config, https_config):
    import threading
    import uvicorn
    http_config.load()
    https_config.load()
    http, https = uvicorn.Server(http_config), uvicorn.Server(https_config)

    def serve_http():
        try:
            http.run()
        finally:
            https.should_exit = True

    worker = threading.Thread(target=serve_http, name="WanGP HTTP Redirect")
    worker.start()
    try:
        https.run()
    finally:
        http.should_exit = True
        worker.join()
