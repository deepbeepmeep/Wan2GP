"""HTTP API and standalone UI for a persistent DeepyService."""
from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from shared.deepy import chat
from shared.deepy.gallery import _AUDIO_EXTENSIONS, _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS
from shared.deepy.voice import mount_voice_routes, save_upload
from shared.utils.downloads import _install_routes_on_app

WEB = Path(__file__).with_name("web")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    submission_id: str = Field(min_length=1, max_length=128)
    steering: bool = False


class Control(BaseModel):
    action: Literal["stop", "pause", "reset", "resume", "queued", "abort"]
    payload: dict = Field(default_factory=dict)


def create_app(service, *, token: str | None, voice_language=None, https_port=None):
    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Deepy", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request, call_next):
        public = request.url.path in {"/", "/deepy_api/login"} or request.url.path.startswith("/assets/")
        supplied = request.cookies.get("deepy_access", "")
        if token is not None and not public and not secrets.compare_digest(supplied, token):
            return JSONResponse({"detail": "Sign in to Deepy."}, status_code=401)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_value(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        shell = chat.render_shell_html(service._deps.controller.get_deepy_type())
        page = (WEB / "app.html").read_text(encoding="utf-8").replace("<!-- CHAT -->", shell).replace("<!-- STATS -->", chat.render_stats_html())
        return page if https_port is None else page.replace("<body data-deepy-app>", f'<body data-deepy-app data-deepy-https-port="{int(https_port)}">')

    @app.post("/deepy_api/login")
    async def login(request: Request):
        if token is None:
            return {"ok": True}
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("token"), str) or not secrets.compare_digest(body["token"], token):
            raise HTTPException(401, "Incorrect access key.")
        response = JSONResponse({"ok": True})
        response.set_cookie("deepy_access", token, httponly=True, secure=request.url.scheme == "https", samesite="strict")
        return response

    @app.get("/assets/{name}")
    def asset(name: str):
        if name == "icon.png":
            return FileResponse(WEB.parents[2] / "favicon.png")
        if name not in {"chat.js", "chat.css", "voice.js", "app.js", "app.css", "gradio_transport.js", "manifest.webmanifest", "icon.svg"}:
            raise HTTPException(404)
        return FileResponse(WEB / name, media_type="application/manifest+json" if name == "manifest.webmanifest" else None)

    @app.get("/deepy_api/state")
    def state():
        return service.snapshot()

    @app.get("/deepy_api/media/{media_id}/info")
    def media_info(media_id: str):
        return {"html": service.gallery.media_info(media_id)}

    @app.get("/deepy_api/settings")
    def settings():
        return service.settings()

    @app.post("/deepy_api/settings")
    def update_settings(body: dict):
        return service.update_settings(body)

    @app.get("/deepy_api/events")
    async def events(request: Request, after: int = 0):
        async def stream():
            cursor = after
            while not await request.is_disconnected():
                events = await run_in_threadpool(service.events_after, cursor)
                if not events:
                    yield ": keepalive\n\n"
                for event in events:
                    cursor = event["id"]
                    yield "id: " + str(cursor) + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/deepy_api/messages", status_code=202)
    def message(body: Message):
        if not body.text.strip():
            raise HTTPException(400, "The request is empty.")
        service.submit(body.text, body.submission_id, steering=body.steering)
        return {"submission_id": body.submission_id}

    @app.post("/deepy_api/control")
    def control(body: Control):
        if body.action == "resume" and not isinstance(body.payload.get("id"), str):
            raise HTTPException(400, "A saved session id is required.")
        return service.control(body.action, body.payload)

    @app.post("/deepy_api/media")
    async def upload(file: UploadFile = File(...)):
        directory = Path(__file__).resolve().parents[2] / "deepy_uploads"
        path = await save_upload(file, directory, limit=1024 * 1024 * 1024, extensions=_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS)
        try:
            media_id = await run_in_threadpool(service.import_media, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {"id": media_id, "gallery": service.gallery.snapshot()}

    @app.post("/deepy_api/media/{media_id}/select")
    def select(media_id: str):
        service.select_media(media_id)
        return {"gallery": service.gallery.snapshot()}

    _install_routes_on_app(app)
    mount_voice_routes(app, language=voice_language)
    return app


def _run_http_and_https(http_config, https_config):
    import threading
    import uvicorn

    # Validate TLS before either port starts accepting requests.
    http_config.load()
    https_config.load()
    http, https = uvicorn.Server(http_config), uvicorn.Server(https_config)

    def serve_http():
        try:
            http.run()
        finally:
            https.should_exit = True

    worker = threading.Thread(target=serve_http, name="Deepy HTTP")
    worker.start()
    try:
        https.run()
    finally:
        http.should_exit = True
        worker.join()


def run_server(deps, args):
    import uvicorn
    from shared.deepy.service import DeepyService

    token = None if args.deepy_no_auth else os.environ.get("DEEPY_SERVER_TOKEN") or secrets.token_urlsafe(24)
    host = "0.0.0.0" if args.listen else args.server_name or os.getenv("SERVER_NAME", "localhost")
    port = int(args.server_port) or int(os.getenv("SERVER_PORT", "7860"))
    cert = args.deepy_certfile or os.environ.get("DEEPY_SERVER_CERT")
    key = args.deepy_keyfile or os.environ.get("DEEPY_SERVER_KEY")
    https_port = args.deepy_https_port
    if bool(cert) != bool(key) or (https_port is not None and not cert):
        raise ValueError("HTTPS requires both --deepy-certfile and --deepy-keyfile (or DEEPY_SERVER_CERT and DEEPY_SERVER_KEY).")
    if https_port is not None and (not 1 <= https_port <= 65535 or https_port == port):
        raise ValueError("--deepy-https-port must be between 1 and 65535 and differ from --server-port.")
    print(f"Deepy server: {'https' if cert and https_port is None else 'http'}://{host}:{port}")
    if https_port is not None:
        print(f"Deepy HTTPS: https://{host}:{https_port}")
    print("Deepy authentication disabled." if token is None else f"Deepy access key: {token}")
    app = create_app(DeepyService(deps), token=token, voice_language=args.deepy_voice_language, https_port=https_port)
    if https_port is None:
        uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        _run_http_and_https(uvicorn.Config(app, host=host, port=port, lifespan="off", timeout_graceful_shutdown=5), uvicorn.Config(app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, timeout_graceful_shutdown=5))
    return 0
