"""HTTP API and standalone UI for a persistent DeepyService."""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from shared.deepy import chat, session_store
from shared.deepy.errors import error_payload
from shared.deepy.gallery import _AUDIO_EXTENSIONS, _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS
from shared.deepy.voice import mount_voice_routes, save_upload
from shared.utils.downloads import _install_routes_on_app

WEB = Path(__file__).with_name("web")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    submission_id: str = Field(min_length=1, max_length=128)
    steering: bool = False


class Control(BaseModel):
    action: Literal["stop", "pause", "reset", "resume", "queued", "abort", "rename", "delete"]
    payload: dict = Field(default_factory=dict)


def create_app(service, *, token: str | None, voice_language=None, https_port=None):
    from shared.utils.network_diagnostics import install_network_diagnostics
    install_network_diagnostics()

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Deepy", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request, call_next):
        path = request.scope["path"][len(request.scope.get("root_path", "")):]
        public = path in {"/", "/deepy_api/login"} or path.startswith("/assets/")
        supplied = request.cookies.get("deepy_access", "")
        if token is not None and not public and not secrets.compare_digest(supplied, token):
            return JSONResponse({"detail": "Sign in to Deepy."}, status_code=401)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
                return JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_value(request, error):
        payload = error_payload(error)
        return JSONResponse({"detail": payload['message'], "error": payload}, status_code=400)

    from gradio import Error as GradioError
    app.add_exception_handler(session_store.SessionStoreError, invalid_value)
    app.add_exception_handler(GradioError, invalid_value)

    @app.get("/", response_class=HTMLResponse)
    def index():
        shell = chat.render_shell_html(service._deps.controller.get_deepy_type())
        page = (WEB / "app.html").read_text(encoding="utf-8").replace("<!-- CHAT -->", shell).replace("<!-- STATS -->", chat.render_stats_html())
        def version_asset(match):
            name = match[1]
            path = WEB.parents[1] / 'gradio' / name if name in {'form_sync.js', 'progress.css'} else WEB / name
            return f'assets/{name}?v={path.stat().st_mtime_ns}'
        page = re.sub(r'assets/([\w.-]+\.(?:js|css))', version_asset, page)
        if https_port is not None:
            page = page.replace("<body data-deepy-app>", f'<body data-deepy-app data-deepy-https-port="{int(https_port)}">')
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

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
        if name in {'form_sync.js', 'progress.css'}:
            return FileResponse(WEB.parents[1] / 'gradio' / name)
        if name == "icon.png":
            return FileResponse(WEB.parents[2] / "favicon.png")
        if name not in {"chat.js", "compact_actions.js", "chat.css", "voice.js", "app.js", "app.css", "gradio_transport.js", "manifest.webmanifest", "icon.svg", "transport.js", "hybrid_transport.js", "workspaces.js", "workspaces.css", "media_view.js"}:
            raise HTTPException(404)
        return FileResponse(WEB / name, media_type="application/manifest+json" if name == "manifest.webmanifest" else None)

    @app.get("/deepy_api/state")
    def state():
        return service.snapshot()

    if service.workspace_viewer is not None:
        from shared.deepy.workspace_viewer_api import mount_workspace_viewer
        mount_workspace_viewer(app, service)

    @app.post('/deepy_api/workspaces/{action}')
    def workspace_action(action: Literal['create', 'rename', 'delete', 'select'], payload: dict):
        return service.workspace_control(action, payload)

    @app.get("/deepy_api/media/{media_id}/info")
    def media_info(media_id: str):
        with service._mutation_lock:
            if media_id not in service.gallery._media_paths:
                raise HTTPException(404, "Media no longer in this workspace.")
            return {"html": service.gallery.media_info(media_id)}

    @app.get("/deepy_api/media/{media_id}/file")
    def media_file(media_id: str, download: bool = False):
        from shared.utils.http_disconnect import DisconnectAwareFileResponse
        try:
            path = Path(service.gallery.media_path(media_id))
        except KeyError:
            raise HTTPException(404, "Media no longer in this workspace.")
        if not path.is_file():
            raise HTTPException(404, "Media file not found.")
        return DisconnectAwareFileResponse(path, filename=path.name, content_disposition_type='attachment' if download else 'inline')

    @app.get("/deepy_api/media/{media_id}/thumbnail")
    async def media_thumbnail(media_id: str):
        from fastapi import Response
        from shared.deepy.thumbnails import render_thumbnail
        try:
            path = service.gallery.media_path(media_id)
            if service.gallery._detect_media_type(path) not in ('image', 'video'):
                raise HTTPException(400, "Media has no visual thumbnail.")
            data = await render_thumbnail(path)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Media file not found.")
        if not data:
            raise HTTPException(404, "Thumbnail not available.")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/deepy_api/display-settings")
    def display_settings():
        return service.display_settings()

    @app.post("/deepy_api/display-settings")
    def update_display_settings(body: dict):
        return service.update_display_settings(body)

    @app.get("/deepy_api/settings")
    def settings():
        return service.settings()

    @app.post("/deepy_api/settings")
    def update_settings(body: dict):
        return service.submit_settings(body['baseline'], body['values']) if 'baseline' in body else service.update_settings(body)

    async def event_batches(after):
        while not service._closing:
            events = await run_in_threadpool(service.events_after, after)
            if events:
                after = events[-1]['id']
            yield events

    @app.get("/deepy_api/events")
    async def events(request: Request, after: int = 0):
        async def stream():
            async for events in event_batches(after):
                if await request.is_disconnected():
                    return
                if not events:
                    yield ": keepalive\n\n"
                for event in events:
                    yield "id: " + str(event['id']) + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.websocket('/deepy_api/events')
    async def websocket_events(socket: WebSocket, after: int = 0):
        # HTTP middleware does not run for WebSocket upgrades.
        origin = socket.headers.get('origin')
        scheme = 'https' if socket.url.scheme == 'wss' else 'http'
        if (token is not None and not secrets.compare_digest(socket.cookies.get('deepy_access', ''), token)) or (origin and origin != f'{scheme}://{socket.url.netloc}'):
            await socket.close(code=1008)
            return
        await socket.accept()

        async def send_events():
            async for events in event_batches(after):
                for event in events:
                    # Buffered sends may never suspend. Let the receiver and
                    # Windows transport process a disconnect between messages.
                    await asyncio.sleep(0)
                    if socket.client_state == WebSocketState.DISCONNECTED:
                        return
                    try:
                        await socket.send_json(event)
                    except RuntimeError:
                        # Uvicorn can finish the response as the receive task observes
                        # a closed browser, before the pending sender is cancelled.
                        if socket.client_state == WebSocketState.DISCONNECTED:
                            return
                        raise
            await socket.close()

        async def disconnected():
            async for _ in socket.iter_text():
                pass

        sender = asyncio.create_task(send_events())
        receiver = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                with suppress(WebSocketDisconnect):
                    task.result()
        finally:
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    @app.post("/deepy_api/messages", status_code=202)
    def message(body: Message):
        if not body.text.strip():
            raise HTTPException(400, "The request is empty.")
        queued = service.submit(body.text, body.submission_id, steering=body.steering)
        return {"submission_id": body.submission_id, "queued_for_restoration": queued}

    @app.post("/deepy_api/control")
    def control(body: Control):
        if body.action == "resume" and not isinstance(body.payload.get("id"), str):
            raise HTTPException(400, "A saved session id is required.")
        if body.action == "rename" and not all(isinstance(body.payload.get(key), str) for key in ('id', 'title')):
            raise HTTPException(400, "A session id and title are required.")
        return service.control(body.action, body.payload)

    @app.post("/deepy_api/media")
    async def upload(file: UploadFile = File(...), from_chat: bool = Form(False)):
        from gradio import processing_utils
        from gradio_client.utils import strip_invalid_filename_characters

        session_epoch = service._session.chat_epoch

        with TemporaryDirectory(prefix="wangp-upload-") as directory:
            path = await save_upload(file, Path(directory), limit=1024 * 1024 * 1024, extensions=_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS)
            name = strip_invalid_filename_characters(Path(file.filename).name)
            named_path = path.with_name(name)
            path.rename(named_path)
            cached_path = await run_in_threadpool(processing_utils.save_file_to_cache, named_path, processing_utils.get_upload_folder())
            media_id = await run_in_threadpool(service.import_media, cached_path, session_epoch=session_epoch, from_chat=from_chat)
        with service._mutation_lock:
            return {"id": media_id, "gallery": service.gallery_snapshot()}

    @app.post('/deepy_api/media/unattach-last')
    def unattach_last(payload: dict):
        return {'event': service.remove_last_chat_upload(payload.get('chat_session_id'))}

    @app.post("/deepy_api/media/{media_id}/select")
    def select(media_id: str):
        with service._mutation_lock:
            service.select_media(media_id)
            return {"gallery": service.gallery_snapshot()}

    _install_routes_on_app(app)
    mount_voice_routes(app, get_service=lambda: service, language=voice_language)
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


def server_options(args):
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
    return host, port, cert, key, https_port, token


def run_server(deps, args):
    import uvicorn
    from shared.deepy.service import DeepyService

    host, port, cert, key, https_port, token = server_options(args)
    print(f"Deepy server: {'https' if cert and https_port is None else 'http'}://{host}:{port}")
    if https_port is not None:
        print(f"Deepy HTTPS: https://{host}:{https_port}")
    print("Deepy authentication disabled." if token is None else f"Deepy access key: {token}")
    service = DeepyService(deps)
    service.configure_workspaces(args.workspaces_dir or str(WEB.parents[2] / 'workspaces'))
    app = create_app(service, token=token, voice_language=args.deepy_voice_language, https_port=https_port)
    if https_port is None:
        uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        _run_http_and_https(uvicorn.Config(app, host=host, port=port, lifespan="off", timeout_graceful_shutdown=5), uvicorn.Config(app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, timeout_graceful_shutdown=5))
    return 0
