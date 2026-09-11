"""Native Gradio workspace viewer routes; not mounted by standalone Deepy."""
from typing import Literal
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field, StrictBool, StrictInt
from starlette.concurrency import run_in_threadpool

from shared.deepy.gallery import _AUDIO_EXTENSIONS, _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS
from shared.deepy.voice import save_upload
from shared.utils.media_imports import persist_gallery_import


class ViewerAction(BaseModel):
    workspace: str
    source: Literal['video', 'audio']
    revision: str
    keys: list[str] = Field(min_length=1)
    before: str | None = None
    edge: Literal['start'] | None = None
    target: str | None = None
    name: str = ''
    confirmed: bool = False


class ViewerPage(BaseModel):
    workspace: str
    source: Literal['video', 'audio'] = 'video'
    page: int = Field(default=0, ge=0)
    selected: list[str] = Field(default_factory=list)
    initial: bool = False


class ViewerRetention(BaseModel):
    days: StrictInt


class ViewerProtection(BaseModel):
    workspace: str
    protected: StrictBool


def mount_workspace_viewer(app, service):
    viewer = service.workspace_viewer

    @app.get('/deepy_api/workspace_viewer')
    def page(workspace: str, source: Literal['video', 'audio'] = 'video', page: int = Query(0, ge=0)):
        return viewer.page(workspace, source, page)

    @app.post('/deepy_api/workspace_viewer')
    def selected_page(payload: ViewerPage):
        return viewer.page(**payload.model_dump())

    @app.get('/deepy_api/workspace_viewer/retention')
    def retention():
        return viewer.retention()

    @app.post('/deepy_api/workspace_viewer/retention')
    def save_retention(payload: ViewerRetention):
        return viewer.retention(payload.days)

    @app.post('/deepy_api/workspace_viewer/protection')
    def protection(payload: ViewerProtection):
        return viewer.protect(payload.workspace, payload.protected)

    @app.get('/deepy_api/workspace_viewer/info')
    def info(workspace: str, source: Literal['video', 'audio'], key: str):
        return viewer.info(workspace, source, key)

    @app.get('/deepy_api/workspace_viewer/thumbnail')
    async def thumbnail(workspace: str, source: Literal['video', 'audio'], key: str):
        from shared.deepy.thumbnails import render_thumbnail
        path = await run_in_threadpool(viewer.thumbnail_path, workspace, source, key)
        if service.gallery._detect_media_type(path) not in ('image', 'video'):
            raise HTTPException(400, 'Not an image or video.')
        try:
            image = await render_thumbnail(path)
        except FileNotFoundError:
            raise HTTPException(404, 'Media file not found.')
        if not image:
            raise HTTPException(404, 'Preview unavailable.')
        return Response(image, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=3600'})

    @app.post('/deepy_api/workspace_viewer/import')
    async def upload(workspace: str, file: UploadFile = File(...)):
        # The target is captured before uploading: never import into a workspace
        # selected by another browser while this upload was in flight.
        output = Path(service._deps.get_server_config()['save_path'])
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.import-', dir=output) as temporary:
            source = await save_upload(file, Path(temporary), limit=1024 * 1024 * 1024, extensions=_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS, preserve_filename=True)
            def add():
                with service._mutation_lock:
                    if workspace != service.workspace_id:
                        raise ValueError('The active workspace changed during upload. Please try again.')
                    path, reused = persist_gallery_import(source, output, move=True)
                    try:
                        media_id = service.import_media(path, context='gallery')
                    except Exception:
                        if not reused:
                            Path(path).unlink(missing_ok=True)
                        raise
                    return {'id': media_id, 'reused': reused, 'filename': Path(path).name}
            return await run_in_threadpool(add)

    @app.post('/deepy_api/workspace_viewer/{action}')
    def action(action: Literal['reorder', 'eject', 'delete', 'copy', 'archive'], payload: ViewerAction):
        return viewer.action(action, payload.model_dump())
