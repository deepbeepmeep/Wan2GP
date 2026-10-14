# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Restart-capable, range-based transfers backed by original files and one upload partial."""

import base64
import hashlib
import mimetypes
import os
import time
from pathlib import Path

from .server import _MediaTransferStore, _MAX_UPLOAD_BYTES, _compact_gallery_record, _gallery_item, _mcp_media_type, _register_gallery_media, _resolve_io_source

CHUNK_BYTES = 8 * 1024**2
TTL_SECONDS = 24 * 60 * 60


def file_version(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]


def checksum(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


class ResumableTransfers(_MediaTransferStore):
    def __init__(self, session, policy, state, *, http):
        super().__init__(session, policy, http=http)
        self.state = state
        self.scope = hashlib.sha256(repr((policy.mode, [str(root) for root in policy.output_roots])).encode()).hexdigest()

    def _seal(self, kind, **fields):
        context = getattr(self.session, '_agent_context', None)
        token = self.state.seal({"kind": kind, "scope": self.scope, "expires": time.time() + TTL_SECONDS, **({'agent_context': context} if context else {}), **fields})
        return f"{token}.{context}" if context else token  # the suffix routes HTTP transfers to the agent context holding the grant

    def _open(self, token, kind):
        record = self.state.unseal(token.partition(".")[0], kind)
        if record["scope"] != self.scope:
            raise ValueError("Transfer permissions have changed; request a new transfer.")
        return record

    def create_file_download(self, source, *, include_sha256=False):
        path, gallery = _resolve_io_source(self.session, self.policy, source, file=True)
        path = Path(path).resolve()
        version = file_version(path)
        digest = checksum(path) if include_sha256 else None
        if file_version(path) != version:
            raise ValueError("Source changed while preparing the download; request a new transfer.")
        token = self._seal("download", path=str(path), version=version, gallery=bool(gallery))
        result = {"status": "done", "token": token, "path": self.policy.virtualize_path(path), "filename": path.name,
                  "size_bytes": version[0], "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                  "expires_in_seconds": TTL_SECONDS, "resource_uri": f"wangp://files/{token}/0/{min(version[0], CHUNK_BYTES)}",
                  "resource_chunk_bytes": CHUNK_BYTES, "resumable": True,
                  "continuation": "Read wangp://files/<token>/<offset>/<length>; offsets count decoded bytes and length is at most 8388608. Keep token and offset across reconnects."}
        if digest:
            result["sha256"] = digest
        if self.http:
            result.update(download_url=f"/wangp_api/files/download/{token}", method="GET", range_header="Range: bytes=<offset>-<last-byte>")
        return result

    def create_download(self, media_id):
        record = _gallery_item(self.session, media_id)
        return {**self.create_file_download(media_id), "media_id": record["media_id"], "media_type": record["media_type"]}

    def download_path(self, token):
        record = self._open(token, "download")
        path = Path(record["path"])
        # A signed Gallery grant authorizes this exact file even after in-memory Gallery history is gone.
        if not record["gallery"]:
            self.policy.require_read(path, file=True)
        if not path.is_file() or file_version(path) != record["version"]:
            raise ValueError("Download source changed or vanished; request a new transfer.")
        return path

    def read_file_range(self, token, offset, length):
        if offset < 0 or not 0 <= length <= CHUNK_BYTES:
            raise ValueError("offset must be non-negative and length must be 0..8388608 bytes.")
        path = self.download_path(token)
        with path.open("rb") as reader:
            if offset > path.stat().st_size:
                raise ValueError("offset is beyond the end of the file.")
            reader.seek(offset)
            data = reader.read(length)
        self.download_path(token)
        return data

    def create_upload(self, filename, *, gallery=True, size_bytes=None, sha256=None):
        if not filename or Path(filename).name != filename or filename in {".", ".."} or any(char in filename for char in ':\\/\x00'):
            raise ValueError("filename must be a non-empty file name, without directories.")
        if gallery and not _mcp_media_type(filename):
            raise ValueError("filename must have a supported image, video, audio, or 3D extension.")
        if size_bytes is not None and (isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or not 0 <= size_bytes <= _MAX_UPLOAD_BYTES):
            raise ValueError(f"size_bytes must be an integer in 0..{_MAX_UPLOAD_BYTES}.")
        if sha256 is not None and (len(sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in sha256)):
            raise ValueError("sha256 must contain 64 hexadecimal characters.")
        path = self.upload_path(filename, gallery=gallery)
        partial = path.with_name(path.name + ".part")
        with partial.open("xb") as writer:
            writer.flush()
            os.fsync(writer.fileno())
        token = self._seal("upload", path=str(path), gallery=gallery, size_bytes=size_bytes, sha256=sha256.lower() if sha256 else None)
        result = {"token": token, "filename": filename, "expires_in_seconds": TTL_SECONDS, "max_bytes": _MAX_UPLOAD_BYTES,
                  "chunk_bytes": CHUNK_BYTES, "offset": 0, "size_bytes": size_bytes, "resumable": True,
                  "next": "Use upload_chunk(token, offset, data_base64), get_upload(token) after reconnects, then finish_upload(token). HTTP PUT also accepts Upload-Offset; complete uploads finalize automatically."}
        if self.http:
            route = "gallery" if gallery else "files"
            result.update(upload_url=f"/wangp_api/{route}/upload/{token}", method="PUT")
        return result

    def _upload_paths(self, token):
        record = self._open(token, "upload")
        path = Path(record["path"])
        # The sealed destination retains the original workspace across server restarts.
        if not any(path.resolve().is_relative_to(root.resolve()) for root in self.policy.output_roots) and not path.resolve().is_relative_to(self.policy.workspace.resolve()):
            raise PermissionError("The upload destination is no longer within the server's output/workspace roots.")
        return record, path, path.with_name(path.name + ".part")

    def _upload_status(self, record, path, partial):
        complete = path.is_file()
        result = {"status": "done" if complete else "uploading", "offset": (path if complete else partial).stat().st_size, "size_bytes": record["size_bytes"], "filename": path.name}
        if complete:
            result.update(path=self.policy.virtualize_path(path), mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if record["gallery"]:
                result.update(_compact_gallery_record(_register_gallery_media(self.session, str(path))))
        return result

    def get_upload(self, token):
        with self.state.transaction():
            return self._upload_status(*self._upload_paths(token))

    def upload_bytes(self, token, offset, data):
        if offset < 0 or len(data) > CHUNK_BYTES:
            raise ValueError("offset must be non-negative; each chunk is limited to 8388608 decoded bytes.")
        with self.state.transaction():
            record, path, partial = self._upload_paths(token)
            current = path if path.is_file() else partial
            size = current.stat().st_size
            limit = _MAX_UPLOAD_BYTES if record["size_bytes"] is None else record["size_bytes"]
            if offset + len(data) > limit:
                raise ValueError("Chunk exceeds the declared upload size or server upload limit.")
            if offset < size or path.is_file():
                with current.open("rb") as reader:
                    reader.seek(offset)
                    if reader.read(len(data)) != data:
                        raise ValueError("Retry overlaps different or incomplete bytes; call get_upload and resume from its offset.")
            elif offset != size:
                raise ValueError(f"Upload offset mismatch; next offset is {size}.")
            else:
                with partial.open("ab") as writer:
                    writer.write(data)
                    writer.flush()
                    os.fsync(writer.fileno())
            return self._upload_status(record, path, partial)

    def upload_chunk(self, token, offset, data_base64):
        if len(data_base64) > 4 * ((CHUNK_BYTES + 2) // 3):
            raise ValueError("Chunk exceeds 8388608 decoded bytes.")
        try:
            data = base64.b64decode(data_base64, validate=True)
        except ValueError as exc:
            raise ValueError("data_base64 must be valid base64.") from exc
        return self.upload_bytes(token, offset, data)

    def finish_upload(self, token):
        with self.state.transaction():
            record, path, partial = self._upload_paths(token)
            if not path.exists():
                if record["size_bytes"] is not None and partial.stat().st_size != record["size_bytes"]:
                    raise ValueError("Upload is incomplete; resume from get_upload's offset.")
                if record["sha256"] and checksum(partial) != record["sha256"]:
                    raise ValueError("Upload SHA-256 mismatch; the partial was not published. Start a new upload.")
                os.replace(partial, path)
            result = self._upload_status(record, path, partial)
            if record["sha256"]:
                result["sha256"] = record["sha256"]
            return result

    async def http_upload(self, request):
        from starlette.responses import JSONResponse
        try:
            token = request.path_params["token"]
            if request.method == "GET":
                return JSONResponse(self.get_upload(token))
            offset = int(request.headers.get("upload-offset", "0"))
            if offset < 0:
                raise ValueError("Upload-Offset must be non-negative.")
            # Commit bounded pieces; the on-disk length remains authoritative even after disconnects.
            async for chunk in request.stream():
                for start in range(0, len(chunk), CHUNK_BYTES):
                    piece = chunk[start:start + CHUNK_BYTES]
                    self.upload_bytes(token, offset, piece)
                    offset += len(piece)
            result = self.get_upload(token)
            if result["size_bytes"] == result["offset"] or "upload-offset" not in request.headers and result["size_bytes"] is None:
                result = self.finish_upload(token)
            return JSONResponse(result, headers={"Upload-Offset": str(result["offset"])})
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    async def http_download(self, request):
        from starlette.responses import FileResponse, JSONResponse
        try:
            path = self.download_path(request.path_params["token"])
            return FileResponse(path, filename=path.name, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
