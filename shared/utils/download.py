import inspect, os, shutil, sys, time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Callable


@dataclass(frozen=True)
class DownloadProgress:
    phase: str
    source: str
    filename: str
    current: int
    total: int | None
    unit: str = "bytes"
    speed_bps: float | None = None
    eta_seconds: float | None = None
    repo_id: str | None = None
    file_index: int | None = None
    file_count: int | None = None


DownloadProgressCallback = Callable[[DownloadProgress], None]


class _DownloadProgressTracker:
    def __init__(self, *, callback, source, filename, repo_id=None, unit="bytes", file_index=None, file_count=None, emit_interval=0.5, clock=time.monotonic):
        self.callback = callback
        self.source = source
        self.filename = filename
        self.repo_id = repo_id
        self.unit = unit
        self.file_index = file_index
        self.file_count = file_count
        self.emit_interval = emit_interval
        self.clock = clock
        self.lock = Lock()
        self.last_time = None
        self.last_current = None
        self.last_emit_time = None
        self.speeds = deque(maxlen=5)

    def _update(self, phase, current, total, force):
        current = max(0, int(current or 0))
        total = int(total) if total is not None and total > 0 else None
        if total is not None:
            current = min(current, total)
        now = self.clock()
        with self.lock:
            if self.last_time is None:
                self.last_time, self.last_current = now, current
            elif now - self.last_time >= self.emit_interval:
                elapsed = now - self.last_time
                transferred = current - self.last_current
                if self.unit == "bytes" and elapsed > 0 and transferred >= 0:
                    self.speeds.append(transferred / elapsed)
                self.last_time, self.last_current = now, current
            speed = sum(self.speeds) / len(self.speeds) if self.speeds else None
            eta = (total - current) / speed if self.unit == "bytes" and total is not None and current < total and speed else None
            update = DownloadProgress(
                phase=phase,
                source=self.source,
                filename=self.filename,
                current=current,
                total=total,
                unit=self.unit,
                speed_bps=speed,
                eta_seconds=eta,
                repo_id=self.repo_id,
                file_index=self.file_index,
                file_count=self.file_count,
            )
            should_emit = self.callback is not None and (self.last_emit_time is None or force or now - self.last_emit_time >= self.emit_interval)
            if should_emit:
                try:
                    self.callback(update)
                except Exception as exc:
                    print(f"Download progress callback disabled: {exc}", file=sys.stderr)
                    self.callback = None
                self.last_emit_time = now
            return update

    def update(self, current, total, *, force=False):
        return self._update("downloading", current, total, force)

    def complete(self, current, total):
        return self._update("complete", current, total, True)


def _format_bytes(value):
    value = float(value or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


class _URLProgressHook:
    def __init__(self, filename, progress_callback=None, *, source="http", repo_id=None, file_index=None, file_count=None):
        self.path = filename
        self.filename = os.path.basename(filename) or "Unknown file"
        self.tracker = _DownloadProgressTracker(
            callback=progress_callback,
            source=source,
            filename=self.filename,
            repo_id=repo_id,
            file_index=file_index,
            file_count=file_count,
        )
        self.current = 0
        self.total = None

    def __call__(self, block_num, block_size, total_size):
        self.total = total_size if total_size > 0 else None
        self.current = block_num * block_size
        if self.total is not None:
            self.current = min(self.current, self.total)
        update = self.tracker.update(self.current, self.total)
        speed = f" @ {_format_bytes(update.speed_bps)}/s" if update.speed_bps else ""
        if self.total is None:
            line = f"\r{self.filename}: {_format_bytes(self.current)}{speed}"
            sys.stdout.write(line.ljust(80))
        else:
            percent = min(100, self.current / self.total * 100)
            filled = int(40 * percent / 100)
            bar = "█" * filled + "░" * (40 - filled)
            line = f"\r{self.filename}: [{bar}] {percent:.1f}% ({_format_bytes(self.current)}/{_format_bytes(self.total)}){speed}"
            sys.stdout.write(line.ljust(100))
            if self.current >= self.total:
                sys.stdout.write("\n")
        sys.stdout.flush()
        return update

    def complete(self):
        current = os.path.getsize(self.path) if os.path.isfile(self.path) else self.current
        return self.tracker.complete(current, self.total)


def progress_hook(block_num, block_size, total_size, filename=None):
    """Compatibility console hook; use create_progress_hook for transfer state."""
    return _URLProgressHook(filename or "Unknown file")(block_num, block_size, total_size)


def create_progress_hook(filename, progress_callback=None, *, source="http", repo_id=None, file_index=None, file_count=None):
    return _URLProgressHook(filename, progress_callback, source=source, repo_id=repo_id, file_index=file_index, file_count=file_count)


def create_hf_progress_class(progress_callback, *, source, filename, repo_id=None, unit="bytes", file_index=None, file_count=None):
    from tqdm.auto import tqdm

    class HuggingFaceProgress(tqdm):
        _bars = []

        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            super().__init__(*args, **kwargs)
            self._download_tracker = _DownloadProgressTracker(
                callback=progress_callback,
                source=source,
                filename=filename,
                repo_id=repo_id,
                unit=unit,
                file_index=file_index,
                file_count=file_count,
            )
            self._download_tracker.update(self.n, self.total, force=True)
            self._download_closed = False
            self.__class__._bars.append(self)

        def update(self, n=1):
            if self.disable:
                self.n += n
                result = None
            else:
                result = super().update(n)
            self._download_tracker.update(self.n, self.total)
            return result

        def close(self):
            if self._download_closed:
                return None
            self._download_closed = True
            result = super().close()
            self._download_tracker.update(self.n, self.total, force=True)
            return result

        @classmethod
        def finish(cls, success=True):
            for bar in cls._bars:
                bar.close()
                if success:
                    bar._download_tracker.complete(bar.n, bar.total)
            cls._bars.clear()

    return HuggingFaceProgress


def _accepts_tqdm_class(download):
    try:
        parameters = inspect.signature(download).parameters
    except (TypeError, ValueError):
        return False
    return "tqdm_class" in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


_ACTIVE_HF_PROGRESS_CLASS = ContextVar("wangp_hf_progress_class", default=None)
_HF_PROGRESS_ROUTER_LOCK = Lock()
_HF_TRANSFER_PROGRESS_NAMES = {"huggingface_hub.http_get", "huggingface_hub.xet_get"}


def _install_hf_progress_router():
    try:
        from huggingface_hub import file_download
        progress_context = file_download._get_progress_bar_context
        parameters = inspect.signature(progress_context).parameters
    except (AttributeError, ImportError, TypeError, ValueError):
        return False
    if "_tqdm_bar" not in parameters or "name" not in parameters:
        return False
    if getattr(progress_context, "_wangp_progress_router", False):
        return True

    with _HF_PROGRESS_ROUTER_LOCK:
        progress_context = file_download._get_progress_bar_context
        if getattr(progress_context, "_wangp_progress_router", False):
            return True
        try:
            signature = inspect.signature(progress_context)
        except (TypeError, ValueError):
            return False
        if "_tqdm_bar" not in signature.parameters or "name" not in signature.parameters:
            return False
        is_tqdm_disabled = getattr(progress_context, "__globals__", {}).get("is_tqdm_disabled")

        @wraps(progress_context)
        def route_progress(*args, **kwargs):
            Progress = _ACTIVE_HF_PROGRESS_CLASS.get()
            if Progress is None:
                return progress_context(*args, **kwargs)
            try:
                values = signature.bind_partial(*args, **kwargs).arguments
            except TypeError:
                return progress_context(*args, **kwargs)
            if values.get("name") not in _HF_TRANSFER_PROGRESS_NAMES or values.get("_tqdm_bar") is not None:
                return progress_context(*args, **kwargs)
            disable = is_tqdm_disabled(log_level=values.get("log_level")) if callable(is_tqdm_disabled) else None
            kwargs["_tqdm_bar"] = Progress(
                total=values.get("total"),
                initial=values.get("initial", 0),
                desc=values.get("desc"),
                unit=values.get("unit", "B"),
                unit_scale=values.get("unit_scale", True),
                disable=disable,
            )
            return progress_context(*args, **kwargs)

        route_progress._wangp_progress_router = True
        file_download._get_progress_bar_context = route_progress
    return True


def _run_hf_download(download, kwargs, progress_callback=None, progress_options=None, *, legacy_router=True):
    if progress_callback is None:
        return download(**kwargs)

    Progress = create_hf_progress_class(progress_callback, **progress_options)
    token = None
    fallback = None
    if _accepts_tqdm_class(download):
        kwargs = {**kwargs, "tqdm_class": Progress}
    elif legacy_router and _install_hf_progress_router():
        token = _ACTIVE_HF_PROGRESS_CLASS.set(Progress)
    else:
        fallback = _DownloadProgressTracker(callback=progress_callback, **progress_options)
        fallback.update(0, None, force=True)

    try:
        result = download(**kwargs)
    except Exception:
        Progress.finish(success=False)
        raise
    else:
        Progress.finish()
        if fallback is not None:
            fallback.complete(0, None)
        return result
    finally:
        if token is not None:
            _ACTIVE_HF_PROGRESS_CLASS.reset(token)


def process_files_def(repoId=None, sourceFolderList=None, fileList=None, targetFolderList=None, progress_callback=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    from shared.utils import files_locator as fl

    if targetFolderList is None:
        targetFolderList = [None] * len(sourceFolderList)
    for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList, fileList):
        if targetFolder is not None and len(targetFolder) == 0:
            targetFolder = None
        explicit_target = targetFolder if targetFolder is not None else (sourceFolder if len(sourceFolder) > 0 else None)
        targetRoot = fl.get_smart_download_root(explicit_target)
        local_dir = os.path.join(targetRoot, targetFolder) if targetFolder is not None else targetRoot
        if len(files) == 0:
            if fl.locate_folder(sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder), error_if_none=False) is None:
                _run_hf_download(
                    snapshot_download,
                    {"repo_id": repoId, "allow_patterns": sourceFolder + "/*", "local_dir": local_dir},
                    progress_callback,
                    {"source": "huggingface_snapshot", "filename": f"{repoId}/{sourceFolder}", "repo_id": repoId, "unit": "files"},
                    legacy_router=False,
                )
        else:
            for file_index, onefile in enumerate(files, 1):
                display_name = f"{sourceFolder}/{onefile}" if len(sourceFolder) > 0 else onefile
                progress_options = {"source": "huggingface", "filename": display_name, "repo_id": repoId, "file_index": file_index, "file_count": len(files)}
                if len(sourceFolder) > 0:
                    if fl.locate_file((sourceFolder + "/" + onefile) if targetFolder is None else os.path.join(targetFolder, sourceFolder, onefile), error_if_none=False) is None:
                        _run_hf_download(
                            hf_hub_download,
                            {"repo_id": repoId, "filename": onefile, "local_dir": local_dir, "subfolder": sourceFolder},
                            progress_callback,
                            progress_options,
                        )
                else:
                    if fl.locate_file(onefile if targetFolder is None else os.path.join(targetFolder, onefile), error_if_none=False) is None:
                        _run_hf_download(
                            hf_hub_download,
                            {"repo_id": repoId, "filename": onefile, "local_dir": local_dir},
                            progress_callback,
                            progress_options,
                        )


def _download_relpath(source_folder, filename, target_folder=None):
    source_folder = "" if source_folder is None else source_folder
    if target_folder is not None and len(target_folder) == 0:
        target_folder = None
    if target_folder is None:
        return os.path.join(source_folder, filename) if len(source_folder) > 0 else filename
    return os.path.join(target_folder, source_folder, filename) if len(source_folder) > 0 else os.path.join(target_folder, filename)


def download_def_missing_files(download_def):
    from shared.utils import files_locator as fl

    if download_def is None:
        return []
    if isinstance(download_def, list):
        missing = []
        for one_def in download_def:
            missing.extend(download_def_missing_files(one_def))
        return missing
    source_folders = download_def.get("sourceFolderList", [])
    file_lists = download_def.get("fileList", [])
    target_folders = download_def.get("targetFolderList")
    if target_folders is None:
        target_folders = [None] * len(source_folders)
    missing = []
    for source_folder, files, target_folder in zip(source_folders, file_lists, target_folders):
        if len(files) == 0:
            rel_folder = _download_relpath(source_folder, "", target_folder).rstrip("\\/")
            if fl.locate_folder(rel_folder, error_if_none=False) is None:
                missing.append(rel_folder)
            continue
        for filename in files:
            rel_path = _download_relpath(source_folder, filename, target_folder)
            if fl.locate_file(rel_path, error_if_none=False) is None:
                missing.append(rel_path)
    return missing


def send_download_status(send_cmd=None, status_text=None):
    if send_cmd is not None and status_text:
        send_cmd("status", status_text)


def process_files_def_if_needed(download_def, send_cmd=None, status_text=None, progress_callback=None):
    if download_def is None or len(download_def_missing_files(download_def)) == 0:
        return False
    send_download_status(send_cmd, status_text)
    if isinstance(download_def, list):
        for one_def in download_def:
            process_files_def(**one_def, progress_callback=progress_callback)
    else:
        process_files_def(**download_def, progress_callback=progress_callback)
    return True


def query_audio_background_replacement_download_def():
    return {
        "repoId": "DeepBeepMeep/Wan2.1",
        "sourceFolderList": ["roformer"],
        "fileList": [["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"]],
    }


def download_audio_background_replacement(send_cmd=None, status_text="Downloading audio background replacement model files..."):
    return process_files_def_if_needed(query_audio_background_replacement_download_def(), send_cmd=send_cmd, status_text=status_text)


def process_download_defs(download_defs, progress_callback=None):
    if isinstance(download_defs, dict):
        process_files_def(**download_defs, progress_callback=progress_callback)
        return
    for download_def in download_defs or []:
        if download_def is not None:
            process_files_def(**download_def, progress_callback=progress_callback)


def download_file(url, filename, progress_callback=None):
    from huggingface_hub import hf_hub_download
    from shared.utils import files_locator as fl

    url = url.split("|")[0]
    if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
        base_dir = os.path.dirname(filename)
        url = url[len("https://huggingface.co/"):]
        url_parts = url.split("/resolve/main/")
        repoId = url_parts[0]
        onefile = os.path.basename(url_parts[-1])
        sourceFolder = os.path.dirname(url_parts[-1])
        display_name = f"{sourceFolder}/{onefile}" if len(sourceFolder) > 0 else onefile
        progress_options = {"source": "huggingface", "filename": display_name, "repo_id": repoId}
        if len(sourceFolder) == 0:
            _run_hf_download(
                hf_hub_download,
                {"repo_id": repoId, "filename": onefile, "local_dir": fl.get_download_location() if len(base_dir) == 0 else base_dir},
                progress_callback,
                progress_options,
            )
        else:
            tgt = fl.get_download_location() if len(base_dir) == 0 else base_dir
            os.makedirs(tgt, exist_ok=True)
            temp_dir_path = os.path.join(tgt, f"_temp{time.time()}")
            temp_full_path = os.path.join(temp_dir_path, sourceFolder)
            os.makedirs(temp_full_path, exist_ok=True)
            _run_hf_download(
                hf_hub_download,
                {"repo_id": repoId, "filename": onefile, "local_dir": temp_dir_path, "subfolder": sourceFolder},
                progress_callback,
                progress_options,
            )
            shutil.move(os.path.join(temp_full_path, onefile), tgt)
            shutil.rmtree(temp_dir_path)
    else:
        from urllib.request import urlretrieve

        hook = create_progress_hook(filename, progress_callback)
        urlretrieve(url, filename, hook)
        hook.complete()

