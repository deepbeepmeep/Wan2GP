"""Cache Gradio constructor bookkeeping and safely traverse shared state."""
import inspect
import os
import shutil
import tempfile
from functools import lru_cache, wraps
from pathlib import Path


def _updateable(fn):
    from gradio.component_meta import get_local_contexts

    # Gradio 5.29 inspects the same constructor on every component/update.
    fn_args = inspect.getfullargspec(fn).args

    @wraps(fn)
    def wrapper(*args, **kwargs):
        self = args[0]
        initialized_before = hasattr(self, '_constructor_args')
        if not initialized_before:
            self._constructor_args = []
        for i, arg in enumerate(args):
            if i == 0 or i >= len(fn_args):
                continue
            kwargs[fn_args[i]] = arg
        self._constructor_args.append(kwargs)
        in_event_listener, is_render = get_local_contexts()
        if in_event_listener and initialized_before and not is_render:
            return None
        return fn(self, **kwargs)

    return wrapper


@lru_cache(maxsize=8)
def _resolved_upload_folder(temp_dir):
    return str((Path(temp_dir) / 'gradio').resolve())


def _get_upload_folder():
    # Preserve explicit overrides, including relative paths, verbatim. Key the
    # default by its absolute base so tempfile.tempdir/cwd changes invalidate it.
    return os.environ.get('GRADIO_TEMP_DIR') or _resolved_upload_folder(os.path.abspath(tempfile.gettempdir()))


def _move_uploaded_files_to_cache(files, destinations):
    for source, destination in zip(files, destinations):
        # Upload destinations include the content hash. On Windows rename fails
        # when one already exists; shutil.move would then truncate the cached
        # file while browsers may still be reading it with its original length.
        if Path(destination).exists():
            Path(source).unlink()
        else:
            shutil.move(source, destination)


def _snapshot_traverse(value, func, is_root):
    if is_root(value):
        return func(value)
    if isinstance(value, dict):
        # gr.State can retain dictionaries shared with a background worker.
        # Snapshot entries before recursion/file I/O can let that worker resize
        # the mapping. Keep the original leaves and file-cache processing.
        return {key: _snapshot_traverse(item, func, is_root) for key, item in list(value.items())}
    if isinstance(value, (list, tuple)):
        return [_snapshot_traverse(item, func, is_root) for item in value]
    return value


def install():
    from gradio import blocks, component_meta, data_classes, processing_utils, routes, utils
    from gradio_client import utils as client_utils
    from shared.gradio import gradio_frontend_patch, gradio_model_change_queue, gradio_queue_wakeup_patch

    gradio_queue_wakeup_patch.install()
    gradio_model_change_queue.install()
    gradio_frontend_patch.install()
    routes.move_uploaded_files_to_cache = _move_uploaded_files_to_cache

    if component_meta.updateable is _updateable:
        return
    original = component_meta.updateable
    wrapper_code = original(lambda self: None).__code__
    # Most built-ins are imported before WanGP installs its patches. Rewrap only
    # Gradio's own updateable wrappers, leaving plugin constructor decorators intact.
    pending, visited = [blocks.Block], set()
    while pending:
        cls = pending.pop()
        if cls in visited:
            continue
        visited.add(cls)
        pending.extend(cls.__subclasses__())
        constructor = cls.__dict__.get('__init__')
        if inspect.isfunction(constructor) and constructor.__code__ is wrapper_code:
            cls.__init__ = _updateable(constructor.__wrapped__)
    component_meta.updateable = _updateable
    blocks.get_upload_folder = utils.get_upload_folder = processing_utils.get_upload_folder = _get_upload_folder
    client_utils.traverse = data_classes.traverse = _snapshot_traverse
