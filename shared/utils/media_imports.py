"""Durable gallery imports with readable, collision-safe filenames."""
import re
import shutil
import threading
from pathlib import Path

_import_lock = threading.Lock()


def _same_contents(source, target):
    if source.stat().st_size != target.stat().st_size:
        return False
    with source.open('rb') as left, target.open('rb') as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
        return not right.read(1)

def open_import_file(directory, filename):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(filename).replace('\\', '/').rsplit('/', 1)[-1])
    path = directory / name
    duplicate = 0
    while True:
        try:
            return path, path.open('xb')
        except FileExistsError:
            duplicate += 1
            path = directory / f'{Path(name).stem} ({duplicate}){Path(name).suffix}'


def persist_gallery_import(source, directory, *, move=False):
    source = Path(source)
    # Already durable: do not duplicate media selected from the output directory.
    if source.resolve().parent == Path(directory).resolve():
        return str(source.resolve()), True
    # Compare and publish together so simultaneous imports see complete files.
    with _import_lock:
        directory = Path(directory).resolve()
        name = source.name
        candidate = directory / name
        duplicate = 0
        while candidate.exists():
            if _same_contents(source, candidate):
                return str(candidate), True
            duplicate += 1
            candidate = directory / f'{source.stem} ({duplicate}){source.suffix}'
        path, writer = open_import_file(directory, name)
        try:
            if move:
                writer.close()
                source.replace(path)
            else:
                with writer, source.open('rb') as reader:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                shutil.copystat(source, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    # Gradio's content-addressed cache can still be referenced by another page.
    # Its lifetime belongs to Gradio; the workspace now owns the durable copy.
    return str(path), False
