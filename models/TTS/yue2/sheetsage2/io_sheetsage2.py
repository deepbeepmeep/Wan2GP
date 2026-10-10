import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _target_path(path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _flush_path(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _replacement_mode(target: Path) -> int:
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mask = os.umask(0)
        os.umask(mask)
        return 0o666 & ~mask


@contextmanager
def atomic_output_path(path):
    target = _target_path(path)
    mode = _replacement_mode(target)
    fd, tmp_name = tempfile.mkstemp(
        # Keep the temporary basename independent of the final basename.  Some
        # annotation IDs are already close to NAME_MAX; repeating the target
        # name here made an otherwise valid final path fail with ENAMETOOLONG.
        prefix=".tmp_",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield str(tmp_path)
        os.chmod(tmp_path, mode)
        _flush_path(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path, text: str, encoding: str = "utf-8", newline: str = "\n") -> str:
    with atomic_output_path(path) as tmp_path:
        with open(tmp_path, "w", encoding=encoding, newline=newline) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    return str(path)


def atomic_write_pretty_midi(midi, path) -> str:
    with atomic_output_path(path) as tmp_path:
        midi.write(tmp_path)
    return str(path)
