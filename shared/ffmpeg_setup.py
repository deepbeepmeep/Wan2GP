import os
import sys
import shutil
import tarfile
import tempfile
import typing
from pathlib import Path

import requests
from tqdm import tqdm
import zipfile


__all__ = ["download_ffmpeg"]


def download_ffmpeg(bin_directory: typing.Optional[typing.Union[str, Path]] = None):
    """Ensure ffmpeg/ffprobe binaries (and their shared libs) are downloaded and on PATH."""
    required_binaries = ["ffmpeg", "ffprobe"]
    if os.name == "nt":
        required_binaries.append("ffplay")

    if bin_directory is None:
        repo_root = Path(__file__).resolve().parents[1]
        bin_dir = repo_root / "ffmpeg_bins"
    else:
        bin_dir = Path(bin_directory)

    bin_dir.mkdir(parents=True, exist_ok=True)
    repo_root = bin_dir.parent

    def _ensure_bin_dir_on_path():
        current_path = os.environ.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        dirs_to_add = []
        # Add ffmpeg_bins and repo root to PATH
        for d in [bin_dir, repo_root]:
            if str(d) not in path_parts:
                dirs_to_add.append(str(d))
        if dirs_to_add:
            os.environ["PATH"] = os.pathsep.join(dirs_to_add + path_parts)

    def _ensure_library_path():
        if os.name == "nt":
            return
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [p for p in current_ld.split(os.pathsep) if p]
        if str(bin_dir) not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(bin_dir)] + ld_parts) if current_ld else str(bin_dir)

    _ensure_bin_dir_on_path()
    _ensure_library_path()

    def _candidate_name(name: str) -> str:
        if os.name == "nt" and not name.endswith(".exe"):
            return f"{name}.exe"
        return name

    def _resolve_path(name: str) -> typing.Optional[Path]:
        # Check ffmpeg_bins folder first
        candidate = bin_dir / _candidate_name(name)
        if candidate.exists():
            return candidate
        # Check repo root folder (some users put ffmpeg there)
        repo_root = bin_dir.parent
        candidate_root = repo_root / _candidate_name(name)
        if candidate_root.exists():
            return candidate_root
        # Fall back to system PATH
        resolved = shutil.which(name)
        return Path(resolved) if resolved else None

    def _binary_exists(name: str) -> bool:
        return _resolve_path(name) is not None

    def _libs_present() -> bool:
        if os.name == "nt":
            return True
        ffmpeg_path = _resolve_path("ffmpeg")
        if ffmpeg_path and ffmpeg_path.parent != bin_dir:
            # System-installed FFmpeg should already have its shared libs available.
            return True
        return any(bin_dir.glob("libavdevice.so*"))

    def _set_env_vars():
        ffmpeg_path = _resolve_path("ffmpeg")
        ffprobe_path = _resolve_path("ffprobe")
        ffplay_path = _resolve_path("ffplay") if "ffplay" in required_binaries else None
        if ffmpeg_path:
            os.environ["FFMPEG_BINARY"] = str(ffmpeg_path)
        if ffprobe_path:
            os.environ["FFPROBE_BINARY"] = str(ffprobe_path)
        if ffplay_path:
            os.environ["FFPLAY_BINARY"] = str(ffplay_path)

    missing = [binary for binary in required_binaries if not _binary_exists(binary)]
    libs_ok = _libs_present()
    if not missing and libs_ok:
        _set_env_vars()
        return

    def _download_file(url: str, destination: Path):
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            with open(destination, "wb") as file_handle, tqdm(
                total=total if total else None,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {destination.name}"
            ) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    progress.update(len(chunk))

    def _download_windows_build():
        exes = [_candidate_name(name) for name in required_binaries]
        api_url = "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest"
        response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"}, timeout=30)
        response.raise_for_status()
        assets = response.json().get("assets", [])
        zip_asset = next((asset for asset in assets if asset.get("name", "").endswith("essentials_build.zip")), None)
        if not zip_asset:
            raise RuntimeError("Unable to locate FFmpeg essentials build for Windows.")
        zip_path = bin_dir / zip_asset["name"]
        _download_file(zip_asset["browser_download_url"], zip_path)

        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    normalized = member.replace("\\", "/")
                    if "/bin/" not in normalized:
                        continue
                    base_name = os.path.basename(normalized)
                    if base_name not in exes and not base_name.lower().endswith(".dll"):
                        continue
                    destination = bin_dir / base_name
                    with archive.open(member) as source, open(destination, "wb") as target:
                        shutil.copyfileobj(source, target)
                    destination.chmod(0o755)
        finally:
            try:
                zip_path.unlink(missing_ok=True)
            except TypeError:
                if zip_path.exists():
                    zip_path.unlink()

    def _download_posix_build():
        api_url = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
        response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"}, timeout=30)
        response.raise_for_status()
        assets = response.json().get("assets", [])
        if sys.platform.startswith("linux"):
            keywords = ["linux64", "gpl"]
        elif sys.platform == "darwin":
            keywords = ["macos64", "gpl"]
        else:
            raise RuntimeError("Unsupported platform for automatic FFmpeg download.")

        tar_asset = next(
            (
                asset for asset in assets
                if asset.get("name", "").endswith(".tar.xz") and all(k in asset.get("name", "") for k in keywords)
            ),
            None
        )
        if not tar_asset:
            raise RuntimeError("Unable to locate a suitable FFmpeg build for this platform.")

        tar_path = bin_dir / tar_asset["name"]
        _download_file(tar_asset["browser_download_url"], tar_path)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                with tarfile.open(tar_path, "r:xz") as archive:
                    archive.extractall(tmp_path)

                build_root = None
                for candidate in tmp_path.iterdir():
                    if (candidate / "bin").exists():
                        build_root = candidate
                        break

                if build_root is None:
                    raise RuntimeError("Unable to locate FFmpeg bin directory in downloaded archive.")

                bin_source = build_root / "bin"
                lib_source = build_root / "lib"

                for binary in required_binaries:
                    source_file = bin_source / binary
                    if not source_file.exists():
                        continue
                    destination = bin_dir / binary
                    shutil.copy2(source_file, destination)
                    destination.chmod(0o755)

                if lib_source.exists():
                    for lib_file in lib_source.rglob("*.so*"):
                        destination = bin_dir / lib_file.name
                        shutil.copy2(lib_file, destination)
                        destination.chmod(0o755)
        finally:
            try:
                tar_path.unlink(missing_ok=True)
            except TypeError:
                if tar_path.exists():
                    tar_path.unlink()

    try:
        if os.name == "nt":
            _download_windows_build()
        else:
            _download_posix_build()
    except Exception as exc:
        print(f"Failed to download FFmpeg binaries automatically: {exc}")
        return

    if not all(_binary_exists(binary) for binary in required_binaries):
        print("FFmpeg binaries are still missing after download; please install them manually.")
        return

    _ensure_bin_dir_on_path()
    _ensure_library_path()
    _set_env_vars()
