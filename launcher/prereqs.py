"""Detect and, when missing, install what WanGP needs before setup.py can run.

setup.py builds its virtual environment with ``py -3.11 -m venv`` on Windows,
so the real prerequisite is the Python launcher *plus* a 3.11 interpreter --
not merely "some Python". Git is only needed to update an existing install, so
a missing Git is reported but never fatal.
"""

import hashlib
import logging
import os
import shutil
import urllib.request

from . import config, process

LOGGER = logging.getLogger(__name__)

_DOWNLOAD_CHUNK = 256 * 1024


class PrereqError(RuntimeError):
    pass


def _report(on_line, message):
    LOGGER.info(message)
    if on_line is not None:
        on_line(message)


def download(url, destination, on_line=None, expected_sha256=""):
    _report(on_line, f"[*] Downloading {os.path.basename(destination)} ...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = destination + ".part"
    digest = hashlib.sha256()
    last_percent = -5
    request = urllib.request.Request(url, headers={"User-Agent": "WanGP-Launcher"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URLs
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                read += len(chunk)
                if total:
                    percent = int(read * 100 / total)
                    if percent >= last_percent + 5:
                        last_percent = percent
                        _report(on_line, f"    {percent}%  ({read // 1048576} / {total // 1048576} MB)")

    if expected_sha256:
        actual = digest.hexdigest()
        if actual.lower() != expected_sha256.lower():
            os.remove(partial)
            raise PrereqError(
                f"Checksum mismatch for {url}\n  expected {expected_sha256}\n  got      {actual}"
            )
        _report(on_line, "    checksum verified")

    os.replace(partial, destination)
    return destination


def _python_launcher_ok(series):
    code, out = process.run_quiet(["py", f"-{series}", "-c", "import sys; print(sys.executable)"])
    if code == 0 and out:
        return out.splitlines()[-1].strip()
    return None


def find_python(series="3.11"):
    """Return the path of the interpreter ``py -<series>`` resolves to."""
    found = _python_launcher_ok(series)
    if found:
        LOGGER.info("Python %s available via the py launcher: %s", series, found)
    return found


def python_present_but_unlaunchable(series="3.11"):
    """A 3.11 install that ``py`` cannot see -- the launcher itself is missing."""
    compact = series.replace(".", "")
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", f"Python{compact}", "python.exe"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), f"Python{compact}", "python.exe"),
        os.path.join("C:\\", f"Python{compact}", "python.exe"),
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def install_python(cfg, on_line=None):
    """Install Python for the current user, including the py launcher."""
    spec = cfg["python"]
    url = spec.get("installer_url")
    if not url:
        raise PrereqError("No Python installer URL is configured.")

    installer = os.path.join(config.data_dir(), "downloads", os.path.basename(url))
    if not os.path.isfile(installer):
        download(url, installer, on_line=on_line, expected_sha256=spec.get("installer_sha256", ""))
    else:
        _report(on_line, f"[*] Reusing cached installer: {installer}")

    _report(on_line, f"[*] Installing {spec.get('label', 'Python')} (per-user, no admin rights needed) ...")
    _report(on_line, "    the Python setup window will show its own progress bar")
    code = process.stream(
        [
            installer,
            "/passive",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_launcher=1",
            "Include_pip=1",
            "Include_test=0",
            "AssociateFiles=0",
            "Shortcuts=0",
        ],
        on_line=on_line if on_line is not None else (lambda _line: None),
        env=process.child_env(),
    )

    # 0 = success, 3010 = success but a reboot is pending, 1602 = user cancelled.
    if code == 1602:
        raise PrereqError("Python installation was cancelled.")
    if code not in (0, 3010):
        raise PrereqError(f"The Python installer failed with exit code {code}.")

    found = find_python(spec.get("series", "3.11"))
    if not found:
        raise PrereqError(
            "Python was installed but the 'py' launcher still cannot find it.\n"
            "Sign out and back in (or reboot) and start WanGP again."
        )
    _report(on_line, f"[*] Python ready: {found}")
    return found


def portable_git_exe():
    path = os.path.join(config.tools_dir(), "git", "cmd", "git.exe")
    return path if os.path.isfile(path) else None


def find_git():
    """Return a usable git.exe: the system one first, then our portable copy."""
    system_git = shutil.which("git")
    if system_git:
        return system_git
    return portable_git_exe()


def install_portable_git(cfg, on_line=None):
    """Unpack Git for Windows into our tools folder -- self-contained, no admin."""
    spec = cfg.get("git", {})
    url = spec.get("portable_url")
    if not url:
        raise PrereqError("No portable Git URL is configured.")

    archive = os.path.join(config.data_dir(), "downloads", os.path.basename(url))
    if not os.path.isfile(archive):
        download(url, archive, on_line=on_line, expected_sha256=spec.get("portable_sha256", ""))

    target = os.path.join(config.tools_dir(), "git")
    _report(on_line, f"[*] Extracting {spec.get('label', 'portable Git')} to {target} ...")
    os.makedirs(target, exist_ok=True)
    # The PortableGit download is a 7-Zip self-extracting archive.
    code, out = process.run_quiet([archive, "-o" + target, "-y"], timeout=1800)
    if code != 0:
        raise PrereqError(f"Could not extract portable Git (exit {code}).\n{out}")

    git_exe = portable_git_exe()
    if not git_exe:
        raise PrereqError("Portable Git was extracted but git.exe is missing.")
    _report(on_line, f"[*] Git ready: {git_exe}")
    return git_exe


def env_with_git(git_exe, extra=None):
    """Child environment with the resolved git.exe reachable on PATH."""
    env = process.child_env(extra)
    if git_exe:
        git_dir = os.path.dirname(git_exe)
        env["PATH"] = git_dir + os.pathsep + env.get("PATH", "")
    return env
