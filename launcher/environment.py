"""Drive the repository's own setup.py from the launcher.

Everything about creating environments, picking a CUDA/ROCm profile and
installing acceleration kernels already lives in setup.py. The launcher only
calls it and mirrors run.bat's rules for resolving the active environment.
"""

import logging
import os

from . import process

LOGGER = logging.getLogger(__name__)

SETUP_SCRIPT = "setup.py"
MAIN_SCRIPT = "wgp.py"
ENVS_FILE = "envs.json"


class EnvironmentError_(RuntimeError):
    """Raised when the WanGP environment cannot be resolved or built."""


def is_wangp_checkout(root):
    return all(os.path.isfile(os.path.join(root, name)) for name in (SETUP_SCRIPT, MAIN_SCRIPT))


def has_environment(root):
    return os.path.isfile(os.path.join(root, ENVS_FILE))


def get_env_info(root, python_exe):
    """Ask setup.py which environment is active.

    Mirrors run.bat: setup.py prints ``ENV_INFO|<type>|<path>`` and exits
    non-zero when nothing is configured yet.
    """
    code, out = process.run_quiet([python_exe, SETUP_SCRIPT, "get_env_info"], cwd=root, timeout=120)
    if code != 0:
        return None
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0] == "ENV_INFO":
            env_type, env_path = parts[1], parts[2]
            LOGGER.info("Active environment: type=%s path=%s", env_type, env_path)
            return env_type, env_path
    return None


def env_python(root, env_type, env_path, fallback):
    """Interpreter for an environment, matching setup.py's ENV_TEMPLATES."""
    if env_type == "none" or not env_path:
        return fallback
    absolute = env_path if os.path.isabs(env_path) else os.path.normpath(os.path.join(root, env_path))
    if env_type == "conda":
        candidate = os.path.join(absolute, "python.exe")
    else:  # venv and uv both use the standard Scripts layout on Windows
        candidate = os.path.join(absolute, "Scripts", "python.exe")
    if os.path.isfile(candidate):
        return candidate
    raise EnvironmentError_(
        f"The active environment '{env_path}' has no interpreter at {candidate}.\n"
        "Use 'Repair / reinstall' to rebuild it."
    )


def resolve_runtime_python(root, system_python):
    """Return the interpreter that should run wgp.py, or None if not installed."""
    info = get_env_info(root, system_python)
    if info is None:
        return None
    return env_python(root, info[0], info[1], system_python)


def install_auto(root, python_exe, env_type, on_line, env=None, cancelled=None):
    """Run setup.py's one-click install: detects the GPU and builds the env."""
    on_line(f"[*] Building the '{env_type}' environment with setup.py --auto")
    on_line("[*] This downloads PyTorch and the acceleration kernels; expect a long first run.")
    code = process.stream(
        [python_exe, SETUP_SCRIPT, "install", "--env", env_type, "--auto"],
        on_line=on_line,
        cwd=root,
        env=env or process.child_env(),
        cancelled=cancelled,
    )
    if code == -1:
        raise EnvironmentError_("Installation cancelled.")
    if code != 0:
        raise EnvironmentError_(
            f"setup.py install failed with exit code {code}. See the log for the failing command."
        )
    on_line("[*] Environment ready.")


def bootstrap_git_repo(root, git_exe, repo_url, branch, on_line, env=None):
    """Attach the installed tree to its git remote so updates can work.

    The installer ships the source without a .git directory. setup.py's own
    repair path hardcodes the upstream repository, so we wire up this fork's
    remote ourselves before handing over to setup.py update.
    """
    if os.path.isdir(os.path.join(root, ".git")):
        return True
    if not git_exe:
        on_line("[!] Git is not available, so the code cannot be updated.")
        return False

    env = env or process.child_env()
    on_line(f"[*] Linking this installation to {repo_url} ({branch}) ...")
    steps = [
        [git_exe, "init"],
        [git_exe, "remote", "add", "origin", repo_url],
        [git_exe, "fetch", "--depth", "1", "origin", branch],
        [git_exe, "reset", "--mixed", f"origin/{branch}"],
        [git_exe, "checkout", "-B", branch],
        [git_exe, "branch", f"--set-upstream-to=origin/{branch}", branch],
    ]
    for step in steps:
        code = process.stream(step, on_line=on_line, cwd=root, env=env)
        if code != 0 and step[1] not in ("remote", "branch"):
            on_line(f"[!] '{' '.join(step[1:])}' failed with exit code {code}.")
            return False
    on_line("[*] Repository linked.")
    return True


def update(root, python_exe, on_line, env=None, cancelled=None):
    """Run setup.py update: git pull plus a requirements refresh when needed."""
    code = process.stream(
        [python_exe, SETUP_SCRIPT, "update"],
        on_line=on_line,
        cwd=root,
        env=env or process.child_env(),
        cancelled=cancelled,
    )
    if code == -1:
        raise EnvironmentError_("Update cancelled.")
    if code != 0:
        raise EnvironmentError_(f"setup.py update failed with exit code {code}.")
    on_line("[*] Update finished.")


def open_manage_console(root, python_exe, env=None):
    """Hand the user setup.py's interactive environment manager in a console."""
    return process.popen(
        ["cmd", "/k", f'"{python_exe}" {SETUP_SCRIPT} manage'],
        cwd=root,
        env=env or process.child_env(),
        visible_console=True,
        capture=False,
    )


def open_upgrade_console(root, python_exe, env=None):
    """setup.py upgrade is a menu-driven flow; it belongs in a real console."""
    return process.popen(
        ["cmd", "/k", f'"{python_exe}" {SETUP_SCRIPT} upgrade'],
        cwd=root,
        env=env or process.child_env(),
        visible_console=True,
        capture=False,
    )


def read_extra_args(root):
    """Honour scripts/args.txt exactly like run.bat does, minus browser flags.

    ``--open-browser`` would pop an external browser next to our own window,
    and the server host/port are ours to decide, so those are filtered out.
    """
    path = os.path.join(root, "scripts", "args.txt")
    if not os.path.isfile(path):
        return []

    tokens = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                tokens.extend(line.split())
    except OSError as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return []

    dropped_flags = {"--open-browser", "--share", "--listen"}
    dropped_options = {"--server-port", "--server-name"}
    cleaned = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in dropped_flags:
            LOGGER.info("Ignoring %s from args.txt: the launcher owns the window", token)
            continue
        if token in dropped_options:
            LOGGER.info("Ignoring %s from args.txt: the launcher owns the server address", token)
            skip_next = True
            continue
        if token.split("=", 1)[0] in dropped_options:
            continue
        cleaned.append(token)
    return cleaned
