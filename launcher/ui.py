"""The launcher window: a setup/progress page that becomes the WanGP web UI."""

import logging
import os
import threading
import webbrowser

import webview

from . import config, environment, prereqs, process, server

LOGGER = logging.getLogger(__name__)

MAX_CONSOLE_LINES = 500


def _load_page():
    path = os.path.join(config.bundle_dir(), "web", "setup.html")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class LauncherApi:
    """Bridge exposed to the setup page as ``window.pywebview.api``."""

    def __init__(self, app):
        self._app = app

    def get_state(self):
        return self._app.snapshot()

    def retry(self):
        self._app.retry()
        return True

    def open_logs(self):
        process.open_in_explorer(config.logs_dir())
        return True

    def quit(self):
        self._app.shutdown()
        return True


class LauncherApp:
    """Bootstraps prerequisites, the environment and the server, then shows it."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = config.install_root()
        self.window = None
        self.server = None

        self._lock = threading.Lock()
        self._lines = []
        self._title = "Starting…"
        self._detail = ""
        self._error = ""
        self._done = False
        self._can_retry = False
        self._busy = threading.Event()
        self._shutting_down = threading.Event()
        self._system_python = None
        self._runtime_python = None
        self._git_exe = None

    # ----------------------------------------------------------------- state

    def snapshot(self):
        with self._lock:
            return {
                "version": self.cfg.get("app_version", ""),
                "title": self._title,
                "detail": self._detail,
                "lines": list(self._lines),
                "error": self._error,
                "done": self._done,
                "can_retry": self._can_retry,
            }

    def log(self, line):
        LOGGER.info("%s", line)
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > MAX_CONSOLE_LINES:
                del self._lines[: len(self._lines) - MAX_CONSOLE_LINES]

    def phase(self, title, detail=""):
        with self._lock:
            self._title = title
            self._detail = detail
            self._error = ""
            self._done = False
            self._can_retry = False

    def fail(self, message):
        LOGGER.error("%s", message)
        with self._lock:
            self._error = message
            self._title = "Something went wrong"
            self._detail = "Fix the problem above, then retry."
            self._can_retry = True
            self._done = False

    def finish(self, title, detail=""):
        with self._lock:
            self._title = title
            self._detail = detail
            self._done = True
            self._can_retry = False

    # ------------------------------------------------------------- lifecycle

    def show_setup_page(self):
        if self.window is not None:
            self.window.load_html(_load_page())

    def boot(self, window):
        self.window = window
        self._run_guarded(self._bootstrap)

    def retry(self):
        if self._busy.is_set():
            return
        self._run_guarded(self._bootstrap)

    def _run_guarded(self, target):
        def wrapper():
            self._busy.set()
            try:
                target()
            except Exception as exc:  # surfaced in the page, detailed in the log
                LOGGER.exception("Launcher step failed")
                self.fail(str(exc))
            finally:
                self._busy.clear()

        threading.Thread(target=wrapper, name="wangp-launcher", daemon=True).start()

    def _bootstrap(self):
        with self._lock:
            self._lines = []
        self.phase("Checking the installation…")

        if not environment.is_wangp_checkout(self.root):
            raise environment.EnvironmentError_(
                f"No WanGP installation found in:\n  {self.root}\n\n"
                "wgp.py and setup.py should sit next to the launcher. Reinstall WanGP."
            )
        self.log(f"[*] Installation folder: {self.root}")

        self._ensure_python()
        self._ensure_git()
        self._ensure_environment()
        self._start_server()

    def _ensure_python(self):
        series = self.cfg["python"].get("series", "3.11")
        self.phase(f"Looking for Python {series}…")
        found = prereqs.find_python(series)
        if found:
            self.log(f"[*] Python {series} found: {found}")
            self._system_python = found
            return

        stray = prereqs.python_present_but_unlaunchable(series)
        if stray:
            self.log(f"[!] Found {stray} but the 'py' launcher does not know about it.")
        self.phase(
            f"Installing Python {series}…",
            "Required to build the WanGP environment. Nothing else on your system is changed.",
        )
        self._system_python = prereqs.install_python(self.cfg, on_line=self.log)

    def _ensure_git(self):
        self.phase("Looking for Git…")
        self._git_exe = prereqs.find_git()
        if self._git_exe:
            self.log(f"[*] Git found: {self._git_exe}")
            return
        if not self.cfg.get("git", {}).get("portable_url"):
            self.log("[!] Git is missing. WanGP will run, but updates will be unavailable.")
            return
        try:
            self.phase("Installing a private copy of Git…", "Used only to update WanGP later.")
            self._git_exe = prereqs.install_portable_git(self.cfg, on_line=self.log)
        except prereqs.PrereqError as exc:
            # Git is only needed for updates, so this must never block a launch.
            self.log(f"[!] Could not install Git: {exc}")
            self.log("[!] WanGP will still run; install Git manually to enable updates.")

    def _env(self):
        return prereqs.env_with_git(self._git_exe)

    def _ensure_environment(self):
        self.phase("Checking the Python environment…")
        runtime = None
        if environment.has_environment(self.root):
            try:
                runtime = environment.resolve_runtime_python(self.root, self._system_python)
            except environment.EnvironmentError_ as exc:
                self.log(f"[!] {exc}")

        if runtime is None:
            self.phase(
                "First-time setup",
                "Detecting your GPU and installing PyTorch and the acceleration kernels. "
                "This takes a while and needs several GB of disk space.",
            )
            environment.install_auto(
                self.root,
                self._system_python,
                self.cfg.get("env_type", "venv"),
                on_line=self.log,
                env=self._env(),
                cancelled=self._shutting_down.is_set,
            )
            runtime = environment.resolve_runtime_python(self.root, self._system_python)
            if runtime is None:
                raise environment.EnvironmentError_(
                    "Setup finished but no active environment was registered.\n"
                    "Open 'Manage environments…' to select one."
                )
            config.write_state(installed=True)

        self.log(f"[*] Using interpreter: {runtime}")
        self._runtime_python = runtime

    def _start_server(self):
        self.phase("Starting WanGP…", "Loading the model list; the interface opens by itself.")
        self.server = server.WanGPServer(self.root, self._runtime_python, self.cfg, on_line=self.log)
        self.server.start()
        if not self.server.wait_until_ready(cancelled=self._shutting_down.is_set):
            if self._shutting_down.is_set():
                return
            raise environment.EnvironmentError_(
                "WanGP started but its web interface never answered.\n"
                "The last lines above usually say why."
            )
        self.finish("WanGP is ready", "Opening the interface…")
        self.window.load_url(self.server.url)

    # ----------------------------------------------------------- menu actions

    def _requires_python(self):
        """Maintenance needs the interpreter the bootstrap resolved."""
        if self._system_python:
            return True
        self.show_setup_page()
        self.fail(
            "WanGP has not finished its first-time setup yet, so maintenance is "
            "unavailable.\nUse Retry to finish the setup first."
        )
        return False

    def _with_stopped_server(self, title, detail, action):
        """Run maintenance with the server down, then bring it back up."""
        if self._busy.is_set() or not self._requires_python():
            return

        def task():
            self.show_setup_page()
            self.phase(title, detail)
            with self._lock:
                self._lines = []
            if self.server is not None:
                self.server.stop()
                self.server = None
            action()
            self._start_server()

        self._run_guarded(task)

    def action_update(self):
        def run():
            environment.bootstrap_git_repo(
                self.root,
                self._git_exe,
                self.cfg.get("repo_url", ""),
                self.cfg.get("repo_branch", "main"),
                on_line=self.log,
                env=self._env(),
            )
            environment.update(self.root, self._system_python, on_line=self.log, env=self._env())

        self._with_stopped_server(
            "Updating WanGP…",
            "Pulling the latest code and refreshing requirements.",
            run,
        )

    def action_repair(self):
        def run():
            environment.install_auto(
                self.root,
                self._system_python,
                self.cfg.get("env_type", "venv"),
                on_line=self.log,
                env=self._env(),
                cancelled=self._shutting_down.is_set,
            )
            self._runtime_python = environment.resolve_runtime_python(self.root, self._system_python)
            if self._runtime_python is None:
                raise environment.EnvironmentError_("The rebuilt environment could not be resolved.")

        self._with_stopped_server(
            "Rebuilding the environment…",
            "The existing environment is replaced. Models, LoRAs and outputs are untouched.",
            run,
        )

    def action_restart_server(self):
        def run():
            self.log("[*] Restarting WanGP …")

        self._with_stopped_server("Restarting WanGP…", "", run)

    def action_manage_envs(self):
        if self._requires_python():
            environment.open_manage_console(self.root, self._system_python, env=self._env())

    def action_upgrade_components(self):
        if self._requires_python():
            environment.open_upgrade_console(self.root, self._system_python, env=self._env())

    def action_open_install_folder(self):
        process.open_in_explorer(self.root)

    def action_open_outputs(self):
        outputs = os.path.join(self.root, "outputs")
        process.open_in_explorer(outputs if os.path.isdir(outputs) else self.root)

    def action_open_logs(self):
        process.open_in_explorer(config.logs_dir())

    def action_open_in_browser(self):
        if self.server is not None and self.server.is_running():
            webbrowser.open(self.server.url)

    def shutdown(self):
        self._shutting_down.set()
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:  # the window may already be gone
                LOGGER.debug("Window already destroyed", exc_info=True)


def _build_menu(app):
    """Native menu bar. Older pywebview builds lack this API, so it is optional."""
    try:
        from webview.menu import Menu, MenuAction, MenuSeparator
    except ImportError:
        LOGGER.info("This pywebview build has no menu support; running without a menu bar.")
        return None

    return [
        Menu(
            "WanGP",
            [
                MenuAction("Restart WanGP", app.action_restart_server),
                MenuAction("Open in web browser", app.action_open_in_browser),
                MenuSeparator(),
                MenuAction("Quit", app.shutdown),
            ],
        ),
        Menu(
            "Maintenance",
            [
                MenuAction("Update WanGP…", app.action_update),
                MenuAction("Repair / reinstall environment…", app.action_repair),
                MenuSeparator(),
                MenuAction("Manage environments…", app.action_manage_envs),
                MenuAction("Upgrade Torch / Triton / Sage…", app.action_upgrade_components),
            ],
        ),
        Menu(
            "Folders",
            [
                MenuAction("Installation folder", app.action_open_install_folder),
                MenuAction("Outputs", app.action_open_outputs),
                MenuAction("Logs", app.action_open_logs),
            ],
        ),
    ]


def run(cfg):
    app = LauncherApp(cfg)
    window_cfg = cfg.get("window", {})
    window = webview.create_window(
        cfg.get("app_name", "WanGP"),
        html=_load_page(),
        js_api=LauncherApi(app),
        width=window_cfg.get("width", 1450),
        height=window_cfg.get("height", 940),
        min_size=(window_cfg.get("min_width", 900), window_cfg.get("min_height", 620)),
        text_select=True,
    )
    window.events.closed += app.shutdown

    start_kwargs = {"gui": "edgechromium", "private_mode": False, "storage_path": config.data_dir()}
    menu = _build_menu(app)
    if menu:
        start_kwargs["menu"] = menu

    try:
        webview.start(app.boot, window, **start_kwargs)
    except TypeError:
        # Fall back for pywebview builds that reject one of the newer options.
        LOGGER.warning("Retrying webview.start with reduced options", exc_info=True)
        webview.start(app.boot, window, gui="edgechromium")
    finally:
        app.shutdown()
