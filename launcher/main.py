"""Entry point for the WanGP Windows launcher."""

import ctypes
import logging
import os
import sys

if __package__ in (None, ""):
    # Allow "python launcher/main.py" during development.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from launcher import config  # noqa: E402  (path bootstrap must come first)


def _enable_dpi_awareness():
    """Keep the WanGP UI crisp on scaled displays."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _fatal(message):
    logging.getLogger(__name__).error("%s", message)
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "WanGP", 0x10)
        except (AttributeError, OSError):
            pass
    else:
        print(message, file=sys.stderr)


def _selftest():
    """Verify a frozen build can find its resources and its window backend.

    Runs headless in CI, where there is no GPU and no interactive session, so
    it deliberately stops short of creating a window.
    """
    logger = logging.getLogger(__name__)
    checks = []

    cfg = config.load_config()
    checks.append(("config loaded", bool(cfg.get("app_name"))))
    checks.append(("version stamped", cfg.get("app_version", "") != ""))

    page = os.path.join(config.bundle_dir(), "web", "setup.html")
    checks.append(("setup page bundled", os.path.isfile(page)))

    try:
        import webview  # noqa: F401  (import is the check)
        from webview.platforms import edgechromium  # noqa: F401

        checks.append(("webview + edgechromium importable", True))
    except ImportError as exc:
        logger.error("webview import failed: %s", exc)
        checks.append(("webview + edgechromium importable", False))

    try:
        from launcher import environment, prereqs, process, server, ui  # noqa: F401

        checks.append(("launcher modules importable", True))
    except ImportError as exc:
        logger.error("launcher import failed: %s", exc)
        checks.append(("launcher modules importable", False))

    checks.append(("writable data dir", os.path.isdir(config.data_dir())))

    for name, passed in checks:
        logger.info("selftest %-38s %s", name, "OK" if passed else "FAILED")
    failures = [name for name, passed in checks if not passed]
    if failures:
        logger.error("selftest failures: %s", ", ".join(failures))
        return 1
    logger.info("selftest passed")
    return 0


def main():
    log_path = config.setup_logging()
    logger = logging.getLogger(__name__)
    cfg = config.load_config()
    logger.info(
        "%s launcher %s starting (root=%s, log=%s)",
        cfg.get("app_name", "WanGP"),
        cfg.get("app_version", "?"),
        config.install_root(),
        log_path,
    )
    if "--selftest" in sys.argv[1:]:
        return _selftest()

    _enable_dpi_awareness()

    try:
        # Imported here, not at module scope, so a missing WebView2 backend is
        # reported in a message box instead of a bare import traceback.
        from launcher import ui

        ui.run(cfg)
    except ImportError as exc:
        _fatal(
            "The launcher could not start its window component.\n\n"
            f"{exc}\n\n"
            "Reinstall WanGP, or install the Microsoft Edge WebView2 Runtime."
        )
        return 2
    except Exception as exc:  # last resort: never die without telling the user
        logger.exception("Launcher crashed")
        _fatal(f"WanGP failed to start:\n\n{exc}\n\nDetails: {log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
