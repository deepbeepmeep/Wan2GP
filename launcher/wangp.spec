# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the WanGP Windows launcher.

Built as a one-folder bundle placed in ``launcher-bin`` inside the WanGP
installation directory. One-folder starts faster than one-file and avoids the
temp-extraction pattern that antivirus products tend to flag.
"""

import os

from PyInstaller.utils.hooks import collect_all

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
REPO_ROOT = os.path.dirname(SPEC_DIR)
ICON = os.path.join(SPEC_DIR, "assets", "wangp.ico")

# pywebview ships the WebView2 interop assemblies as package data; without
# collect_all the frozen build starts and then fails to create a window.
webview_datas, webview_binaries, webview_hidden = collect_all("webview")

a = Analysis(
    [os.path.join(SPEC_DIR, "main.py")],
    pathex=[REPO_ROOT],
    binaries=webview_binaries,
    datas=webview_datas
    + [
        (os.path.join(SPEC_DIR, "web"), "web"),
        (os.path.join(SPEC_DIR, "launcher_config.json"), "."),
    ],
    hiddenimports=webview_hidden
    + [
        "webview.platforms.edgechromium",
        "clr_loader",
        "pythonnet",
    ],
    hookspath=[],
    runtime_hooks=[],
    # WanGP's own heavy stack is never imported by the launcher: it runs in the
    # separate environment setup.py builds.
    excludes=["torch", "gradio", "numpy", "PIL", "tkinter", "matplotlib", "scipy"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WanGP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON if os.path.isfile(ICON) else None,
    version=os.path.join(SPEC_DIR, "assets", "version_info.txt")
    if os.path.isfile(os.path.join(SPEC_DIR, "assets", "version_info.txt"))
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="launcher-bin",
)
