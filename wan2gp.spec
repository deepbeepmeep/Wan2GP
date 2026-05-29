# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Wan2GP API server (macOS single-file executable).
"""
import os
import sys
from pathlib import Path

project_root = os.path.abspath(".")

# Bundled package directories (as data trees so they're available at runtime)
PACKAGE_DIRS = ["wgp_fastapi", "shared"]
DATA_DIRS = ["defaults", "profiles"]
DATA_FILES = ["plugins.json"]

datas = []
for d in PACKAGE_DIRS + DATA_DIRS:
    src = os.path.join(project_root, d)
    if os.path.exists(src):
        datas.append((src, d))
for f in DATA_FILES:
    src = os.path.join(project_root, f)
    if os.path.exists(src):
        datas.append((src, "."))

a = Analysis(
    ["run_app.py"],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Web framework stack
        "uvicorn",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.middleware.wsgi",
        "fastapi",
        "pydantic",
        "pydantic.dataclasses",
        "starlette",
        "starlette.applications",
        "starlette.routing",
        "starlette.middleware",
        "starlette.middleware.cors",
        "starlette.responses",
        "starlette.requests",
        "starlette.datastructures",
        "starlette.concurrency",
        "starlette.convertors",
        # HTTP handling
        "multipart",
        "multipart.multipart",
        "h11",
        "httptools",
        # Image processing (imported by upscaler at module level)
        "PIL",
        "PIL._imaging",
        "PIL.Image",
        "PIL.ImageFile",
        "PIL.JpegImagePlugin",
        "PIL.PngImagePlugin",
        "PIL.GifImagePlugin",
        "PIL.BmpImagePlugin",
        "PIL.TiffImagePlugin",
        "cv2",
        "cv2.gapi",
        # Core numeric
        "numpy",
        "numpy.core._multiarray_umath",
        # pillow-heif support
        "pillow_heif",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "tkinter.ttk",
        "test", "unittest",
        "distutils", "setuptools", "pip", "pydoc", "doctest",
        "matplotlib.tests", "numpy.testing", "PIL.SelfTest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Wan2GP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=["*.bz2", "*.lzma"],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
