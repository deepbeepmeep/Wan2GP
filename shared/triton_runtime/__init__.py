# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Use bundled CUDA/HIP helpers for the validated Windows Triton stacks only."""
import sys


def install_prebuilt_helper():
    # CUDA uses CPython 3.11; HIP uses CPython 3.12. Other stacks retain stock Triton.
    if sys.platform != "win32" or sys.implementation.name != "cpython" or sys.version_info[:2] not in ((3, 11), (3, 12)):
        return False
    import struct
    if struct.calcsize("P") != 8:
        return False
    from importlib.metadata import PackageNotFoundError, version
    try:
        release = version("triton-windows")
    except PackageNotFoundError:
        return False
    if release not in ("3.7.0.post26", "3.7.1.post27"):
        return False

    import hashlib
    import importlib
    import importlib.util
    import json
    from pathlib import Path
    from threading import Lock
    hip = sys.version_info[:2] == (3, 12)
    directory = Path(__file__).parent
    manifest_name = "manifest_hip.json" if hip else "manifest.json"
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    if hip:
        try:
            if version("torch") != manifest["torch_version"] or version("rocm-sdk-core") != manifest["rocm_version"]:
                return False
        except PackageNotFoundError:
            return False
    import triton
    driver = importlib.import_module("triton.backends.amd.driver" if hip else "triton.backends.nvidia.driver")
    helper_name = "hip_utils" if hip else "cuda_utils"
    label = "HIP" if hip else "CUDA"

    if triton.__version__ != release.split(".post")[0]:
        return False
    original = driver.compile_module_from_src
    if getattr(original, "_wangp_prebuilt", False):
        return True
    # An embedding application may already have initialized Triton. Do not mix
    # extension types with kernel arguments created by its existing helper.
    if hasattr(driver.HIPUtils if hip else driver.CudaUtils, "instance"):
        return False

    build = manifest["builds"][release]
    binary = directory / manifest["binary"]
    loaded = None
    lock = Lock()

    def compile_or_load(src, name, **kwargs):
        nonlocal loaded
        if name != helper_name:
            return original(src, name, **kwargs)
        with lock:
            source = Path(driver.__file__).with_name("driver.c").read_text(encoding="utf-8") if hip else src
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != build["driver_source_sha256"]:
                raise RuntimeError(f"WanGP's prebuilt {label} helper does not match Triton's driver.c. Reinstall the official triton-windows wheel.")
            if hip:
                runtime_path = driver._get_path_to_hip_runtime_dylib().replace("\\", "\\\\")
                if src != source.replace("/*py_libhip_search_path*/", runtime_path, 1):
                    raise RuntimeError("Unexpected Triton HIP helper source or runtime path.")
            if loaded is None:
                driver_text = Path(driver.__file__).read_text(encoding="utf-8")
                python_hash = build["driver_python_sha256"] if hip else manifest["driver_python_sha256"]
                if hashlib.sha256(driver_text.encode("utf-8")).hexdigest() != python_hash:
                    raise RuntimeError(f"WanGP's prebuilt {label} helper does not match Triton's driver.py. Reinstall the official triton-windows wheel.")
                try:
                    data = binary.read_bytes()
                except OSError as error:
                    raise RuntimeError(f"WanGP's prebuilt {label} helper is unavailable: {binary}. Restore this file from the WanGP repository and check antivirus history.") from error
                if hashlib.sha256(data).hexdigest() != manifest["binary_sha256"]:
                    raise RuntimeError(f"WanGP's prebuilt {label} helper checksum differs. Restore the helper from the WanGP repository.")
                spec = importlib.util.spec_from_file_location(helper_name, binary)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded = module
                print(f"[Triton] Using WanGP's Prebuilt {label} Helper ({release})")
            return loaded

    compile_or_load._wangp_prebuilt = True
    driver.compile_module_from_src = compile_or_load
    return True
