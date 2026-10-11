# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Rebuilds the prebuilt mmgp VRAM allocator libraries after a change to vmm_alloc.cpp: python -m mmgp.allocator.build
On Windows it builds vmm_alloc_win_amd64.dll with MSVC (Visual Studio or Build Tools 2017+ with the C++ workload, found with vswhere),
then vmm_alloc_linux_x86_64.so in WSL when WSL is installed (g++ needed there; MMGP_WSL_PYTHON: the Python run there, python3 by default).
With PyTorch importable, it also builds the companion library that throws PyTorch's own out of memory error (oom_error.cpp) and the MMGP RAM
allocator with the RAM debug mode's hooks (ram_alloc.cpp), both linked with PyTorch's c10 library (on Linux only for a PyTorch with the C++11 ABI). On Linux it builds the .so (g++, C++17), the C++ standard
library linked statically so that the library only needs glibc 2.17+ (and libgcc_s, which every C++ program loads). No CUDA toolkit is needed: the driver is loaded at run time."""
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "vmm_alloc.cpp")
COMPANION = os.path.join(HERE, "oom_error.cpp")
RAM_ALLOC = os.path.join(HERE, "ram_alloc.cpp")


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _msvc_environment():
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        raise RuntimeError("Visual Studio (or the Build Tools) with the C++ workload is required: vswhere.exe not found.")
    root = subprocess.run([vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if not root:
        raise RuntimeError("No Visual Studio installation with the C++ x64 tools was found.")
    tools = sorted(glob.glob(os.path.join(root, "VC", "Tools", "MSVC", "*")))[-1]
    kits = os.path.join(program_files, "Windows Kits", "10")
    sdk = sorted(os.path.basename(p) for p in glob.glob(os.path.join(kits, "Lib", "10.*")))[-1]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([os.path.join(tools, "bin", "Hostx64", "x64"), os.path.join(kits, "bin", sdk, "x64"), env.get("PATH", "")])
    env["INCLUDE"] = os.pathsep.join([os.path.join(tools, "include")] + [os.path.join(kits, "Include", sdk, part) for part in ("ucrt", "shared", "um")])
    env["LIB"] = os.pathsep.join([os.path.join(tools, "lib", "x64")] + [os.path.join(kits, "Lib", sdk, part, "x64") for part in ("ucrt", "um")])
    return env, os.path.join(tools, "bin", "Hostx64", "x64", "cl.exe")  # Windows resolves the executable with the parent's PATH


def build_windows():
    env, cl = _msvc_environment()
    target = "vmm_alloc_win_amd64.dll"
    # /MD: PyTorch's C runtime, as std::function arguments of the CUDA graph hooks cross the boundary
    result = subprocess.run([cl, "/nologo", "/O2", "/EHs", "/LD", "/MD", "/std:c++17", SOURCE, f"/Fe{target}", "/link", "/NODEFAULTLIB:LIBCMT"], cwd=HERE, env=env)
    for leftover in glob.glob(os.path.join(HERE, "vmm_alloc*.obj")) + glob.glob(os.path.join(HERE, "vmm_alloc*.lib")) + glob.glob(os.path.join(HERE, "vmm_alloc*.exp")):
        os.remove(leftover)
    return result.returncode or build_windows_companion(env, cl), target


def build_windows_companion(env, cl):
    torch = _torch()
    if torch is None:
        print("PyTorch is not installed: the companion libraries (PyTorch's out of memory error, the RAM allocator) were not built")
        return 0
    torch_root, code = os.path.dirname(torch.__file__), 0
    for source, target in ((COMPANION, "oom_error_win_amd64.dll"), (RAM_ALLOC, "ram_alloc_win_amd64.dll")):
        stem = os.path.splitext(os.path.basename(source))[0]
        result = subprocess.run([cl, "/nologo", "/O2", "/EHs", "/LD", "/MD", "/std:c++17", "/I" + os.path.join(torch_root, "include"), source, f"/Fe{target}",
                                 "/link", "/LIBPATH:" + os.path.join(torch_root, "lib"), "c10.lib", "/NODEFAULTLIB:LIBCMT"], cwd=HERE, env=env)
        for leftover in glob.glob(os.path.join(HERE, stem + "*.obj")) + glob.glob(os.path.join(HERE, stem + "*.lib")) + glob.glob(os.path.join(HERE, stem + "*.exp")):
            os.remove(leftover)
        print(f"{'built' if result.returncode == 0 else 'failed to build'} {os.path.join(HERE, target)} (PyTorch {torch.__version__} headers)")
        code = code or result.returncode
    return code


def build_linux():
    compiler = os.environ.get("CXX", "g++")
    if shutil.which(compiler) is None:
        raise RuntimeError(f"{compiler} is required (C++17).")
    machine = platform.machine().lower()
    target = f"vmm_alloc_linux_{machine}.so"
    # glibc 2.17 compatibility (manylinux2014) on x86_64: the static C++ runtime's pthread calls go to their oldest versions (vmm_alloc.cpp)
    wrapped = ["-Wl,--wrap=" + name for name in ("pthread_key_create", "__pthread_key_create", "pthread_key_delete", "pthread_getspecific", "pthread_setspecific", "pthread_once")] if machine == "x86_64" else []
    # libgcc stays shared: the out of memory exception unwinds through PyTorch's frames, which needs a single unwinder (libgcc_s)
    result = subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-static-libstdc++", "-pthread",
                             SOURCE, "-o", target, *wrapped, "-Wl,--no-as-needed", "-l:libpthread.so.0", "-l:libdl.so.2"], cwd=HERE)
    if result.returncode == 0 and shutil.which("objdump"):
        symbols = subprocess.run(["objdump", "-T", os.path.join(HERE, target)], capture_output=True, text=True).stdout
        versions = sorted({tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_([0-9.]+)", symbols)})
        if versions:
            print(f"{target} needs glibc {'.'.join(map(str, versions[-1]))} or newer")
    return result.returncode or build_linux_companion(compiler), target


def build_linux_companion(compiler):
    torch = _torch()
    if torch is None or not torch.compiled_with_cxx11_abi():
        print("PyTorch with the C++11 ABI is not installed: the companion libraries (PyTorch's out of memory error, the RAM allocator) were not built")
        return 0
    torch_root, code = os.path.dirname(torch.__file__), 0
    for source, name in ((COMPANION, "oom_error"), (RAM_ALLOC, "ram_alloc")):
        target = f"{name}_linux_{platform.machine().lower()}.so"
        # the C++ runtime stays shared, as PyTorch's: c10 builds the error that PyTorch catches, and the hooks exchange PyTorch's objects
        result = subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-D_GLIBCXX_USE_CXX11_ABI=1", "-I" + os.path.join(torch_root, "include"),
                                 source, "-o", target, "-L" + os.path.join(torch_root, "lib"), "-lc10"], cwd=HERE)
        if result.returncode == 0 and shutil.which("objdump"):
            symbols = subprocess.run(["objdump", "-T", os.path.join(HERE, target)], capture_output=True, text=True).stdout
            needs = {key: max((tuple(int(x) for x in v.split(".")) for v in re.findall(key + r"_([0-9.]+)", symbols)), default=None) for key in ("GLIBC", "GLIBCXX", "CXXABI")}
            print(f"{target} needs " + ", ".join(f"{key} {'.'.join(map(str, version))}" for key, version in needs.items() if version))
        print(f"{'built' if result.returncode == 0 else 'failed to build'} {os.path.join(HERE, target)} (PyTorch {torch.__version__} headers)")
        code = code or result.returncode
    return code


def build_linux_in_wsl():
    if shutil.which("wsl") is None:
        print("WSL is not installed: the Linux library was not rebuilt (run python3 mmgp/allocator/build.py on Linux).")
        return 0
    folder = subprocess.run(["wsl", "-e", "wslpath", "-a", HERE.replace("\\", "/")], capture_output=True, text=True).stdout.strip()
    return subprocess.run(["wsl", "-e", os.environ.get("MMGP_WSL_PYTHON", "python3"), f"{folder}/build.py"]).returncode


def main():
    if sys.platform == "win32":
        code, target = build_windows()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
        code = code or build_linux_in_wsl()
    elif sys.platform.startswith("linux"):
        code, target = build_linux()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
    else:
        raise RuntimeError("The mmgp VRAM allocator supports Windows and Linux.")
    sys.exit(code)


if __name__ == "__main__":
    main()
