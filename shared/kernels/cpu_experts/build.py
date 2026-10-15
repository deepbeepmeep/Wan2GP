# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Rebuilds the CPU expert libraries after a change to experts.cpp: python -m shared.kernels.cpu_experts.build
On Windows it builds cpu_experts_win_amd64.dll with MSVC (Visual Studio or Build Tools with the C++ workload), then
cpu_experts_linux_x86_64.so in WSL when WSL is installed (g++ needed there; CPU_EXPERTS_WSL_PYTHON: the Python run there, python3 by
default). On Linux it builds the .so with g++ (OpenMP, AVX2/FMA/F16C), the C++ runtime linked statically: it needs glibc and libgomp."""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "experts.cpp")


def build_windows():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    from mmgp.allocator.build import _msvc_environment
    env, cl = _msvc_environment()
    target = "cpu_experts_win_amd64.dll"
    result = subprocess.run([cl, "/nologo", "/O2", "/EHsc", "/LD", "/MD", "/std:c++17", "/arch:AVX2", "/openmp", SOURCE, f"/Fe{target}"], cwd=HERE, env=env)
    for name in os.listdir(HERE):
        if name.startswith(("experts.", "cpu_experts_win_amd64.")) and name.endswith((".obj", ".lib", ".exp")):
            os.remove(os.path.join(HERE, name))
    return result.returncode, target


def build_linux():
    compiler = os.environ.get("CXX", "g++")
    if shutil.which(compiler) is None:
        raise RuntimeError(f"{compiler} is required (C++17, OpenMP).")
    target = "cpu_experts_linux_x86_64.so"
    result = subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden", "-mavx2", "-mfma", "-mf16c", "-fopenmp",
                             SOURCE, "-o", target], cwd=HERE)
    if result.returncode == 0 and shutil.which("objdump"):
        symbols = subprocess.run(["objdump", "-T", os.path.join(HERE, target)], capture_output=True, text=True).stdout
        versions = sorted({tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_([0-9.]+)", symbols)})
        if versions:
            print(f"{target} needs glibc {'.'.join(map(str, versions[-1]))} or newer")
    return result.returncode, target


def build_linux_in_wsl():
    if shutil.which("wsl") is None:
        print("WSL is not installed: the Linux library was not rebuilt (run python3 shared/kernels/cpu_experts/build.py on Linux).")
        return 0
    folder = subprocess.run(["wsl", "-e", "wslpath", "-a", HERE.replace("\\", "/")], capture_output=True, text=True).stdout.strip()
    return subprocess.run(["wsl", "-e", os.environ.get("CPU_EXPERTS_WSL_PYTHON", "python3"), f"{folder}/build.py"]).returncode


def main():
    if sys.platform == "win32":
        code, target = build_windows()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
        code = code or build_linux_in_wsl()
    elif sys.platform.startswith("linux"):
        code, target = build_linux()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
    else:
        raise RuntimeError("The CPU expert kernels support Windows and Linux.")
    sys.exit(code)


if __name__ == "__main__":
    main()
