# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Explicit, ahead-of-time build: python -m shared.kernels.native_int8.build."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import library_path


def windows_environment():
    env = os.environ.copy()
    if shutil.which("cl"):
        return env
    installer = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    install = subprocess.check_output([str(installer), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    vc_root = Path(install) / "VC/Tools/MSVC"
    version = lambda p: tuple(int(v) for v in p.name.split("."))
    vc = max(vc_root.iterdir(), key=version)
    sdk = Path(os.environ["ProgramFiles(x86)"]) / "Windows Kits/10"
    sdk_version = max((sdk / "Include").iterdir(), key=version).name
    env["PATH"] = str(vc / "bin/Hostx64/x64") + os.pathsep + env["PATH"]
    env["INCLUDE"] = ";".join(map(str, [vc / "include", *[sdk / "Include" / sdk_version / p for p in ("ucrt", "shared", "um")]]))
    env["LIB"] = ";".join(map(str, [vc / "lib/x64", *[sdk / "Lib" / sdk_version / p / "x64" for p in ("ucrt", "um")]]))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--architectures", default="75,80,86,89,90,100,120", help="SM targets accepted by the selected CUDA toolkit")
    parser.add_argument("--output", type=Path, default=library_path())
    args = parser.parse_args()
    architectures = sorted(set(int(value) for value in args.architectures.split(",")))
    if not architectures or min(architectures) < 60:
        parser.error("Supported build targets start at SM60 (older targets require CUDA 12.x)")
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in architectures]
    flags.append(f"-gencode=arch=compute_{architectures[-1]},code=compute_{architectures[-1]}")
    flags += ["-Xcompiler", "/MD" if os.name == "nt" else "-fPIC"]
    env = windows_environment() if os.name == "nt" else os.environ.copy()
    source = Path(__file__).with_name("linear.cu")
    # Keep intermediate objects away from the checkout; inference never invokes nvcc.
    with tempfile.TemporaryDirectory(prefix="wangp-int8-build-") as work:
        output = Path(work) / target.name
        subprocess.run([args.nvcc, "-shared", "-O3", "-std=c++17", *flags, str(source), "-o", str(output)], cwd=work, env=env, check=True)
        shutil.copyfile(output, target)
    print(f"Built {target} for SM {', '.join(map(str, architectures))}")


if __name__ == "__main__":
    main()
