# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Maintainer recipe: rebuild only CUDA/HIP helpers from official wheels, never Triton.

Run with CPython 3.11 (CUDA) or 3.12 (HIP), MSVC x64 and Windows SDK;
HIP also takes --clang-cl pointing to ROCm's clang-cl.exe:
  python rebuild.py --backend cuda --wheel PATH --msvc TOOLSET_ROOT --sdk SDK_ROOT --output BUILD_DIR
Normal WanGP users do not run this script or need these build tools.
The output directory must be new. Review its binary hash and validate both releases
before replacing the bundled binary and updating manifest.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import sysconfig
import zipfile


def prepare_source(source, backend):
    if backend == "cuda":
        source = source.replace(b"CUlaunchAttribute clusterAttr = {};", b"CUlaunchAttribute clusterAttr = {0};")
        return source.replace(b"CUlaunchAttribute clusterSchedulingAttr = {};", b"CUlaunchAttribute clusterSchedulingAttr = {0};")
    source = source.replace(b'static const char *hipLibSearchPaths[] = {"/*py_libhip_search_path*/"};', b'#include "hip_library.c"')
    start = source.index(b"  void *lib;", source.index(b"bool initSymbolTable()"))
    end = source.index(b"  int hipVersion = checkDriverVersion(lib);", start)
    return source[:start] + b"  void *lib = wangp_load_hip_library();\n  if (!lib)\n    return false;\n\n" + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--msvc", type=Path, required=True, help="VC/Tools/MSVC/<version>")
    parser.add_argument("--clang-cl", type=Path, help="ROCm clang-cl.exe, required for HIP")
    parser.add_argument("--sdk", type=Path, required=True, help="Windows Kits/10")
    parser.add_argument("--sdk-version", default="10.0.22000.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    minor = 11 if args.backend == "cuda" else 12
    if sys.platform != "win32" or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, minor) or struct.calcsize("P") != 8:
        parser.error(f"Use 64-bit CPython 3.{minor} on Windows.")
    manifest_name = "manifest.json" if args.backend == "cuda" else "manifest_hip.json"
    manifest = json.loads(Path(__file__).with_name(manifest_name).read_text(encoding="utf-8"))
    if args.backend == "hip" and args.clang_cl is None:
        parser.error("HIP's bundled headers require --clang-cl.")
    backend = "nvidia" if args.backend == "cuda" else "amd"
    prefix = f"triton/backends/{backend}"
    build = manifest["builds"][manifest["built_from"]]
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != build["wheel_sha256"]:
        raise RuntimeError("Use the official wheel identified by manifest.json.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith((prefix + "/include/", prefix + "/lib/")) and not name.endswith("/"):
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        source = archive.read(prefix + "/driver.c")
    name = args.backend + "_utils"
    (output / "driver-original.c").write_bytes(source)
    source = prepare_source(source, args.backend)
    if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != manifest["build_source_sha256"]:
        raise RuntimeError("Unexpected helper source after the MSVC compatibility edits.")
    (output / f"{name}.c").write_bytes(source)
    if args.backend == "hip":
        adapter = Path(__file__).with_name("hip_library.c").read_bytes()
        if hashlib.sha256(adapter.replace(b"\r\n", b"\n")).hexdigest() != manifest["hip_library_sha256"]:
            raise RuntimeError("Unexpected HIP runtime adapter source.")
        (output / "hip_library.c").write_bytes(adapter)
    vc, sdk = args.msvc.resolve(), args.sdk.resolve()
    compiler_dir = vc / "bin/Hostx64/x64"
    env = os.environ.copy()
    env["PATH"] = str(compiler_dir) + os.pathsep + env["PATH"]
    env["INCLUDE"] = os.pathsep.join(str(p) for p in [vc / "include", *(sdk / "Include" / args.sdk_version / part for part in ("ucrt", "shared", "um"))])
    env["LIB"] = os.pathsep.join(str(p) for p in [vc / "lib/x64", *(sdk / "Lib" / args.sdk_version / part / "x64" for part in ("ucrt", "um"))])
    link_options = ["/LIBPATH:triton/backends/nvidia/lib/x64", "cuda.lib"] if args.backend == "cuda" else []
    compiler = args.clang_cl.resolve() if args.clang_cl else compiler_dir / "cl.exe"
    command = [str(compiler), f"{name}.c", "/nologo", *manifest["build_options"], "/wd4819",
               "/I" + prefix + "/include", "/I" + sysconfig.get_paths()["include"], f"/Fo{name}.obj", "/link",
               *link_options, "/LIBPATH:" + str(Path(sys.base_prefix) / "libs"),
               f"python3{minor}.lib", "/OUT:" + manifest["binary"], f"/IMPLIB:{name}.lib", "/Brepro"]
    result = subprocess.run(command, cwd=output, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "build.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout)
    result.check_returncode()
    binary = output / manifest["binary"]
    dependencies = subprocess.check_output([str(compiler_dir / "dumpbin.exe"), "/DEPENDENTS", str(binary)], env=env, text=True)
    (output / "dependencies.txt").write_text(dependencies, encoding="utf-8")
    print(dependencies)
    print("SHA256:", hashlib.sha256(binary.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
