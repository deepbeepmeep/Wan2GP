#!/usr/bin/env bash
# Installs SpargeAttn (spas_sage_attn) from source.
#
# A plain `pip install git+https://github.com/woct0rdho/SpargeAttn.git` fails
# to build on CUDA 13 with:
#
#   /usr/local/cuda/include/cuda_fp8.hpp(...): error: identifier "__assert_fail" is undefined
#
# CUDA 13's cuda_fp8.hpp / cuda_fp6.hpp / cuda_fp4.hpp call assert() inside
# CUDAFORCEINLINE host/device functions. On glibc, assert() expands to a call
# to __assert_fail(), which nvcc's device frontend does not know about.
# Adding -DNDEBUG to the nvcc flags disables the assert macro and fixes the
# build.
#
# Additionally, setup.py runs the kernel-instantiation generators with a bare
# `python` and ignores their exit status. If `python` does not resolve to the
# build interpreter (e.g. pyenv shims, an unactivated venv), the generated
# kernels are missing from the wheel and the sm89 module fails to import.
# This script clones the repo, applies both fixes and installs it.
#
# Usage:
#   scripts/install-spargeattn.sh
#
# The build directory is re-cloned on every run (an existing directory is
# removed first), so the script is safe to re-run even after an interrupted
# build or an upstream layout change.
#
# Optional environment variables:
#   SPARGEATTN_REPO          git URL (default: https://github.com/woct0rdho/SpargeAttn.git)
#   SPARGEATTN_BUILD_DIR     where to clone (default: ./SpargeAttn-build)
#   SPARGEATTN_PYTHON        python interpreter / pip (default: python)
#   TORCH_CUDA_ARCH_LIST     e.g. "12.0" (RTX 50xx), "8.9" (RTX 40xx) or
#                        "8.6" (RTX 30xx) to only build for your GPU and
#                        speed up the build
set -euo pipefail

REPO="${SPARGEATTN_REPO:-https://github.com/woct0rdho/SpargeAttn.git}"
BUILD_DIR="${SPARGEATTN_BUILD_DIR:-$(pwd)/SpargeAttn-build}"
PYTHON="${SPARGEATTN_PYTHON:-python}"

"$PYTHON" -m pip install ninja wheel packaging

# Fresh clone every time: a leftover (possibly patched or half-built)
# directory would make `git clone` fail, and stale sources could mask
# upstream layout changes the patches below rely on.
rm -rf "$BUILD_DIR"
git clone --depth 1 "$REPO" "$BUILD_DIR"

# Patch setup.py:
#   1. add -DNDEBUG to NVCC_FLAGS_COMMON so the CUDA 13 __assert_fail build
#      failure does not occur,
#   2. run the autogen scripts with the interpreter that is building the
#      package (a bare `python` may not resolve) and fail loudly on errors.
"$PYTHON" - "$BUILD_DIR/setup.py" <<'EOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
changed = False

if '"-DNDEBUG"' not in source:
    if 'NVCC_FLAGS_COMMON' in source:
        source = source.replace(
            '    "-std=c++17",\n',
            '    "-std=c++17",\n    "-DNDEBUG",\n',
            1,
        )
        changed = True
        print("patched: added -DNDEBUG to NVCC_FLAGS_COMMON")
    else:
        print("warning: NVCC_FLAGS_COMMON not found, skipped NDEBUG patch", file=sys.stderr)

if 'os.system(f"python {py_file}")' in source:
    source = source.replace(
        'os.system(f"python {py_file}")',
        'subprocess.check_call([sys.executable, str(py_file)])',
    )
    if 'import sys' not in source:
        source = source.replace('import os\n', 'import os\nimport sys\n', 1)
    changed = True
    print("patched: autogen runs with sys.executable and fails loudly")

if changed:
    path.write_text(source, encoding="utf-8")
EOF

"$PYTHON" -m pip install --no-build-isolation "$BUILD_DIR"