#!/usr/bin/env bash
#
# Build and install flash-attn (FlashAttention-2) for Linux.
#
# PyPI has no Linux wheels for flash-attn, so pip builds it from the source
# distribution with nvcc. This script:
#
#   1. Verifies the environment: an active venv (WanGP's), torch with CUDA,
#      and a CUDA toolkit (nvcc) whose major version matches torch's.
#   2. Builds flash-attn 2.8.3.post1 from the PyPI sdist with
#      --no-build-isolation (a plain pip run would build in an isolated
#      environment that does not contain the venv's PyTorch). The 2.8.3
#      sdist vendors its CUTLASS 4.0 headers, so no git clone is needed
#      and nothing is left in the repo.
#   3. Verifies the result: the package imports and a small causal
#      forward+backward runs on the current GPU.
#
# Why 2.8.3.post1: it is the first release line with native consumer
# Blackwell (sm_120, RTX 50xx) kernels, and unlike 2.7.x it builds
# unpatched on CUDA 12.8+/13.x (2.7.x's bundled CUTLASS fails on CUDA 13
# with "PFN_cuTensorMapEncodeTiled is undefined"). Verified on an RTX 3090
# (sm_86) and an RTX 5090 (sm_120) against torch 2.10.0+cu130 + CUDA 13.0:
# output is bit-identical to a patched 2.7.2.post1 build, and causal
# forward/backward matches torch's SDPA within bf16 tolerance.
#
# Usage:
#   scripts/install-flashattn.sh
#
# Optional environment variables:
#   FLASHATTN_VERSION    flash-attn version to build
#                        (default: 2.8.3.post1)
#   FLASH_ATTN_CUDA_ARCHS  which kernels to compile, semicolon-separated
#                          (default "80;120": Ampere/Ada and consumer
#                          Blackwell). Limiting it to the GPU(s) the
#                          environment will run on shortens the build:
#                            "80"  RTX 30xx/40xx only
#                            "120" RTX 50xx only
#                          Each entry adds one gencode pass per kernel.
#                          2.8.3's setup.py silently skips 90/100/120 when
#                          nvcc is too old (11.8/12.8/12.8); this script
#                          errors out when sm_120 is requested with
#                          nvcc < 12.8, so the build never silently lacks
#                          the kernels a GPU would need.
#   MAX_JOBS               parallel nvcc processes (ninja workers).
#                          Default: auto = min(half the cores, free RAM in
#                          GB / 9, each job peaking at ~8-9 GB); the
#                          script prints the resolved value before
#                          building. Lower it to keep CPU load/heat down,
#                          e.g. MAX_JOBS=4
#   NVCC_THREADS         worker threads per nvcc process (passed to nvcc
#                          as --threads N). Default: 4 (flash-attn's own
#                          default). Peak CPU usage is roughly
#                          MAX_JOBS x NVCC_THREADS threads, e.g.
#                          MAX_JOBS=2 NVCC_THREADS=1 keeps it to ~2-4
#
# The build is CPU-bound (the GPU stays idle) and takes ~10-30 minutes
# depending on the CPU. Re-running the script is safe: if the package is
# already installed at the pinned version, pip skips the build.
set -euo pipefail
cd "$(dirname "$0")/.."

FLASHATTN_VERSION="${FLASHATTN_VERSION:-2.8.3.post1}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;120}"

# ---------------------------------------------------------------------------
# 1. Environment checks
# ---------------------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: no active Python virtual environment."
    echo "  Activate WanGP's environment first, e.g.:  source venv/bin/activate"
    exit 1
fi

PYTHON_BIN="$(command -v python)"
Torch_Cuda="$( "$PYTHON_BIN" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true )"
if [ -z "$Torch_Cuda" ]; then
    echo "ERROR: torch is not installed in this environment, or it was built without CUDA."
    echo "  Run:  pip install torch --index-url https://download.pytorch.org/whl/cu130"
    exit 1
fi
echo "Python:      $PYTHON_BIN"
echo "torch:       $( "$PYTHON_BIN" -c 'import torch; print(torch.__version__)' )"
echo "torch CUDA:  $Torch_Cuda"
# Locate nvcc: $CUDA_HOME first, then /usr/local/cuda* toolkits, preferring
# one whose major version matches torch's.
NVCC_BIN="${CUDA_HOME:-}/bin/nvcc"
if [ ! -x "$NVCC_BIN" ]; then
    NVCC_BIN=""
    for d in /usr/local/cuda-*/; do
        [ -x "$d/bin/nvcc" ] || continue
        major="$( "$d/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1 || true )"
        if [ "${major:-}" = "${Torch_Cuda%%.*}" ]; then
            NVCC_BIN="$d/bin/nvcc"
            break
        fi
        [ -z "$NVCC_BIN" ] && NVCC_BIN="$d/bin/nvcc"
    done
fi
if [ -z "$NVCC_BIN" ] || [ ! -x "$NVCC_BIN" ]; then
    echo "ERROR: nvcc not found (looked in \$CUDA_HOME and /usr/local/cuda-*)."
    echo "  Install the CUDA toolkit, e.g.:"
    echo "     sudo apt install nvidia-cuda-toolkit"
    echo "  (or, on Ubuntu 24.04 with NVIDIA's CUDA repo:"
    echo "     sudo apt install cuda-nvcc-13-0 cuda-cudart-dev-13-0) and re-run."
    exit 1
fi
NVCC_VERSION="$( "$NVCC_BIN" --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 )"
if [ "${NVCC_VERSION%%.*}" != "${Torch_Cuda%%.*}" ]; then
    echo "ERROR: nvcc $NVCC_VERSION major version does not match torch CUDA $Torch_Cuda."
    echo "  Install a CUDA toolkit matching the major version of your torch build"
    echo "  (e.g. a 13.x toolkit for a cu130 torch), then re-run with"
    echo "  CUDA_HOME pointing at it, e.g.:"
    echo "     CUDA_HOME=/usr/local/cuda-${Torch_Cuda%%.*} $0"
    exit 1
fi
echo "nvcc:        $NVCC_VERSION ($NVCC_BIN)"

# 2.8.3 requires CUDA >= 11.7 (its setup.py aborts below that).
if [ "$(printf '%s\n' "$NVCC_VERSION" "11.7" | sort -V | head -1)" != "11.7" ]; then
    echo "ERROR: flash-attn 2.8.3 requires CUDA >= 11.7, but nvcc is $NVCC_VERSION."
    exit 1
fi

# sm_120 (consumer Blackwell) kernels need nvcc >= 12.8; 2.8.3's setup.py
# would silently skip them on an older toolkit, leaving the RTX 50xx
# without kernels, so fail here instead.
case "$FLASH_ATTN_CUDA_ARCHS" in
    *120*)
        if [ "$(printf '%s\n' "$NVCC_VERSION" "12.8" | sort -V | head -1)" != "12.8" ]; then
            echo "ERROR: FLASH_ATTN_CUDA_ARCHS='$FLASH_ATTN_CUDA_ARCHS' includes sm_120 (RTX 50xx),"
            echo "  but nvcc $NVCC_VERSION < 12.8 cannot compile it."
            echo "  Install a CUDA toolkit >= 12.8, or build for the other GPUs, e.g.:"
            echo "     FLASH_ATTN_CUDA_ARCHS=80 $0"
            exit 1
        fi
        ;;
esac

# The sdist build imports these; make sure they are in the venv.
if ! "$PYTHON_BIN" -c 'import ninja, psutil, wheel, packaging' 2>/dev/null; then
    echo "Installing build dependencies (ninja, psutil, wheel, packaging)..."
    "$PYTHON_BIN" -m pip install ninja psutil wheel packaging
fi

# ---------------------------------------------------------------------------
# 2. Build and install
# ---------------------------------------------------------------------------
export FLASH_ATTN_CUDA_ARCHS
# Force the from-source build instead of letting setup.py probe GitHub
# releases for a prebuilt wheel: no released wheel matches WanGP's
# torch 2.10 + CUDA 13 environment, and a cu12 wheel (if one ever
# appeared) may lack the sm_120 kernels the RTX 50xx would need.
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# Report the parallelism ninja will use (2.8.3's setup.py applies the
# same heuristic when MAX_JOBS is unset; it passes NVCC_THREADS to nvcc
# as --threads).
if [ -z "${MAX_JOBS:-}" ]; then
    MAX_JOBS="$( "$PYTHON_BIN" -c 'import psutil; print(max(1, min((psutil.cpu_count() or 2) // 2, int(psutil.virtual_memory().available / (1024 ** 3) / 9))))')"
fi
export MAX_JOBS
export NVCC_THREADS="${NVCC_THREADS:-4}"
echo "Compiling flash-attn $FLASHATTN_VERSION (archs: $FLASH_ATTN_CUDA_ARCHS, MAX_JOBS: $MAX_JOBS, NVCC_THREADS: $NVCC_THREADS)..."
echo "  This is CPU-bound and takes ~10-30 minutes."

# --no-build-isolation: compile against the venv's own torch instead of an
# isolated build environment (which would pull an unrelated torch).
"$PYTHON_BIN" -m pip install --no-build-isolation "flash-attn==$FLASHATTN_VERSION"

# ---------------------------------------------------------------------------
# 3. Verify
# ---------------------------------------------------------------------------
"$PYTHON_BIN" - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

print(f"flash_attn {flash_attn.__version__} imports OK")
q = torch.randn(1, 256, 4, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
out = flash_attn_func(q, q.clone(), q.clone(), causal=True)
out.sum().backward()
assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
print(f"smoke test on {torch.cuda.get_device_name(0)}: OK")
PY

echo ""
echo "flash-attn $FLASHATTN_VERSION installed and verified."

