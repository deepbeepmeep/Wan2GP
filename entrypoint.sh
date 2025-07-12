#!/usr/bin/env bash
export HOME=/home/user
export PYTHONUNBUFFERED=1
export HF_HOME=/home/user/.cache/huggingface

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Disable audio warnings in Docker
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

exec su -p user -c "python3 wgp.py --listen $*"
