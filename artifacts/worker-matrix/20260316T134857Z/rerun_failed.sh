#!/usr/bin/env bash
set -euo pipefail
cd '/Users/peteromalley/Documents/Headless-Wan2GP'
python scripts/run_worker_matrix.py \
  --case-id 't2v_basic' \
  --case-id 'extract_frame_basic'
