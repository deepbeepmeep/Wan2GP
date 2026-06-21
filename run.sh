#!/bin/bash

# Hardcode conda path for reliability
CONDA="$HOME/miniconda/bin/conda"

# Defaults
RUN_FULL=0
PORT=8000

# Parse flags
while getopts "fp:" opt; do
  case $opt in
    f)
      RUN_FULL=1
      ;;
    p)
      PORT="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo "======================================"
if [ "$RUN_FULL" -eq 1 ]; then
    echo "Starting Full Wan2GP GUI (v2.5.3)..."
    echo "Full Wan2GP will be accessible at: http://<your_mac_ip_address>:7860"
else
    echo "Starting Wan2GP API on port $PORT..."
    echo "It will be accessible on your network at:"
    echo "http://<your_mac_ip_address>:$PORT"
    echo "Enter the ip address of your mac (find this via Settings) into the 'IP Address' field of the FluxMotion app and tap 'Connect'."
fi
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$RUN_FULL" -eq 1 ]; then
    # Run only full Wan2GP GUI (no API)
    $CONDA run -n wan2gp --live-stream python wgp.py --listen
else
    # just starts the server for you; run setup.sh if you have not done so already
    $CONDA run -n wan2gp --live-stream python -m uvicorn wgp_fastapi.api.routes:app --host 0.0.0.0 --port "$PORT"
fi