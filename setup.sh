#!/bin/bash

set -e

echo "===== Wan2GP API Setup Script ====="

# Hardcode conda path for reliability
CONDA="$HOME/miniconda/bin/conda"

# Check for conda
if [ ! -f "$CONDA" ]; then
    echo "Conda not found. Installing Miniconda..."

    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
    else
        INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
    fi

    curl -LO https://repo.anaconda.com/miniconda/$INSTALLER
    bash $INSTALLER -b -p $HOME/miniconda

    export PATH="$HOME/miniconda/bin:$PATH"

    echo "Initializing conda..."
    $CONDA init bash
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    echo "Conda already installed."
    source "$HOME/miniconda/etc/profile.d/conda.sh"
fi

# Accept Anaconda terms of service for channels
echo "Accepting Anaconda terms of service..."
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create environment using conda run (no activation needed)
echo "Creating conda environment..."
$CONDA create -y -n wan2gp python=3.11.14

# install the base Wan2GP requirements
$CONDA run -n wan2gp pip install torch==2.10.0 torchvision torchaudio
# slightly modified version for mac
$CONDA run -n wan2gp pip install -r requirements.txt

# Download improved_klein.safetensors if it doesn't exist
if [ ! -f "loras/flux2_klein_9b/improved_klein.safetensors" ]; then
    echo "Downloading improved_klein.safetensors..."
    mkdir -p "loras/flux2_klein_9b"
    curl -L -o "loras/flux2_klein_9b/improved_klein.safetensors" \
        "https://www.dropbox.com/scl/fi/v48q3apj77w4o6g61yugc/improved_klein.safetensors?rlkey=qqx97pc3hd2djtiep82qm7fj4&e=1&st=tyvoiz7g&dl=1"
    echo "Download complete."
else
    echo "improved_klein.safetensors already exists, skipping."
fi

# Download Flux2-Klein-9B-consistency-V2.safetensors if it doesn't exist
if [ ! -f "loras/flux2_klein_9b/Flux2-Klein-9B-consistency-V2.safetensors" ]; then
    echo "Downloading Flux2-Klein-9B-consistency-V2.safetensors..."
    mkdir -p "loras/flux2_klein_9b"
    curl -L -o "loras/flux2_klein_9b/Flux2-Klein-9B-consistency-V2.safetensors" \
        "https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency/resolve/main/Flux2-Klein-9B-consistency-V2.safetensors"
    echo "Download complete."
else
    echo "Flux2-Klein-9B-consistency-V2.safetensors already exists, skipping."
fi

# run the run.sh shell script
echo ""
echo "NOTE: If you get a macOS firewall prompt asking about Python accepting"
echo "incoming network connections, go to System Settings > Privacy & Security"
echo "and allow Python (python3.10) for local network access."
echo ""

./run.sh