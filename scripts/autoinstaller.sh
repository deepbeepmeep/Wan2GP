#!/bin/bash

MODE=$1

if [ "$MODE" == "python" ]; then
    echo "[*] Attempting to install Python 3.10+ via system package manager..."
    
    if command -v apt-get >/dev/null; then
        echo "[*] Detected Debian/Ubuntu based system."
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip
        exit $?
    elif command -v dnf >/dev/null; then
        echo "[*] Detected Fedora/RHEL based system."
        sudo dnf install -y python3 python3-pip
        exit $?
    elif command -v pacman >/dev/null; then
        echo "[*] Detected Arch based system."
        sudo pacman -Sy --noconfirm python python-pip
        exit $?
    else
        echo "[-] Unsupported package manager. Please install Python 3.10+ manually."
        exit 1
    fi
fi

if [ "$MODE" == "conda" ]; then
    echo "[-] 'conda' not found."
    echo "[*] Downloading Miniconda3..."
    
    DL_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    DL_FILE="miniconda_installer.sh"
    
    if command -v curl >/dev/null; then
        curl -L -o "$DL_FILE" "$DL_URL"
    elif command -v wget >/dev/null; then
        wget -O "$DL_FILE" "$DL_URL"
    else
        echo "[-] curl or wget is required to download Miniconda."
        exit 1
    fi
    
    if [ ! -f "$DL_FILE" ]; then
        echo "[-] Download failed. Please install Miniconda manually."
        exit 1
    fi
    
    echo "[*] Installing Miniconda silently (this may take a minute)..."
    bash "$DL_FILE" -b -p "$HOME/miniconda3"
    rm "$DL_FILE"
    
    echo "[*] Auto-accepting Conda Terms of Service and configuring..."
    "$HOME/miniconda3/bin/conda" config --set auto_activate_base false
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1
    
    echo "[*] Miniconda installation complete!"
    echo "[*] Note: You may need to restart your terminal or run 'source $HOME/miniconda3/bin/activate' to use conda universally."
    exit 0
fi

exit 1