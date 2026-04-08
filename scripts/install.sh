#!/bin/bash
cd "$(dirname "$0")/.."

check_python() {
    command -v python3 >/dev/null 2>&1 && python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1
}

install_python() {
    echo "[*] Attempting to install Python 3.10+ via system package manager..."
    
    if command -v apt-get >/dev/null 2>&1; then
        echo "[*] Detected Debian/Ubuntu based system."
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip
        return $?
    elif command -v dnf >/dev/null 2>&1; then
        echo "[*] Detected Fedora/RHEL based system."
        sudo dnf install -y python3 python3-pip
        return $?
    elif command -v pacman >/dev/null 2>&1; then
        echo "[*] Detected Arch based system."
        sudo pacman -Sy --noconfirm python python-pip
        return $?
    else
        echo "[-] Unsupported package manager. Please install Python 3.10+ manually."
        return 1
    fi
}

install_conda() {
    echo "[-] 'conda' not found."
    echo "[*] Downloading Miniconda3..."
    
    local DL_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    local DL_FILE="miniconda_installer.sh"
    
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$DL_FILE" "$DL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$DL_FILE" "$DL_URL"
    else
        echo "[-] curl or wget is required to download Miniconda."
        return 1
    fi
    
    if [ ! -f "$DL_FILE" ]; then
        echo "[-] Download failed. Please install Miniconda manually."
        return 1
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
    return 0
}

if ! check_python; then
    echo "[*] Python 3.10+ not found. Running automated installer..."
    install_python
    
    if ! check_python; then
        echo "[-] Automated installation failed or Python is still not recognized."
        echo "[*] Please install Python 3.10+ manually."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

clear
echo "======================================================"
echo "               WAN2GP INSTALLER MENU"
echo "======================================================"
echo "1. Automatic Install (1-Click, Venv, Auto-Detect GPU)"
echo "2. Custom/Manual Install"
echo "3. Exit"
echo "------------------------------------------------------"
read -p "Select an option (1-3): " main_choice

main_choice=$(echo "$main_choice" | tr -d ' "')

if [ "$main_choice" == "1" ]; then
    ENV_TYPE="venv"
    AUTO_FLAG="--auto"
elif [ "$main_choice" == "2" ]; then
    AUTO_FLAG=""
    clear
    echo "======================================================"
    echo "               SELECT ENVIRONMENT TYPE"
    echo "======================================================"
    echo "1. Use 'venv' (Easiest - Comes prepackaged with python)"
    echo "2. Use 'uv' (Recommended - Handles Python 3.11 better)"
    echo "3. Use 'Conda'"
    echo "4. No Environment (Not Recommended)"
    echo "5. Exit"
    echo "------------------------------------------------------"
    read -p "Select an option (1-5): " choice

    choice=$(echo "$choice" | tr -d ' "')

    if [ "$choice" == "1" ]; then
        ENV_TYPE="venv"
        
    elif [ "$choice" == "2" ]; then
        ENV_TYPE="uv"
        if ! command -v uv &> /dev/null; then
            echo "[-] 'uv' not found."
            echo "1. Install 'uv' via curl (Recommended)"
            echo "2. Install 'uv' via Pip"
            read -p "Select method: " uv_choice
            
            if [ "$uv_choice" == "1" ]; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
                source "$HOME/.cargo/env" 2>/dev/null || true
            elif [ "$uv_choice" == "2" ]; then
                python3 -m pip install uv
            fi
        fi

    elif [ "$choice" == "3" ]; then
        ENV_TYPE="conda"
        CONDA_FOUND=0
        
        if command -v conda &> /dev/null; then CONDA_FOUND=1; fi
        if [ -f "$HOME/miniconda3/bin/conda" ]; then CONDA_FOUND=1; fi
        if [ -f "$HOME/anaconda3/bin/conda" ]; then CONDA_FOUND=1; fi

        if [ "$CONDA_FOUND" == "0" ]; then
            install_conda
            if [ $? -ne 0 ]; then
                echo "[-] Miniconda installation failed or was aborted."
                read -p "Press Enter to exit..."
                exit 1
            fi
        fi

    elif [ "$choice" == "4" ]; then
        ENV_TYPE="none"
        
    elif [ "$choice" == "5" ]; then
        exit 0
    else
        exit 0
    fi
elif [ "$main_choice" == "3" ]; then
    exit 0
else
    exit 0
fi

python3 setup.py install --env "$ENV_TYPE" $AUTO_FLAG
echo "Installation complete. Run ./run.sh to start."
read -p "Press Enter to exit..."