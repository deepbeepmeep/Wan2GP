#!/bin/bash
cd "$(dirname "$0")/.."

check_python() {
    command -v python3 >/dev/null 2>&1 && python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1
}

if ! check_python; then
    if [ -f "$(dirname "$0")/scripts/autoinstaller.sh" ]; then
        echo "[*] Python 3.10+ not found. Running automated installer..."
        bash "$(dirname "$0")/scripts/autoinstaller.sh" python
        
        if ! check_python; then
            echo "[-] Automated installation failed or Python is still not recognized."
            echo "[*] Please install Python 3.10+ manually."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "[-] Python 3.10+ is required but was not found."
        echo "[-] 'autoinstaller.sh' was not found (or was deleted)."
        echo "[*] Please install Python 3.10+ manually and run this script again."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

clear
echo "======================================================"
echo "               WAN2GP INSTALLER MENU"
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
        if [ -f "$(dirname "$0")/scripts/autoinstaller.sh" ]; then
            bash "$(dirname "$0")/scripts/autoinstaller.sh" conda
            if [ $? -ne 0 ]; then
                echo "[-] Miniconda installation failed or was aborted."
                read -p "Press Enter to exit..."
                exit 1
            fi
        else
            echo "[-] 'conda' not found and 'autoinstaller.sh' was deleted."
            echo "[*] Please install Miniconda manually to use this option."
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

python3 setup.py install --env "$ENV_TYPE"
echo "Installation complete. Run ./run.sh to start."
read -p "Press Enter to exit..."