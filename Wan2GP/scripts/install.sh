#!/bin/bash
cd "$(dirname "$0")/.."
clear
echo "======================================================"
echo "               WAN2GP INSTALLER MENU"
echo "======================================================"
echo "1. Use 'venv' (Easiest - Comes prepackaged)"
echo "2. Use 'uv' (Recommended - Fast)"
echo "3. Use 'Conda'"
echo "4. No Environment (Not Recommended)"
echo "5. Exit"
echo "------------------------------------------------------"
read -p "Select an option (1-4): " choice

if [ "$choice" == "1" ]; then
    ENV_TYPE="venv"
elif [ "$choice" == "2" ]; then
    ENV_TYPE="uv"
    if ! command -v uv &> /dev/null; then
        echo "[!] 'uv' not found."
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi
elif [ "$choice" == "3" ]; then
    ENV_TYPE="conda"
elif [ "$choice" == "4" ]; then
    ENV_TYPE="none"
else
    exit 0
fi

python3 setup.py install --env $ENV_TYPE
echo "Installation complete. Run ./run.sh to start."
read -p "Press Enter to exit..."