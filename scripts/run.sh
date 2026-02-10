#!/bin/bash
cd "$(dirname "$0")/.."
python3 setup.py run
read -p "Press Enter to exit..."