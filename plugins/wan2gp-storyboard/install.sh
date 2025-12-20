#!/bin/bash
# Installation script for wan2gp-storyboard plugin
# Usage: From your Wan2GP root directory, run:
#   bash plugins/wan2gp-storyboard/install.sh

set -e

echo "wan2gp-storyboard Plugin Installer"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "wgp.py" ]; then
    echo "Error: This script must be run from the Wan2GP root directory"
    echo "Usage: bash plugins/wan2gp-storyboard/install.sh"
    exit 1
fi

# Check if plugin files exist
if [ ! -f "plugins/wan2gp-storyboard/plugin.py" ]; then
    echo "Error: Plugin files not found in plugins/wan2gp-storyboard/"
    exit 1
fi

echo "✓ Plugin files found"

# Check if wgp_config.json exists
if [ ! -f "wgp_config.json" ]; then
    echo "Error: wgp_config.json not found"
    echo "Please run WanGP at least once to generate the config file"
    exit 1
fi

echo "✓ Config file found"

# Check if plugin is already enabled
if grep -q '"wan2gp-storyboard"' wgp_config.json; then
    echo "✓ Plugin is already enabled in config"
else
    echo "Adding plugin to enabled_plugins list..."

    # Use Python to safely modify the JSON
    python3 << 'EOF'
import json

with open('wgp_config.json', 'r') as f:
    config = json.load(f)

if 'enabled_plugins' not in config:
    config['enabled_plugins'] = []

if 'wan2gp-storyboard' not in config['enabled_plugins']:
    config['enabled_plugins'].append('wan2gp-storyboard')

    with open('wgp_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print("✓ Plugin added to config")
else:
    print("✓ Plugin already in config")
EOF
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Restart WanGP"
echo "2. Look for the 'Storyboard' tab in the UI"
echo ""
echo "For usage instructions, see: plugins/wan2gp-storyboard/README.md"
