@echo off
REM Installation script for wan2gp-storyboard plugin
REM Usage: From your Wan2GP root directory, run:
REM   plugins\wan2gp-storyboard\install.bat

echo wan2gp-storyboard Plugin Installer
echo ===================================
echo.

REM Check if we're in the right directory
if not exist "wgp.py" (
    echo Error: This script must be run from the Wan2GP root directory
    echo Usage: plugins\wan2gp-storyboard\install.bat
    pause
    exit /b 1
)

REM Check if plugin files exist
if not exist "plugins\wan2gp-storyboard\plugin.py" (
    echo Error: Plugin files not found in plugins\wan2gp-storyboard\
    pause
    exit /b 1
)

echo [32mOK[0m Plugin files found

REM Check if wgp_config.json exists
if not exist "wgp_config.json" (
    echo Error: wgp_config.json not found
    echo Please run WanGP at least once to generate the config file
    pause
    exit /b 1
)

echo [32mOK[0m Config file found

REM Use Python to safely modify the JSON
python -c "import json; config = json.load(open('wgp_config.json')); config.setdefault('enabled_plugins', []); config['enabled_plugins'].append('wan2gp-storyboard') if 'wan2gp-storyboard' not in config['enabled_plugins'] else None; json.dump(config, open('wgp_config.json', 'w'), indent=4); print('[32mOK[0m Plugin enabled in config')"

if %errorlevel% neq 0 (
    echo Error: Failed to update config
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Restart WanGP
echo 2. Look for the 'Storyboard' tab in the UI
echo.
echo For usage instructions, see: plugins\wan2gp-storyboard\README.md
echo.
pause
