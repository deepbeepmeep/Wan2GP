@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
title WanGP Installer

:MENU
cls
echo ======================================================
echo                WAN2GP INSTALLER MENU
echo ======================================================
echo 1. Use 'venv' (Easiest - Comes prepackaged with python)
echo 2. Use 'uv' (Recommended - Handles Python 3.11 better)
echo 3. Use 'Conda'
echo 4. No Environment (Not Recommended)
echo 5. Exit
echo ------------------------------------------------------
set /p choice="Select an option (1-4): "

if "%choice%"=="1" (
    set "ENV_TYPE=venv"
    goto START_INSTALL
)

if "%choice%"=="2" (
    set "ENV_TYPE=uv"
    where uv >nul 2>nul
    if !errorlevel! neq 0 (
        echo [!] 'uv' not found.
        echo 1. Install 'uv' via PowerShell (Recommended)
        echo 2. Install 'uv' via Pip
        set /p uv_choice="Select method: "
        if "!uv_choice!"=="1" (
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            set "PATH=!USERPROFILE!\.local\bin;!APPDATA!\uv\bin;!PATH!"
        )
        if "!uv_choice!"=="2" python -m pip install uv
    )
    goto START_INSTALL
)

if "%choice%"=="3" (
    set "ENV_TYPE=conda"
    goto START_INSTALL
)

if "%choice%"=="4" (
    set "ENV_TYPE=none"
    goto START_INSTALL
)

if "%choice%"=="5" exit
goto MENU

:START_INSTALL
python setup.py install --env !ENV_TYPE!
pause