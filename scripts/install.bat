@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
title WanGP Installer

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :MENU

if exist "%~dp0autoinstaller.bat" (
    echo [*] Python 3.10+ not found. Running automated installer...
    call "%~dp0autoinstaller.bat" python

    python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [-] Automated installation failed or Python is still not recognized.
        echo [*] Please install Python 3.10+ manually.
        pause
        exit /b 1
    )
    goto :MENU
) else (
    echo [-] Python 3.10+ is required but was not found.
    echo [-] 'autoinstaller.bat' was not found (or was deleted).
    echo [*] Please install Python 3.10+ manually and run this script again.
    pause
    exit /b 1
)

:MENU
set "choice="
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
set /p choice="Select an option (1-5): "

if "!choice!"=="" goto MENU
set "choice=!choice:"=!"
set "choice=!choice: =!"

if "!choice!"=="1" (
    set "ENV_TYPE=venv"
    goto START_INSTALL
)

if "!choice!"=="2" (
    set "ENV_TYPE=uv"
    where uv >nul 2>nul
    if !errorlevel! neq 0 (
        echo [-] 'uv' not found.
        echo 1. Install 'uv' via PowerShell ^(Recommended^)
        echo 2. Install 'uv' via Pip
        set /p uv_choice="Select method: "
        if "!uv_choice!"=="1" powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        if "!uv_choice!"=="2" python -m pip install uv
    )
    goto START_INSTALL
)

if "!choice!"=="3" (
    set "ENV_TYPE=conda"
    set "CONDA_FOUND=0"
    
    where conda >nul 2>nul
    if !errorlevel! equ 0 set "CONDA_FOUND=1"
    if exist "!USERPROFILE!\Miniconda3\condabin\conda.bat" set "CONDA_FOUND=1"
    if exist "!USERPROFILE!\Anaconda3\condabin\conda.bat" set "CONDA_FOUND=1"
    if exist "C:\ProgramData\Miniconda3\condabin\conda.bat" set "CONDA_FOUND=1"

    if "!CONDA_FOUND!"=="0" (
        if exist "%~dp0autoinstaller.bat" (
            call "%~dp0autoinstaller.bat" conda
            if !errorlevel! neq 0 (
                echo [-] Miniconda installation failed or was aborted.
                pause
                goto MENU
            )
        ) else (
            echo [-] 'conda' not found and 'autoinstaller.bat' was deleted.
            echo [*] Please install Miniconda manually to use this option.
            pause
            goto MENU
        )
    )
    goto START_INSTALL
)

if "!choice!"=="4" (
    set "ENV_TYPE=none"
    goto START_INSTALL
)

if "!choice!"=="5" exit
goto MENU

:START_INSTALL
if "!ENV_TYPE!"=="" set "ENV_TYPE=venv"
python setup.py install --env !ENV_TYPE!

pause
goto MENU