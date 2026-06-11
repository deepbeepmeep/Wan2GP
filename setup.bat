@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

@echo off

:: Disable QuickEdit which can cause the uvicorn server to hang
call quickEdit 2

echo ===== Wan2GP API Setup Script =====

:: Go to the directory, including separate/external drives
pushd "%~dp0"

:: Check for conda
IF NOT EXIST "C:\Miniconda3" (
    echo Conda not found. Launching Miniconda installer...

    :: Launch install_miniconda.bat in a new admin command window
    powershell -NoProfile -Command "Start-Process cmd.exe -ArgumentList '/k \"%CD%\install_miniconda.bat\"' -Verb RunAs"

    echo Miniconda installation started in new window. Please wait for it to complete...
    exit \B
)

echo Conda already installed. Setting PATH...

SET "PATH=C:\Miniconda3;C:\Miniconda3\Scripts;%PATH%"

:: Accepting TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

:: Create environment if it doesn't exist
echo Creating conda environment...
conda create -y -n wan2gp python=3.11.14

:: Setup Wan2GP itself
call conda run -n wan2gp --live-stream pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
call conda run -n wan2gp --live-stream pip install -r requirements.txt

:: Download improved_klein.safetensors if it doesn't exist (Dropbox link - may need manual download)
if not exist "loras\flux2_klein_9b\improved_klein.safetensors" (
    echo Downloading improved_klein.safetensors...
    if not exist "loras\flux2_klein_9b" mkdir "loras\flux2_klein_9b"
    powershell -NoProfile -Command "& { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.dropbox.com/scl/fi/v48q3apj77w4o6g61yugc/improved_klein.safetensors?rlkey=qqx97pc3hd2djtiep82qm7fj4&e=1&st=tyvoiz7g&dl=1' -OutFile 'loras\flux2_klein_9b\improved_klein.safetensors' }"
    echo Download complete.
) else (
    echo improved_klein.safetensors already exists, skipping.
)

:: Now setup the FastAPI server wrapper
pushd "wgp_fastapi"
call conda run -n wan2gp --live-stream uv sync

echo Setup done. Please open 'run.bat' to run the server.

pause

exit \B