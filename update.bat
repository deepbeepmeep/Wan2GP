@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Disable QuickEdit which can cause the uvicorn server to hang
call quickEdit 2

SET "PATH=C:\Miniconda3;C:\Miniconda3\Scripts;%PATH%"

echo Updating core Wan2GP packages...

call conda run -n wan2gp --live-stream pip install -r requirements.txt

echo Updating API server packages...

pushd "wgp_fastapi"

call conda run -n wan2gp --live-stream uv sync

echo Done.

pause