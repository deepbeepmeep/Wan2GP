@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Disable QuickEdit which can cause the uvicorn server to hang
call quickEdit 2

SET "PATH=C:\Miniconda3;C:\Miniconda3\Scripts;%PATH%"

:: Get local IP
FOR /F "tokens=2 delims=:" %%A IN ('ipconfig ^| findstr /i "IPv4"') DO (
    SET "LOCAL_IP=%%A"
    SET "LOCAL_IP=!LOCAL_IP:~1!"
    GOTO :breakLoop
)
:breakLoop

:: Parse flags
SET RUN_FULL=0
:parseArgs
IF "%~1"=="" GOTO :endParse
IF "%~1"=="-f" (
    SET RUN_FULL=1
    SHIFT
    GOTO :parseArgs
)
IF "%~1"=="-F" (
    SET RUN_FULL=1
    SHIFT
    GOTO :parseArgs
)
SHIFT
GOTO :parseArgs
:endParse

echo ======================================
echo Starting Wan2GP API...
echo It will be accessible on your network at:
echo http://%LOCAL_IP%:8888
echo ======================================

SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Just runs the server; please run setup.bat if you have not already
IF "%RUN_FULL%"=="1" (
    conda run -n wan2gp python wgp.py --listen
) ELSE (
    conda run -n wan2gp --live-stream python -m uvicorn wgp_fastapi.api.routes:app --host 0.0.0.0 --port 8000
)

pause