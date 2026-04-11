@echo off
REM Launches the Reigh worker with default profile 3 and 15-min idle release.
REM Pass --reigh-access-token <token> or set REIGH_ACCESS_TOKEN env var.

setlocal

if "%REIGH_ACCESS_TOKEN%"=="" (
    echo [start_worker] REIGH_ACCESS_TOKEN not set. Set it with:
    echo     set REIGH_ACCESS_TOKEN=your-token-here
    echo then re-run start_worker.bat
    exit /b 1
)

"C:/Users/MC/.local/bin/uv.exe" run --python 3.10 --extra cuda128 python run_worker.py --reigh-access-token %REIGH_ACCESS_TOKEN% --wgp-profile 3 --idle-release-minutes 15

endlocal
