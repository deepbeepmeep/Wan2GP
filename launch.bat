REM Activate the Conda environment named "Wan2gp"
echo Activating Conda environment "Wan2gp"...
call conda activate Wan2gp
if errorlevel 1 (
    echo Failed to activate the Conda environment "Wan2gp".
    pause
    exit /b
)

REM --- ensure we are in the same folder as this .bat ---
cd /d "%~dp0"

REM Set PYTHON to the Python executable from the active Conda environment
set PYTHON=%CONDA_PREFIX%\python.exe
if not exist "%PYTHON%" (
    echo Python executable not found in the Conda environment. Ensure that the environment is correctly set up.
    pause
    exit /b
)
echo Using Python from Conda environment: %PYTHON%

REM Launch Wan2.1
echo Launching Wan2GP
python wgp.py
echo Wan2.1 Launched