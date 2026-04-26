@set @x=0 /*
@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
title WanGP Installer

where git >nul 2>nul
if !errorlevel! neq 0 (
    call :INSTALL_GIT
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :MENU

echo [*] Python 3.10+ not found. Running automated installer...
call :INSTALL_PYTHON

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo [-] Automated installation failed or Python is still not recognized.
    echo [*] Please install Python 3.10+ manually.
    pause
    exit /b 1
)
goto :MENU

:MENU
set "choice="
set "AUTO_FLAG="
cls
echo ======================================================
echo                WAN2GP INSTALLER MENU
echo ======================================================
echo 1. Automatic Install (1-Click)
echo 2. Manual Install (Select env type, python/torch/kernel versions)
echo 3. Exit
echo ------------------------------------------------------
set /p main_choice="Select an option (1-3): "

if "!main_choice!"=="1" (
    set "ENV_TYPE=venv"
    set "AUTO_FLAG=--auto"
    goto START_INSTALL
)
if "!main_choice!"=="3" exit
if not "!main_choice!"=="2" goto MENU

:ENV_MENU
cls
echo ======================================================
echo                SELECT ENVIRONMENT TYPE
echo ======================================================
echo 1. Use 'venv' (Easiest - Comes prepackaged with python)
echo 2. Use 'uv' (Recommended - Faster but requires installing uv)
echo 3. Use 'conda'
echo 4. No Environment (Not Recommended)
echo 5. Exit
echo ------------------------------------------------------
set /p choice="Select an option (1-5): "

if "!choice!"=="" goto ENV_MENU
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
        call :INSTALL_CONDA
        if !errorlevel! neq 0 (
            echo [-] Miniconda installation failed or was aborted.
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
goto ENV_MENU

:START_INSTALL
if "!ENV_TYPE!"=="" set "ENV_TYPE=venv"
python setup.py install --env !ENV_TYPE! !AUTO_FLAG!

pause
goto MENU

:INSTALL_GIT
echo [-] 'git' not found.
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.54.0.windows.1/Git-2.54.0-64-bit.exe"

echo [*] Downloading Git...
call :DOWNLOAD "%GIT_URL%" || (
    echo [-] Download failed. Please install Git manually.
    exit /b 1
)

for %%F in ("%GIT_URL%") do set "GIT_FILE=%%~nxF"

echo [*] Installing Git silently ^(this may take a minute^)...
start /wait "" "%GIT_FILE%" /VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES
del "%GIT_FILE%"

set "PATH=%PATH%;C:\Program Files\Git\cmd"
exit /b 0

:INSTALL_PYTHON
if exist "C:\Program Files\PyManager\pymanager.exe" goto :INSTALL_PY311

set "PY_URL=https://www.python.org/ftp/python/pymanager/python-manager-26.0.msi"

echo [*] Downloading PyManager installer...
call :DOWNLOAD "%PY_URL%" || exit /b 1

echo [*] Installing PyManager...
for %%F in ("%PY_URL%") do set "PY_FILE=%%~nxF"
start /wait msiexec /i "%PY_FILE%" /passive /norestart
del "%PY_FILE%"

if not exist "C:\Program Files\PyManager\pymanager.exe" (
    echo [-] Installation failed.
    exit /b 1
)
echo [*] PyManager installed successfully.

:INSTALL_PY311
echo [*] Configuring Python 3.11...
set "PATH=C:\Program Files\PyManager;%PATH%"

call pymanager install --configure >nul 2>&1
call pymanager install 3.11 >nul 2>&1
call pymanager install --aliases >nul 2>&1

set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
exit /b 0

:INSTALL_CONDA
echo [-] 'conda' not found.

set "CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

echo [*] Downloading Miniconda3...
call :DOWNLOAD "%CONDA_URL%" || (
    echo [-] Download failed. Please install Miniconda manually.
    exit /b 1
)

for %%F in ("%CONDA_URL%") do set "CONDA_FILE=%%~nxF"

echo [*] Installing Miniconda silently ^(this may take a minute^)...
start /wait "" "%CONDA_FILE%" /InstallationType=JustMe /RegisterPython=0 /S /D="%USERPROFILE%\Miniconda3"
del "%CONDA_FILE%"

echo [*] Auto-accepting Conda Terms of Service...
call "%USERPROFILE%\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
call "%USERPROFILE%\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
call "%USERPROFILE%\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1

exit /b 0

:DOWNLOAD
set "DL_URL=%~1"

where curl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    curl -L -O "%DL_URL%"
    exit /b %ERRORLEVEL%
)

for %%F in ("%DL_URL%") do set "TMP_FILE=%%~nxF"

where certutil >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    certutil -urlcache -split -f "%DL_URL%" "%TMP_FILE%"
    if exist "%TMP_FILE%" exit /b 0
)

where bitsadmin >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    bitsadmin /transfer "WanGPDownload" /download /priority normal "%DL_URL%" "%CD%\%TMP_FILE%"
    if exist "%TMP_FILE%" exit /b 0
)

cscript //nologo //E:JScript "%~f0" "%DL_URL%" "%TMP_FILE%"

if exist "%TMP_FILE%" exit /b 0

echo [-] All native download methods failed.
exit /b 1

*/
var args = WScript.Arguments;
if (args.Length >= 2) {
    try {
        var http = new ActiveXObject("WinHttp.WinHttpRequest.5.1");
        http.Open("GET", args(0), false);
        http.Send();
        
        if (http.Status == 200) {
            var stream = new ActiveXObject("ADODB.Stream");
            stream.Open();
            stream.Type = 1;
            stream.Write(http.ResponseBody);
            stream.SaveToFile(args(1), 2);
            stream.Close();
        }
    } catch (e) {
    }
}