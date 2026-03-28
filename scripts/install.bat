@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
title WanGP Installer

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if %errorlevel% equ 0 goto :MENU

if exist "C:\Program Files\PyManager\pymanager.exe" goto :INSTALL_PY311

echo [*] Downloading PyManager installer...
call :DOWNLOAD "https://www.python.org/ftp/python/pymanager/python-manager-26.0.msi" "pymanager_installer.msi"

echo [*] Installing PyManager...
start /wait msiexec /i "pymanager_installer.msi" /passive /norestart
del "pymanager_installer.msi"

if not exist "C:\Program Files\PyManager\pymanager.exe" (
    echo [-] Installation failed.
    pause
    exit /b 1
)
echo [*] PyManager installed successfully.

:INSTALL_PY311
echo [*] Configuring Python 3.11...
set "PATH=C:\Program Files\PyManager;!PATH!"

call pymanager install --configure >nul 2>&1
call pymanager install 3.11 >nul 2>&1
call pymanager install --aliases >nul 2>&1

set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;!PATH!"

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
        echo [-] 'conda' not found.
        echo [*] Downloading Miniconda3...
        call :DOWNLOAD "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" "miniconda_installer.exe"
        if not exist miniconda_installer.exe (
            echo [-] Download failed. Please install Miniconda manually.
            pause
            exit /b 1
        )
        echo [*] Installing Miniconda silently ^(this may take a minute^)...
        start /wait "" miniconda_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D="!USERPROFILE!\Miniconda3"
        del miniconda_installer.exe

        echo [*] Auto-accepting Conda Terms of Service...
        call "!USERPROFILE!\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
        call "!USERPROFILE!\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
        call "!USERPROFILE!\Miniconda3\condabin\conda.bat" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1
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

:DOWNLOAD
set "DL_URL=%~1"
set "DL_FILE=%~2"
where curl >nul 2>nul
if %ERRORLEVEL% EQU 0 ( curl -L -o "%DL_FILE%" "%DL_URL%" & if exist "%DL_FILE%" exit /b 0 )
where certutil >nul 2>nul
if %ERRORLEVEL% EQU 0 ( certutil -urlcache -split -f "%DL_URL%" "%DL_FILE%" & if exist "%DL_FILE%" exit /b 0 )
where bitsadmin >nul 2>nul
if %ERRORLEVEL% EQU 0 ( bitsadmin /transfer "WanGPDownload" /download /priority normal "%DL_URL%" "%CD%\%DL_FILE%" & if exist "%DL_FILE%" exit /b 0 )

echo Set args = WScript.Arguments > dl.vbs
echo Set http = CreateObject("WinHttp.WinHttpRequest.5.1") >> dl.vbs
echo Const WinHttpRequestOption_SecureProtocols = 9 >> dl.vbs
echo http.Option(WinHttpRequestOption_SecureProtocols) = 2048 >> dl.vbs
echo http.Open "GET", args(0), False >> dl.vbs
echo http.Send >> dl.vbs
echo If http.Status = 200 Then >> dl.vbs
echo   Set stream = CreateObject("ADODB.Stream") >> dl.vbs
echo   stream.Open >> dl.vbs
echo   stream.Type = 1 >> dl.vbs
echo   stream.Write http.ResponseBody >> dl.vbs
echo   stream.Position = 0 >> dl.vbs
echo   stream.SaveToFile args(1), 2 >> dl.vbs
echo   stream.Close >> dl.vbs
echo End If >> dl.vbs
cscript //nologo dl.vbs "%DL_URL%" "%DL_FILE%"
del dl.vbs
if exist "%DL_FILE%" exit /b 0
echo [-] All native download methods failed.

exit /b 1