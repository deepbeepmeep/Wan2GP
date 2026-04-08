@echo off

if "%~1"=="python" goto :INSTALL_PYTHON
if "%~1"=="conda" goto :INSTALL_CONDA
exit /b 1

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

echo Set args = WScript.Arguments > dl.vbs
echo Set http = CreateObject("WinHttp.WinHttpRequest.5.1") >> dl.vbs
echo http.Open "GET", args(0), False >> dl.vbs
echo http.Send >> dl.vbs
echo If http.Status = 200 Then >> dl.vbs
echo   Set stream = CreateObject("ADODB.Stream") >> dl.vbs
echo   stream.Open >> dl.vbs
echo   stream.Type = 1 >> dl.vbs
echo   stream.Write http.ResponseBody >> dl.vbs
echo   stream.SaveToFile args(1), 2 >> dl.vbs
echo   stream.Close >> dl.vbs
echo End If >> dl.vbs

cscript //nologo dl.vbs "%DL_URL%" "%TMP_FILE%"
del dl.vbs

if exist "%TMP_FILE%" exit /b 0

echo [-] All native download methods failed.
exit /b 1