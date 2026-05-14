@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Change to script directory
cd /d "%~dp0"

echo Installing Miniconda...

echo Downloading Miniconda installer...
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output .\Miniconda3-latest-Windows-x86_64.exe

echo Installing Miniconda...
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=C:\Miniconda3

:: Fix permissions to allow non-admin users to write
echo Fixing permissions...
icacls "C:\Miniconda3" /grant:r "Users:(OI)(CI)F" /T

echo Cleaning up installer...
del "Miniconda3-latest-Windows-x86_64.exe"

echo Miniconda installed successfully.

:: Launch setup.bat in a new non-admin command window
echo Launching setup.bat...
start cmd /k "call setup.bat"

echo.
echo Miniconda installation complete. Setup will continue in a new window.

exit