@echo off
setlocal
cd /d "%~dp0.."
title WanGP DLSS 5 Installer

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_dlss5.ps1" %*
set "INSTALL_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%INSTALL_EXIT_CODE%"=="0" echo DLSS 5 installation failed with exit code %INSTALL_EXIT_CODE%.
pause
exit /b %INSTALL_EXIT_CODE%
