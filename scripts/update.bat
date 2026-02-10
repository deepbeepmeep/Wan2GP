@echo off
cd /d "%~dp0.."
setlocal enabledelayedexpansion
title WanGP Update & Upgrade

:MENU
cls
echo ======================================================
echo                WAN2GP UPDATE / UPGRADE
echo ======================================================
python setup.py status
echo 1. Update (git pull + install requirements)
echo 2. Upgrade (Upgrade Torch, Triton, Sage Attention, etc.)
echo 3. Platform Migration (Upgrade to Py 3.11/Torch 2.10)
echo 4. Exit
echo ------------------------------------------------------
set /p choice="Select an option (1-4): "

if "%choice%"=="1" (
    python setup.py update
    pause
    goto MENU
)

if "%choice%"=="2" (
    python setup.py upgrade
    pause
    goto MENU
)

if "%choice%"=="3" (
    echo [!] This will rebuild your environment with Python 3.11/Torch 2.10
    python setup.py migrate
    pause
    goto MENU
)

if "%choice%"=="4" exit
goto MENU