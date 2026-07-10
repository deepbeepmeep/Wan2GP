@echo off
setlocal

set "CHROME="

for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
) do (
  if exist "%%~P" set "CHROME=%%~P"
)

if not defined CHROME (
  echo Chrome was not found in the usual install locations.
  echo Edit this script and set CHROME to your chrome.exe path.
  exit /b 1
)

if "%~1"=="" (
  set "TARGET_URL=http://localhost:7860/"
) else (
  set "TARGET_URL=%*"
)

start "" "%CHROME%" ^
  --disable-gpu ^
  --disable-gpu-compositing ^
  --disable-accelerated-2d-canvas ^
  --disable-accelerated-video-decode ^
  --use-angle=swiftshader ^
  --enable-unsafe-swiftshader ^
  --disable-webgpu ^
  %TARGET_URL%
