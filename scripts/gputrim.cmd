@echo off
rem Forces Windows to trim idle VRAM of all processes, then shows per-process usage (see gputrim.py).
for /f %%m in ('nvidia-smi --query-gpu=memory.used --format=csv^,noheader^,nounits') do set "gputrim_before=%%m"
python "%~dp0gputrim.py" %* || exit /b 1
call "%~dp0gpumem.cmd"
echo (before trim: %gputrim_before% MiB used)
