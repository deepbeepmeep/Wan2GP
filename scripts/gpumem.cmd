@echo off
rem Lists dedicated VRAM per process (WDDM hides it in nvidia-smi), then total VRAM used.
powershell -NoProfile -Command "(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | Where-Object CookedValue -gt 20MB | ForEach-Object { $p=[int]($_.InstanceName -replace '^pid_(\d+)_.*','$1'); [pscustomobject]@{PID=$p; Name=(Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName; MB=[int]($_.CookedValue/1MB)} } | Group-Object PID | ForEach-Object { [pscustomobject]@{PID=$_.Name; Name=$_.Group[0].Name; MB=($_.Group | Measure-Object MB -Sum).Sum} } | Sort-Object MB -Descending | Format-Table -AutoSize"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
