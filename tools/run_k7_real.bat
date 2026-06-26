@echo off
setlocal
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"
set "PY=C:\Users\maste\AppData\Local\Python\bin\python.exe"
set "OUT=C:\Projects\Sight\runs\phase_k\k7_offline\real"
cd /d C:\Projects\Sight
if not exist "%OUT%" mkdir "%OUT%"
del "%OUT%\k7_real.done" 2>nul
echo [k7] start %DATE% %TIME% > "%OUT%\k7_real.log"
"%PY%" tools\collect_offline_dataset.py ^
  --out "%OUT%" ^
  --random-eps 30 ^
  --qrdqn-stages "k6_dyn_off_s0:50000,k6_dyn_off_s0:200000,k6_dyn_off_s2:100000,k6_dyn_off_s4:200000,k6_dyn_on_s0:200000,k6_dyn_on_s1:50000,k6_dyn_on_s2:100000,k6_dyn_on_s3:200000" ^
  --qrdqn-eps 6 ^
  --bc-eps 20 ^
  --max-steps 1800 ^
  --cql-steps 100000 ^
  --bc-steps 100000 ^
  --filter-frac 0.25 ^
  --eval-seeds 1000-1009 >> "%OUT%\k7_real.log" 2>&1
echo EXITCODE=%ERRORLEVEL% > "%OUT%\k7_real.done"
echo [k7] end %DATE% %TIME% >> "%OUT%\k7_real.log"
endlocal
