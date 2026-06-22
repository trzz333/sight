@echo off
cd /d C:\Projects\Sight
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set RUNDIR=runs\phase_k\k5_8_noisy_qrdqn_s2
set LOG=%RUNDIR%\k5_8_noisy_s2_200k.log
set SENT=runs\phase_k\k5_8_noisy_qrdqn_s2_200k.sentinel
if not exist %RUNDIR% mkdir %RUNDIR%
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\k5_8_noisy_qrdqn_train.py --timesteps 200000 --seed 2 --out %RUNDIR% > %LOG% 2>&1
echo DONE %ERRORLEVEL% > %SENT%
