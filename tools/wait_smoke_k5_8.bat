@echo off
cd /d C:\Projects\Sight
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set EDONE=runs\phase_k\k5_7_qrdqn_eval_done.sentinel
set SMKDIR=runs\phase_k\k5_8_smoke
set SLOG=%SMKDIR%\k5_8_smoke.log
set SSENT=runs\phase_k\k5_8_smoke_done.sentinel
:waitloop
if exist %EDONE% goto runsmoke
ping -n 31 127.0.0.1 >nul
goto waitloop
:runsmoke
ping -n 6 127.0.0.1 >nul
if not exist %SMKDIR% mkdir %SMKDIR%
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\k5_8_noisy_qrdqn_train.py --timesteps 8000 --seed 0 --out %SMKDIR% > %SLOG% 2>&1
echo DONE %ERRORLEVEL% > %SSENT%
