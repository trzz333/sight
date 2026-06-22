@echo off
cd /d C:\Projects\Sight
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set ELOG=runs\phase_k\k5_8_noisy_qrdqn\k5_8_noisy_eval.log
set ESENT=runs\phase_k\k5_8_noisy_eval_done.sentinel
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\k5_8_noisy_eval_inenv.py --run runs\phase_k\k5_8_noisy_qrdqn --seeds 1000-1009 > %ELOG% 2>&1
echo DONE %ERRORLEVEL% > %ESENT%
