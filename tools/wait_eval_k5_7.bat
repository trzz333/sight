@echo off
cd /d C:\Projects\Sight
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set META=runs\phase_k\k5_7_qrdqn\train_meta.json
set ELOG=runs\phase_k\k5_7_qrdqn\k5_7_final_eval.log
set ESENT=runs\phase_k\k5_7_qrdqn_eval_done.sentinel
:waitloop
if exist %META% goto runeval
ping -n 31 127.0.0.1 >nul
goto waitloop
:runeval
ping -n 6 127.0.0.1 >nul
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\k5_7_qrdqn_eval_inenv.py --run runs\phase_k\k5_7_qrdqn --seeds 1000-1009 > %ELOG% 2>&1
echo DONE %ERRORLEVEL% > %ESENT%
