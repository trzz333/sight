@echo off
setlocal
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"
set "PY=C:\Users\maste\AppData\Local\Python\bin\python.exe"
set "VPY=C:\Projects\Sight\.venv-d3rlpy\Scripts\python.exe"
set "OUT=C:\Projects\Sight\runs\phase_k\k7_offline\real"
cd /d C:\Projects\Sight
del "%OUT%\k7_traineval.done" 2>nul
echo [k7] train start %DATE% %TIME% > "%OUT%\k7_traineval.log"
"%VPY%" tools\d3rlpy_offline_train.py --npz "%OUT%\offline_dataset.npz" --out "%OUT%" --cql-steps 100000 --bc-steps 100000 --filter-frac 0.25 >> "%OUT%\k7_traineval.log" 2>&1
set "TRC=%ERRORLEVEL%"
echo [k7] train exit=%TRC% %DATE% %TIME% >> "%OUT%\k7_traineval.log"
echo [k7] eval start %DATE% %TIME% >> "%OUT%\k7_traineval.log"
"%PY%" tools\k7_eval_only.py >> "%OUT%\k7_traineval.log" 2>&1
set "ERC=%ERRORLEVEL%"
echo [k7] eval exit=%ERC% %DATE% %TIME% >> "%OUT%\k7_traineval.log"
echo TRAIN_EXIT=%TRC% EVAL_EXIT=%ERC% > "%OUT%\k7_traineval.done"
endlocal
