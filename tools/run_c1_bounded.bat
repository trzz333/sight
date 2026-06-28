@echo off
setlocal
set "PY=C:\Projects\Sight\.venv-c1\Scripts\python.exe"
set "ROOT=C:\Projects\Sight"
set "PYTHONNOUSERSITE=1"
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
set "SEED=%1"
if "%SEED%"=="" set "SEED=0"
set "WALL=%2"
if "%WALL%"=="" set "WALL=5400"
set "OUT=%ROOT%\runs\phase_n\c1_screen_s%SEED%"
cd /d "%ROOT%"
if not exist "%OUT%" mkdir "%OUT%"
echo START %DATE% %TIME% seed %SEED% wall %WALL%s >> "%OUT%\run.log"
"%PY%" -u tools\c1_es_train.py --seed %SEED% --vec subproc --n-workers 8 --seeds-per-gen 4 --gens 100 --sigma0 0.1 --resume --max-wall-s %WALL% --out "%OUT%" >> "%OUT%\run.log" 2>&1
echo EXIT %ERRORLEVEL% > "%OUT%\c1_screen.sentinel"
echo END %DATE% %TIME% >> "%OUT%\run.log"
endlocal
