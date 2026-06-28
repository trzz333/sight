@echo off
setlocal
set "PY=C:\Projects\Sight\.venv-c1\Scripts\python.exe"
set "ROOT=C:\Projects\Sight"
set "PYTHONNOUSERSITE=1"
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
set "OUT=%ROOT%\runs\phase_n\c1_smoke_detach"
cd /d "%ROOT%"
if not exist "%OUT%" mkdir "%OUT%"
echo START %DATE% %TIME% > "%OUT%\run.log"
"%PY%" -u tools\c1_es_train.py --seed 0 --vec subproc --n-workers 8 --seeds-per-gen 4 --gens 100 --sigma0 0.1 --smoke --out "%OUT%" >> "%OUT%\run.log" 2>&1
echo EXIT %ERRORLEVEL% > "%OUT%\smoke.sentinel"
echo END %DATE% %TIME% >> "%OUT%\run.log"
endlocal
