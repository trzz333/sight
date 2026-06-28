@echo off
setlocal
set "PY=C:\Projects\Sight\.venv-c1\Scripts\python.exe"
set "ROOT=C:\Projects\Sight"
set "PYTHONNOUSERSITE=1"
set "OUT=%ROOT%\runs\phase_n\c1_trend_s0"
cd /d "%ROOT%"
if not exist "%OUT%" mkdir "%OUT%"
echo START %DATE% %TIME% > "%OUT%\run.log"
"%PY%" -u tools\c1_es_train.py --seed 0 --vec subproc --n-workers 8 --seeds-per-gen 2 --gens 12 --sigma0 0.1 --out "%OUT%" >> "%OUT%\run.log" 2>&1
echo EXIT %ERRORLEVEL% > "%OUT%\c1_trend.sentinel"
echo END %DATE% %TIME% >> "%OUT%\run.log"
endlocal
