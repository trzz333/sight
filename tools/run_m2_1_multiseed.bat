@echo off
setlocal
set "PY=C:\Users\maste\AppData\Local\Python\bin\python.exe"
set "ROOT=C:\Projects\Sight"
set "RUNS=%ROOT%\runs\phase_m"
cd /d "%ROOT%"
if not exist "%RUNS%" mkdir "%RUNS%"
echo [m2.1] batch start %DATE% %TIME% > "%RUNS%\m2_1_multiseed.log"

for %%S in (0 1 2) do (
  echo [m2.1] seed %%S start %DATE% %TIME% >> "%RUNS%\m2_1_multiseed.log"
  if not exist "%RUNS%\m2_1_s%%S" mkdir "%RUNS%\m2_1_s%%S"
  "%PY%" -u tools\m2_state_ppo_train.py --seed %%S --timesteps 1000000 --n-envs 8 --vec subproc --out "%RUNS%\m2_1_s%%S" > "%RUNS%\m2_1_s%%S\train.log" 2>&1
  echo SEED %%S EXIT %ERRORLEVEL% > "%RUNS%\m2_1_s%%S.sentinel"
  echo [m2.1] seed %%S done %DATE% %TIME% >> "%RUNS%\m2_1_multiseed.log"
)

echo [m2.1] batch end %DATE% %TIME% >> "%RUNS%\m2_1_multiseed.log"
echo ALLDONE > "%RUNS%\m2_1_multiseed.done"
endlocal
