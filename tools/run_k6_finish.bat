@echo off
REM K6 FINISHER - sequential, single headless Godot at a time. Replaces the
REM concurrent two-arm launcher (failed twice: external cascade-kill mid s3,
REM and on-arm eval transport timeout under contention). No concurrency here.
REM Phase 1: re-eval on_s0..s2 (trained to 200k, eval previously timed out).
REM Phase 2: train+eval the 4 missing runs off_s3/on_s3/off_s4/on_s4, interleaved.
cd /d C:\Projects\Sight
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set PY=C:\Users\maste\AppData\Local\Python\bin\python.exe
set PROG=runs\phase_k\k6_finish_progress.log
echo [%date% %time%] K6 FINISH START detached > %PROG%

REM --- Phase 1: re-eval ON seeds 0-2 (models valid, eval timed out before) ---
for %%S in (0 1 2) do (
  echo [%date% %time%] reeval on_s%%S START >> %PROG%
  "%PY%" -u tools\k6_dyn_eval_inenv.py --run runs\phase_k\k6_dyn_on_s%%S --seeds 1000-1009 > runs\phase_k\k6_dyn_on_s%%S\eval.log 2>&1
  echo REEVAL %%S DONE > runs\phase_k\k6_dyn_on_s%%S\reeval.sentinel
  echo [%date% %time%] reeval on_s%%S DONE >> %PROG%
)

REM --- Phase 2: train+eval the 4 missing runs, interleaved by seed ---
call :run off 3 0.0
call :run on  3 1.0
call :run off 4 0.0
call :run on  4 1.0

echo [%date% %time%] K6 FINISH DONE >> %PROG%
echo K6 FINISH DONE > runs\phase_k\k6_finish.sentinel
exit /b 0

:run
set A=%1
set S=%2
set B=%3
if not exist runs\phase_k\k6_dyn_%A%_s%S% mkdir runs\phase_k\k6_dyn_%A%_s%S%
echo [%date% %time%] train %A%_s%S% (beta %B%) START >> %PROG%
"%PY%" -u tools\k6_dyn_train.py --timesteps 200000 --seed %S% --dyn-beta %B% --out runs\phase_k\k6_dyn_%A%_s%S% > runs\phase_k\k6_dyn_%A%_s%S%\train.log 2>&1
echo [%date% %time%] eval %A%_s%S% START >> %PROG%
"%PY%" -u tools\k6_dyn_eval_inenv.py --run runs\phase_k\k6_dyn_%A%_s%S% --seeds 1000-1009 > runs\phase_k\k6_dyn_%A%_s%S%\eval.log 2>&1
echo SEED %S% DONE > runs\phase_k\k6_dyn_%A%_s%S%\seed.sentinel
echo [%date% %time%] %A%_s%S% DONE >> %PROG%
exit /b 0
