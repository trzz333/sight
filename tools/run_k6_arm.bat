@echo off
REM K6 one-arm runner: %1=arm label (on/off), %2=dyn_beta. Seeds 0-4, 200k each,
REM chained train+eval, thread-capped so two arms run concurrently on the box.
cd /d C:\Projects\Sight
set ARM=%1
set BETA=%2
set OMP_NUM_THREADS=5
set MKL_NUM_THREADS=5
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set PY=C:\Users\maste\AppData\Local\Python\bin\python.exe
for %%S in (0 1 2 3 4) do (
  if not exist runs\phase_k\k6_dyn_%ARM%_s%%S mkdir runs\phase_k\k6_dyn_%ARM%_s%%S
  "%PY%" -u tools\k6_dyn_train.py --timesteps 200000 --seed %%S --dyn-beta %BETA% --out runs\phase_k\k6_dyn_%ARM%_s%%S > runs\phase_k\k6_dyn_%ARM%_s%%S\train.log 2>&1
  "%PY%" -u tools\k6_dyn_eval_inenv.py --run runs\phase_k\k6_dyn_%ARM%_s%%S --seeds 1000-1009 > runs\phase_k\k6_dyn_%ARM%_s%%S\eval.log 2>&1
  echo SEED %%S DONE > runs\phase_k\k6_dyn_%ARM%_s%%S\seed.sentinel
)
echo ARM %ARM% DONE > runs\phase_k\k6_dyn_%ARM%.sentinel
