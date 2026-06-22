@echo off
cd /d C:\Projects\Sight
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
set PY=C:\Users\maste\AppData\Local\Python\bin\python.exe
for %%S in (3 4 5 6 7 8 9) do (
  if not exist runs\phase_k\k5_8_noisy_qrdqn_s%%S mkdir runs\phase_k\k5_8_noisy_qrdqn_s%%S
  "%PY%" -u tools\k5_8_noisy_qrdqn_train.py --timesteps 200000 --seed %%S --out runs\phase_k\k5_8_noisy_qrdqn_s%%S > runs\phase_k\k5_8_noisy_qrdqn_s%%S\train.log 2>&1
  "%PY%" -u tools\k5_8_noisy_eval_inenv.py --run runs\phase_k\k5_8_noisy_qrdqn_s%%S --seeds 1000-1009 > runs\phase_k\k5_8_noisy_qrdqn_s%%S\eval.log 2>&1
  echo SEED %%S DONE > runs\phase_k\k5_8_noisy_qrdqn_s%%S\seed.sentinel
)
echo DONE 0 > runs\phase_k\k5_8_multiseed_3_9.sentinel
