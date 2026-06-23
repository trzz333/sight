@echo off
REM K6 on/off comparison launcher: starts both arms detached, concurrent.
REM on = dyn_beta 1.0 (self-supervision), off = dyn_beta 0.0 (architecture-matched
REM baseline). Seeds 0-4 each. Final per-arm sentinels: k6_dyn_on.sentinel /
REM k6_dyn_off.sentinel; per-seed sentinels: k6_dyn_<arm>_s<S>\seed.sentinel.
cd /d C:\Projects\Sight
start "k6_on" cmd /c tools\run_k6_arm.bat on 1.0
start "k6_off" cmd /c tools\run_k6_arm.bat off 0.0
echo LAUNCHED > runs\phase_k\k6_launch.marker
