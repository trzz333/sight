import os, subprocess, sys, time

ROOT = r"C:\Projects\Sight"
PY = os.path.join(ROOT, r".venv-c1\Scripts\python.exe")
TRAINER = os.path.join(ROOT, r"tools\sd_godot_ppo_g99.py")
GODOT = (r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
         r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
         r"\Godot_v4.6.2-stable_win64.exe")
CONSOLE = os.path.join(ROOT, r"runs\sd_godot\g99curr_godot_5M_s0_console.log")

env = dict(os.environ)
env["SIGHT_GODOT_EXE"] = GODOT
env["PYTHONPATH"] = "src"

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS

args = [PY, TRAINER,
        "--curriculum", "--n-init-max", "6", "--anneal-frac", "0.7",
        "--steps", "5000000", "--n-envs", "8", "--run-id", "g99curr_godot_5M_s0"]

logf = open(CONSOLE, "w", buffering=1)
p = subprocess.Popen(args, cwd=ROOT, env=env,
                     stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     creationflags=flags)
print("TRAINER_PID", p.pid)
print("PRIORITY BELOW_NORMAL")
print("CONSOLE", CONSOLE)
