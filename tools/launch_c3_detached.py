"""Detached windowless launcher for a C3 elite-BC seed run.

C3 owns its Godot worker pool directly (no separate supervisor), so this
mirrors the proven C1/C2 pattern: spawn the trainer with CREATE_NO_WINDOW
(a hidden-but-valid console, so the SubprocVecEnv Godot children get valid
console handles and produce NO visible windows) plus CREATE_NEW_PROCESS_GROUP
and FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 so a console close/Ctrl event from any
interactive shell cannot kill the run. PYTHONNOUSERSITE=1 keeps the env clean.

Usage: python launch_c3_detached.py <seed> [iters] [bc_epochs]
"""
import os
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

seed = sys.argv[1] if len(sys.argv) > 1 else "0"
iters = sys.argv[2] if len(sys.argv) > 2 else "60"
bc_epochs = sys.argv[3] if len(sys.argv) > 3 else "10"

PY = r"C:\Projects\Sight\.venv-c1\Scripts\python.exe"
out = r"C:\Projects\Sight\runs\phase_n\c3_screen_s%s" % seed
os.makedirs(out, exist_ok=True)

env = dict(os.environ)
env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
env["PYTHONNOUSERSITE"] = "1"

log = open(os.path.join(out, "c3_stdout.log"), "a", buffering=1)
p = subprocess.Popen(
    [PY, "-u", r"C:\Projects\Sight\tools\c3_elitebc_train.py",
     "--seed", seed, "--iters", iters, "--bc-epochs", bc_epochs,
     "--n-workers", "8", "--batch", "48", "--out", out],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    close_fds=True, env=env,
)
print("c3_pid", p.pid, "seed", seed, "iters", iters, "out", out)
