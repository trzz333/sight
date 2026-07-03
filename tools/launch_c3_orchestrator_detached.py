"""Launch the C3 screen orchestrator DETACHED_PROCESS (console-less).

The orchestrator only waits on sentinels and spawns eval/train children
(each with CREATE_NO_WINDOW inside their own launchers), so it needs no
console of its own; DETACHED makes it immune to interactive-shell close
events during the multi-hour screen. Mirrors launch_supervisor_c2_detached.
"""
import os
import subprocess

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
PY = r"C:\Projects\Sight\.venv-c1\Scripts\python.exe"
PHASE = r"C:\Projects\Sight\runs\phase_n"
os.makedirs(PHASE, exist_ok=True)
env = dict(os.environ)
env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
env["PYTHONNOUSERSITE"] = "1"
log = open(os.path.join(PHASE, "c3_orchestrator_stdout.log"), "a", buffering=1)
p = subprocess.Popen(
    [PY, "-u", r"C:\Projects\Sight\tools\c3_screen_orchestrator.py"],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    close_fds=True, env=env,
)
print("orchestrator_pid", p.pid)
