"""Launch the C3 status server windowless (no terminal popup).

CREATE_NO_WINDOW keeps a valid-but-hidden console; DETACHED would also work
(the server spawns only tasklist children). Uses the base interpreter, not the
venv shim, since the status server is stdlib-only (no torch/sb3 needed).
Open http://localhost:8766 after launch.
"""
import os
import subprocess

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PY = r"C:\Python314\python.exe"
PHASE = r"C:\Projects\Sight\runs\phase_n"
os.makedirs(PHASE, exist_ok=True)
env = dict(os.environ)
env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
log = open(os.path.join(PHASE, "c3_status_server.log"), "a", buffering=1)
p = subprocess.Popen(
    [PY, "-u", r"C:\Projects\Sight\tools\c3_status_server.py"],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    close_fds=True, env=env,
)
print("c3_status_pid", p.pid, "url http://localhost:8766")
