"""Detached windowless spawn of the shaped sweep. CREATE_NO_WINDOW keeps it off
screen; CREATE_NEW_PROCESS_GROUP + closed handles let it outlive this launcher."""
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

p = subprocess.Popen(
    [r"C:\Projects\Sight\.venv-c1\Scripts\python.exe",
     r"C:\Projects\Sight\tools\sd_fast_shaped_sweep.py"],
    cwd=r"C:\Projects\Sight",
    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("SWEEP_PID", p.pid)
