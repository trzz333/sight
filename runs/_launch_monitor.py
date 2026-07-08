import subprocess, os
ROOT = r"C:\Projects\Sight"
PY = os.path.join(ROOT, r".venv-c1\Scripts\python.exe")
SRV = os.path.join(ROOT, r"runs\_monitor_server.py")
LOG = os.path.join(ROOT, r"runs\_monitor_server.out.log")
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
logf = open(LOG, "w", buffering=1)
p = subprocess.Popen([PY, SRV], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL,
                     creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
print("MONITOR_PID", p.pid)
