import subprocess, os
# Windowless launch of the sd_fast sweep status server. Mirrors
# launch_status_detached.py: DETACHED_PROCESS is safe because this server has
# no multiprocessing/Godot pool. Child survives this launcher's exit.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
log_path = r"C:\Projects\Sight\runs\sd_fast\status_server.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log = open(log_path, "a", buffering=1)
p = subprocess.Popen(
    [r"C:\Python314\python.exe", "-u", r"C:\Projects\Sight\tools\sd_fast_status_server.py"],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
print("server_pid", p.pid)
