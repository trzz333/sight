import subprocess, sys, os
# Launch the status server console-less so it is immune to console Ctrl/CLOSE
# events that were killing the WMI cmd-console version. Safe here ONLY because
# the server has no multiprocessing/Godot worker pool (DETACHED_PROCESS kills
# such pools; see launch_c1_screen.ps1 notes). Server child survives this
# launcher's exit.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
log_path = r"C:\Projects\Sight\runs\phase_n\status_server.log"
log = open(log_path, "a", buffering=1)
p = subprocess.Popen(
    [r"C:\Python314\python.exe", "-u", r"C:\Projects\Sight\tools\sight_status_server.py"],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
print("server_pid", p.pid)
