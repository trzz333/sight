import subprocess, sys
# Launch the Python supervisor console-less (DETACHED_PROCESS) so console
# Ctrl/CLOSE events from any interactive shell cannot kill it. Safe because the
# supervisor itself has no worker pool; it spawns each trainer with its OWN new
# console (CREATE_NEW_CONSOLE, set inside run_c1_supervised.py) so the Godot
# pool gets valid handles while staying decoupled from caller shells.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
seed = sys.argv[1] if len(sys.argv) > 1 else "0"
log = open(r"C:\Projects\Sight\runs\phase_n\c1_screen_s%s\supervisor_stdout.log" % seed, "a", buffering=1)
p = subprocess.Popen(
    [r"C:\Python314\python.exe", "-u", r"C:\Projects\Sight\tools\run_c1_supervised.py", seed],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
print("supervisor_pid", p.pid)
