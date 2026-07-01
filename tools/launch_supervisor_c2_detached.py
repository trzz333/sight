import subprocess, sys, os
# Launch the C2 supervisor console-less (DETACHED_PROCESS) so console
# Ctrl/CLOSE events from any interactive shell cannot kill it. Safe: the
# supervisor has no worker pool; it spawns the trainer CREATE_NO_WINDOW
# (set inside run_c2_supervised.py) so the Godot pool gets valid console
# handles with NO visible window. Proven pattern from the C1 ~8h rc=0 run.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
seed = sys.argv[1] if len(sys.argv) > 1 else "0"
out = r"C:\Projects\Sight\runs\phase_n\c2_screen_s%s" % seed
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "supervisor_stdout.log"), "a", buffering=1)
p = subprocess.Popen(
    [r"C:\Python314\python.exe", "-u",
     r"C:\Projects\Sight\tools\run_c2_supervised.py", seed],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
print("supervisor_pid", p.pid)
