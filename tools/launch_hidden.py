"""Run a command with no console window, from a windowless (pythonw) parent.

Scheduled tasks that point at a .cmd open a visible console (a Windows
Terminal tab on Win 11), and ViZDoom writes to the console even when stdout
is redirected. Jeff has asked five times for no popup windows. This is the
deterministic fix: Task Scheduler and the Startup entry invoke
    pythonw.exe tools\\launch_hidden.py <path-to-cmd>
pythonw is GUI-subsystem (no console can exist), and the child is spawned
with CREATE_NO_WINDOW so it gets an invisible console that its own children
(python train legs, the doom engine) inherit. CREATE_NEW_PROCESS_GROUP per
project rule; never DETACHED_PROCESS (kills worker pools).
"""
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

if __name__ == "__main__":
    # Pass ALL args through so one-off jobs (probes, evals) can run hidden
    # without needing a wrapper .cmd. Single-.cmd usage unchanged.
    subprocess.Popen(
        ["cmd.exe", "/c"] + sys.argv[1:],
        creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
