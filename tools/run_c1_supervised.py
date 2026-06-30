#!/usr/bin/env python
"""Python supervisor for the C1 ES trainer.

Loops the trainer, auto-resuming from es_state.pkl after the recurring Godot
worker EOFError crash, until gen 100 or a restart cap. subprocess.run manages
the child process tree and returns cleanly on exit (unlike the PowerShell `&`
+ stream-redirect pattern, which wedged when orphaned workers held the pipe).

Run detached:
  C:\\Python314\\python.exe -u tools\\run_c1_supervised.py <seed>
"""
import subprocess, os, sys, json, time

ROOT = r"C:\Projects\Sight"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_RESTARTS = 100
PY = os.path.join(ROOT, ".venv-c1", "Scripts", "python.exe")
OUT = os.path.join(ROOT, "runs", "phase_n", "c1_screen_s%d" % SEED)
HIST = os.path.join(OUT, "es_history.json")
SUPLOG = os.path.join(OUT, "supervisor.log")
RUNLOG = os.path.join(OUT, "run.log")
SENTINEL = os.path.join(OUT, "c1_screen.sentinel")
os.makedirs(OUT, exist_ok=True)


def gens_done():
    if os.path.exists(HIST):
        try:
            return len(json.load(open(HIST)))
        except Exception:
            return -1   # mid-write; treat as unknown, do not act on it
    return 0


def log(msg):
    with open(SUPLOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%m-%d %H:%M:%S"), msg))


def kill_godot():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process godot* -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True)


def main():
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    log("supervisor up seed=%d start_gen=%s pid=%d" % (SEED, gens_done(), os.getpid()))
    restart = 0
    stuck = 0
    last_gen = gens_done()
    while True:
        g = gens_done()
        if g >= 100:
            open(SENTINEL, "w").write("EXIT 0")
            log("DONE gen=%d" % g)
            break
        if restart >= MAX_RESTARTS:
            open(SENTINEL, "w").write("EXIT FAIL_MAXRESTARTS gen=%s" % g)
            log("GAVE UP after %d restarts gen=%s" % (restart, g))
            break
        # detect a hard instant-crash loop: no progress across 6 straight restarts
        if g >= 0:
            if g <= last_gen:
                stuck += 1
            else:
                stuck = 0
            last_gen = g
        if stuck >= 6:
            open(SENTINEL, "w").write("EXIT FAIL_NOPROGRESS gen=%s" % g)
            log("GAVE UP no-progress gen=%s" % g)
            break
        restart += 1
        kill_godot()
        time.sleep(2)
        log("launch #%d gen=%s" % (restart, g))
        with open(RUNLOG, "a") as lf:
            lf.write("\n=== supervised launch #%d gen=%s %s ===\n"
                     % (restart, g, time.strftime("%m-%d %H:%M:%S")))
            lf.flush()
            rc = subprocess.run(
                [PY, "-u", os.path.join(ROOT, "tools", "c1_es_train.py"),
                 "--seed", str(SEED), "--vec", "subproc", "--n-workers", "8",
                 "--seeds-per-gen", "4", "--gens", "100", "--sigma0", "0.1",
                 "--resume", "--out", OUT],
                stdout=lf, stderr=subprocess.STDOUT, env=env,
                creationflags=0x08000000).returncode   # CREATE_NO_WINDOW: console for Godot pool but NO visible window; decoupled from caller shells
        log("exit rc=%s gen=%s" % (rc, gens_done()))
        time.sleep(5)


if __name__ == "__main__":
    main()
