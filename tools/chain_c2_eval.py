import subprocess, os, sys, time
# Wait for the C2 seed-2 sentinel, then run all four held-out evals
# sequentially (s1/s2 x actor/mean). Each eval spins ONE Godot env, so
# running them one at a time after seed-2's 8-worker pool tears down keeps
# pools from contending. Detached so it survives interactive shell churn.
# ADAPT of chain_c2_seed2.py (proven in-repo pattern).
ROOT = r"C:\Projects\Sight"
VENV_PY = os.path.join(ROOT, ".venv-c1", "Scripts", "python.exe")
EVAL = os.path.join(ROOT, "tools", "c1_es_eval.py")
S2_SENT = os.path.join(ROOT, "runs", "phase_n", "c2_screen_s2", "c2_screen.sentinel")
LOG = os.path.join(ROOT, "runs", "phase_n", "c2_chain_eval.log")

def log(m):
    with open(LOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%m-%d %H:%M:%S"), m))

def rundir(seed):
    return os.path.join(ROOT, "runs", "phase_n", "c2_screen_s%d" % seed)

JOBS = []
for seed in (1, 2):
    for kind in ("actor", "mean"):
        JOBS.append((
            seed, kind,
            os.path.join(rundir(seed), "best_%s_vec.npy" % kind),
            "C2-s%d-%s" % (seed, kind),
            os.path.join(rundir(seed), "eval_%s" % kind),
        ))

log("eval chain waiter up pid=%d, waiting on seed-2 sentinel" % os.getpid())
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    if os.path.exists(S2_SENT):
        with open(S2_SENT) as f:
            val = f.read().strip()
        log("seed-2 sentinel seen: %s" % val)
        break
    time.sleep(30)
else:
    log("TIMEOUT waiting on seed-2; NOT running evals")
    sys.exit(1)

time.sleep(15)  # let seed-2 workers fully tear down before eval env starts

CREATE_NO_WINDOW = 0x08000000
for seed, kind, vec, label, out in JOBS:
    if not os.path.exists(vec):
        log("MISSING vec, skip: %s" % vec)
        continue
    cmd = [VENV_PY, EVAL, "--vec", vec, "--label", label,
           "--seeds", "1000-1009", "--out", out]
    log("RUN %s" % label)
    t0 = time.time()
    rc = subprocess.call(
        cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    log("DONE %s rc=%d elapsed=%.1fs" % (label, rc, time.time() - t0))

# Write a completion sentinel so the poller has a single file to watch.
with open(os.path.join(ROOT, "runs", "phase_n", "c2_eval_all.sentinel"), "w") as f:
    f.write("EXIT 0")
log("all evals complete; wrote c2_eval_all.sentinel")
