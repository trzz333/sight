import subprocess, os, sys, time
# Wait for the C2 seed-1 sentinel, then launch seed 2 detached. Runs seeds
# sequentially so the two 8-Godot pools never contend (concurrent pools
# corrupt fitness timing). Detached itself so it survives shell churn.
ROOT = r"C:\Projects\Sight"
S1_SENT = os.path.join(ROOT, "runs", "phase_n", "c2_screen_s1", "c2_screen.sentinel")
LAUNCHER = os.path.join(ROOT, "tools", "launch_supervisor_c2_detached.py")
LOG = os.path.join(ROOT, "runs", "phase_n", "c2_chain_s2.log")

def log(m):
    with open(LOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%m-%d %H:%M:%S"), m))

log("chain waiter up pid=%d, waiting on seed-1 sentinel" % os.getpid())
# Cap the wait so a stuck seed-1 does not hang the chain forever. Seed-1
# budget is ~70 min; 3 h cap covers crash-restart slack.
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    if os.path.exists(S1_SENT):
        with open(S1_SENT) as f:
            val = f.read().strip()
        log("seed-1 sentinel seen: %s" % val)
        break
    time.sleep(30)
else:
    log("TIMEOUT waiting on seed-1; NOT launching seed 2")
    sys.exit(1)

time.sleep(10)  # let seed-1 workers fully tear down
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
p = subprocess.Popen(
    [r"C:\Python314\python.exe", LAUNCHER, "2"],
    stdout=open(LOG, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
log("launched seed-2 supervisor via launcher, chain pid done")
