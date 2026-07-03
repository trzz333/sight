"""Phase N / C3 - autonomous pre-registered seed screen orchestrator.

Runs the whole C3 screen unattended after seed-0 training is launched:

  1. Wait for seed-0's completion sentinel (c3_report.json).
  2. Gate seed-0's TWO vectors (best_actor_vec, best_final_vec) on the sealed
     held-out band 1000-1009 via the UNCHANGED tools\\c1_es_eval.py.
  3. Pre-registered near-miss rule: stage seeds 1+2 iff seed-0 best held-out
     mean_length >= NEAR_MISS (880.0). 880 sits between the best held-out
     either weight-search paradigm ever reached (C1 906.4, C2 845.7); a C3
     seed-0 below it is a clear single-shot miss and Phase N closes FINAL
     NEGATIVE without burning two more seeds.
  4. If staged: launch seed-1, wait for its report, gate both vectors; then
     seed-2, same. Seeds run SEQUENTIALLY so only one 8-worker Godot pool is
     live at a time (no pool contention; eval spins one env after teardown).
  5. Write a single decision file c3_screen_verdict.json + a completion
     sentinel c3_screen_all.sentinel.

Detached (DETACHED_PROCESS) so it survives interactive shell churn during the
multi-hour wait. Mirrors the proven chain_c2_eval.py sentinel pattern.
"""
import json
import os
import subprocess
import sys
import time

ROOT = r"C:\Projects\Sight"
VENV_PY = os.path.join(ROOT, ".venv-c1", "Scripts", "python.exe")
EVAL = os.path.join(ROOT, "tools", "c1_es_eval.py")
TRAIN_LAUNCH = os.path.join(ROOT, "tools", "launch_c3_detached.py")
PHASE = os.path.join(ROOT, "runs", "phase_n")
LOG = os.path.join(PHASE, "c3_screen.log")
NEAR_MISS = 880.0
ITERS = "60"
BC_EPOCHS = "10"
CREATE_NO_WINDOW = 0x08000000

def log(m):
    line = "%s  %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def rundir(seed):
    return os.path.join(PHASE, "c3_screen_s%d" % seed)

def wait_for_report(seed, deadline_s):
    rep = os.path.join(rundir(seed), "c3_report.json")
    end = time.time() + deadline_s
    while time.time() < end:
        if os.path.exists(rep):
            return True
        time.sleep(30)
    return False

def gate_vector(seed, kind):
    """Run the held-out gate on one vector; return (mean_length, pass, summary_path)."""
    vec = os.path.join(rundir(seed), "best_%s_vec.npy" % kind)
    out = os.path.join(rundir(seed), "eval_%s" % kind)
    if not os.path.exists(vec):
        log("MISSING vec seed%d %s: %s" % (seed, kind, vec))
        return None
    label = "C3-s%d-%s" % (seed, kind)
    log("GATE %s" % label)
    t0 = time.time()
    rc = subprocess.call(
        [VENV_PY, EVAL, "--vec", vec, "--label", label,
         "--seeds", "1000-1009", "--out", out],
        stdout=open(LOG, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    summ = os.path.join(out, "c1_eval_summary.json")
    d = None
    if os.path.exists(summ):
        with open(summ, encoding="utf-8") as f:
            d = json.load(f)
    log("GATE %s rc=%d elapsed=%.1fs mean=%s pass=%s" % (
        label, rc, time.time() - t0,
        None if d is None else round(d.get("mean_episode_length", -1), 1),
        None if d is None else d.get("gate_pass")))
    return d

def screen_seed(seed):
    """Gate both vectors for a seed; return best (mean_length, any_pass, detail)."""
    results = {}
    best_mean = -1.0
    any_pass = False
    for kind in ("actor", "final"):
        d = gate_vector(seed, kind)
        if d is None:
            continue
        results[kind] = {
            "mean_episode_length": d.get("mean_episode_length"),
            "action_fractions": d.get("action_fractions"),
            "max_action_fraction": d.get("max_action_fraction"),
            "gate_pass": d.get("gate_pass"),
        }
        m = float(d.get("mean_episode_length", -1))
        if m > best_mean:
            best_mean = m
        any_pass = any_pass or bool(d.get("gate_pass"))
    return best_mean, any_pass, results

def launch_train(seed):
    log("LAUNCH seed-%d training (iters=%s bc_epochs=%s)" % (seed, ITERS, BC_EPOCHS))
    subprocess.call([VENV_PY, TRAIN_LAUNCH, str(seed), ITERS, BC_EPOCHS],
                    stdout=open(LOG, "a", encoding="utf-8"),
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

def main():
    log("orchestrator up pid=%d; waiting on seed-0 report" % os.getpid())
    verdict = {"phase": "N-C3", "near_miss_threshold": NEAR_MISS, "seeds": {}}

    if not wait_for_report(0, deadline_s=6 * 3600):
        log("TIMEOUT waiting on seed-0 report; abort")
        verdict["error"] = "seed0_timeout"
        _finish(verdict)
        return 1
    time.sleep(15)  # let seed-0 pool tear down
    s0_best, s0_pass, s0_detail = screen_seed(0)
    verdict["seeds"]["0"] = {"best_held_out_mean": s0_best,
                             "any_gate_pass": s0_pass, "detail": s0_detail}
    log("seed-0 best held-out mean=%.1f any_pass=%s" % (s0_best, s0_pass))

    staged = s0_pass or (s0_best >= NEAR_MISS)
    verdict["staged_seeds_1_2"] = bool(staged)
    if not staged:
        log("seed-0 below near-miss (%.1f < %.1f) and no pass; NOT staging 1+2. "
            "C3 = clear single-shot miss." % (s0_best, NEAR_MISS))
        verdict["screen_result"] = "C3-NEGATIVE-seed0-clear-miss"
        _finish(verdict)
        return 0

    log("seed-0 clears/near-misses; staging seeds 1 and 2 sequentially")
    for seed in (1, 2):
        launch_train(seed)
        if not wait_for_report(seed, deadline_s=6 * 3600):
            log("TIMEOUT waiting on seed-%d report; abort" % seed)
            verdict["error"] = "seed%d_timeout" % seed
            _finish(verdict)
            return 1
        time.sleep(15)
        best, passed, detail = screen_seed(seed)
        verdict["seeds"][str(seed)] = {"best_held_out_mean": best,
                                       "any_gate_pass": passed, "detail": detail}
        log("seed-%d best held-out mean=%.1f any_pass=%s" % (seed, best, passed))

    passes = sum(1 for s in verdict["seeds"].values() if s["any_gate_pass"])
    means = [s["best_held_out_mean"] for s in verdict["seeds"].values()]
    verdict["n_seed_passes"] = passes
    verdict["mean_of_best_held_out"] = round(sum(means) / len(means), 1)
    # Reliability gate: a paradigm PASS requires the bar cleared reliably, not
    # on a single lucky seed (project finding: NoisyNet's 1/10 lucky seed never
    # counted). Require all 3 seeds to pass.
    if passes == 3:
        verdict["screen_result"] = "C3-PASS-reliable-3of3"
    elif passes >= 1:
        verdict["screen_result"] = "C3-NEGATIVE-unreliable-%dof3" % passes
    else:
        verdict["screen_result"] = "C3-NEGATIVE-0of3"
    _finish(verdict)
    return 0

def _finish(verdict):
    vpath = os.path.join(PHASE, "c3_screen_verdict.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    with open(os.path.join(PHASE, "c3_screen_all.sentinel"), "w", encoding="utf-8") as f:
        f.write("EXIT 0 %s" % verdict.get("screen_result", "ERROR"))
    log("screen complete: %s -> %s" % (verdict.get("screen_result"), vpath))

if __name__ == "__main__":
    sys.exit(main())
