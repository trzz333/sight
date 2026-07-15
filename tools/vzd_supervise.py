"""Crash-tolerant supervisor for vzd_ppo_train.py.

SafeDoom handles engine deaths at the env boundary, but it cannot catch a
worker that dies outside a step/reset call, an OOM, or a driver hiccup. This is
the backstop: relaunch from the newest checkpoint until the run reaches
--target total steps or writes its DONE sentinel.

--target is a TOTAL step count. The supervisor reads the newest checkpoint's
step count out of its filename and passes the remainder to --steps, which SB3
treats as additional steps on resume (reset_num_timesteps=False).

Usage (launch this detached, not the trainer):
  .venv-c1\\Scripts\\python.exe tools\\vzd_supervise.py \\
      --out runs\\vzd\\ppo_deadly_corridor_s3_shaped --target 1500000 -- \\
      --env-id VizdoomDeadlyCorridor-v1 --doom-skill 3 --shape-reward --norm-reward
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".venv-c1" / "Scripts" / "python.exe"
TRAIN = REPO_ROOT / "tools" / "vzd_ppo_train.py"


def newest_ckpt(out: Path) -> tuple[Path | None, int]:
    best, best_n = None, -1
    for p in out.glob("*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", p.name)
        if m and int(m.group(1)) > best_n:
            best, best_n = p, int(m.group(1))
    return best, max(best_n, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, required=True,
                    help="TOTAL timesteps to reach across all legs")
    ap.add_argument("--max-restarts", type=int, default=40)
    ap.add_argument("--min-progress", type=int, default=2048,
                    help="abort if a leg advances the checkpoint by less than this")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after -- are forwarded to vzd_ppo_train.py")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fwd = [a for a in args.rest if a != "--"]
    stall = 0

    for leg in range(args.max_restarts + 1):
        if (out / "DONE").exists():
            print("[sup] DONE sentinel present; stopping", flush=True)
            return 0
        ck, done_steps = newest_ckpt(out)
        remaining = args.target - done_steps
        if remaining <= 0:
            print(f"[sup] target {args.target} reached at {done_steps}", flush=True)
            return 0

        cmd = [str(PY), str(TRAIN), "--out", str(out),
               "--steps", str(remaining)] + fwd
        if ck is not None:
            cmd += ["--resume", str(ck)]
        print(f"[sup] leg {leg}: at {done_steps}, need {remaining} more, "
              f"resume={ck.name if ck else 'None'}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        print(f"[sup] leg {leg} exited rc={rc} after {time.time() - t0:.0f}s",
              flush=True)
        if rc == 0:
            print("[sup] trainer exited clean", flush=True)
            return 0

        _, after = newest_ckpt(out)
        gained = after - done_steps
        print(f"[sup] leg {leg} gained {gained} steps", flush=True)
        # A leg that dies without advancing the checkpoint means the crash is
        # deterministic, not the stochastic engine death this loop is for.
        # Restarting into it forever would just burn the machine overnight.
        stall = stall + 1 if gained < args.min_progress else 0
        if stall >= 3:
            print("[sup] 3 legs with no progress; crash looks deterministic. "
                  "Stopping so it gets diagnosed rather than retried.", flush=True)
            return 2
        time.sleep(10)

    print("[sup] restart budget exhausted", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
