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
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".venv-c1" / "Scripts" / "python.exe"
TRAIN = REPO_ROOT / "tools" / "vzd_ppo_train.py"

_SUPLOG: Path | None = None


def log(msg: str) -> None:
    """Timestamped, to stdout AND a dedicated file.

    Attempt 3 died with the supervisor writing only to the trainer's inherited
    stdout, so afterwards there was no way to tell whether it had exited or been
    killed mid-wait. Its own file, opened and closed per line, answers that.
    """
    line = f"[sup {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _SUPLOG is not None:
        with open(_SUPLOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def heartbeat(out: Path, note: str) -> None:
    """Liveness proof that outlives the process. If this file stops advancing
    while no exit line was logged, the supervisor was killed, not crashed."""
    (out / "SUP_HEARTBEAT").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {note}\n", encoding="utf-8")


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
    ap.add_argument("--watchdog-sec", type=int, default=1800,
                    help="kill the leg tree if the newest checkpoint step count "
                         "does not advance for this many seconds (2026-07-20: a "
                         "leg wedged at ~0 CPU for 15h with every process alive; "
                         "poll() alone cannot see a frozen tree)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after -- are forwarded to vzd_ppo_train.py")
    args = ap.parse_args()

    global _SUPLOG
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _SUPLOG = out / "supervisor.log"
    fwd = [a for a in args.rest if a != "--"]
    stall = 0

    # Single-instance lock. The 30-min repetition trigger (sleep self-heal)
    # must never stack a second supervisor onto a live one.
    lock = out / "SUP_LOCK"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
            os.kill(other, 0)
            log(f"supervisor {other} already alive per SUP_LOCK; exiting")
            return 0
        except (OSError, ValueError):
            pass  # stale lock
    lock.write_text(str(os.getpid()), encoding="utf-8")
    log(f"supervisor up, pid {os.getpid()}, target {args.target}")

    for leg in range(args.max_restarts + 1):
        if (out / "DONE").exists():
            log("DONE sentinel present; stopping")
            return 0
        ck, done_steps = newest_ckpt(out)
        remaining = args.target - done_steps
        if remaining <= 0:
            log(f"target {args.target} reached at {done_steps}")
            return 0

        cmd = [str(PY), str(TRAIN), "--out", str(out),
               "--steps", str(remaining)] + fwd
        if ck is not None:
            cmd += ["--resume", str(ck)]
        log(f"leg {leg}: at {done_steps}, need {remaining} more, "
            f"resume={ck.name if ck else 'None'}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        # Poll rather than block, so the heartbeat keeps advancing while the leg
        # runs. A stalled heartbeat with no exit line means killed, not crashed.
        wd_steps, wd_t = done_steps, time.time()
        while proc.poll() is None:
            heartbeat(out, f"leg {leg} pid {proc.pid} alive "
                           f"{time.time() - t0:.0f}s")
            _, now_steps = newest_ckpt(out)
            if now_steps > wd_steps:
                wd_steps, wd_t = now_steps, time.time()
            elif time.time() - wd_t > args.watchdog_sec:
                log(f"WATCHDOG: no checkpoint progress in "
                    f"{args.watchdog_sec}s; killing leg {leg} tree "
                    f"(pid {proc.pid})")
                subprocess.run(["taskkill", "/PID", str(proc.pid),
                                "/T", "/F"], capture_output=True)
                break
            time.sleep(30)
        try:
            proc.wait(timeout=30)  # reap after a watchdog kill
        except Exception:
            pass
        rc = proc.returncode
        log(f"leg {leg} exited rc={rc} after {time.time() - t0:.0f}s")
        if rc == 0:
            log("trainer exited clean")
            heartbeat(out, "finished clean")
            return 0

        _, after = newest_ckpt(out)
        gained = after - done_steps
        log(f"leg {leg} gained {gained} steps")
        # A leg that dies without advancing the checkpoint means the crash is
        # deterministic, not the stochastic engine death this loop is for.
        # Restarting into it forever would just burn the machine overnight.
        stall = stall + 1 if gained < args.min_progress else 0
        if stall >= 3:
            log("3 legs with no progress; crash looks deterministic. "
                "Stopping so it gets diagnosed rather than retried.")
            return 2
        time.sleep(10)

    log("restart budget exhausted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
