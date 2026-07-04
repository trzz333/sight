"""Sequential matched-seed none-vs-shaped sweep on the fast replica, 5M each.

Control (reward none) reuses existing seeds 0,1,2; this chain adds none seeds
3,4 and shaped seeds 0-4, so both arms are n=5 on identical seeds. Sequential to
avoid CPU contention (8-env DummyVecEnv already saturates cores). Writes a
per-run .log and a final sentinel. runs/ is gitignored; artifacts live on disk.
"""
import subprocess
import sys
from pathlib import Path

PY = r"C:\Projects\Sight\.venv-c1\Scripts\python.exe"
REPO = r"C:\Projects\Sight"
TRAINER = r"tools\sd_fast_ppo.py"
OUT = Path(r"C:\Projects\Sight\runs\sd_fast")

# (reward_mode, seed, run_id)
RUNS = [
    ("none", 3, "sd_fast_m21_s3_5M"),
    ("none", 4, "sd_fast_m21_s4_5M"),
    ("shaped", 0, "sd_fast_m21sh_s0_5M"),
    ("shaped", 1, "sd_fast_m21sh_s1_5M"),
    ("shaped", 2, "sd_fast_m21sh_s2_5M"),
    ("shaped", 3, "sd_fast_m21sh_s3_5M"),
    ("shaped", 4, "sd_fast_m21sh_s4_5M"),
]


def main():
    env = {"PYTHONPATH": "src"}
    import os
    full_env = dict(os.environ)
    full_env.update(env)
    for mode, seed, run_id in RUNS:
        log = OUT / f"{run_id}.log"
        cmd = [
            PY, TRAINER,
            "--run-id", run_id,
            "--reward-mode", mode,
            "--seed", str(seed),
            "--steps", "5000000",
            "--eval-seeds", "30",
        ]
        if mode == "shaped":
            cmd += ["--shape-coef", "0.5"]
        with open(log, "w") as fh:
            fh.write(f"=== {run_id} mode={mode} seed={seed} ===\n")
            fh.flush()
            rc = subprocess.run(
                cmd, cwd=REPO, env=full_env, stdout=fh, stderr=subprocess.STDOUT
            ).returncode
            fh.write(f"\n=== returncode {rc} ===\n")
    (OUT / "shaped_sweep.sentinel").write_text("CHAIN_DONE")


if __name__ == "__main__":
    main()
