"""K5.6 behavioral-cloning dataset generation.

Run the K5.2 ``hazard_reactive_oracle`` across many seeds against the
production ``GodotSignalDodgeEnv`` (state mode, headless). At every step
record the 10-dim state observation (the deployable policy features) and
the oracle's wire action (the label). The oracle reads raw geometry
(``reward_state``); the recorded features are the same 10-dim obs a
trained network will see at deploy time. This yields an on-expert-policy
BC dataset: states the network will encounter, labelled by an expert
that clears the 930.27 survival bar (K5.2: oracle mean 1762.8 vs best
constant 845.7).

No training here. Output:
  runs/phase_k/k5_6_bc/dataset_<seed0>_<seedN>.npz
with X (N,10) float32, y (N,) int64, ep_lengths (E,) int32,
seeds (E,) int32.

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_6_bc_dataset.py --seeds 2000-2039 --out runs\\phase_k\\k5_6_bc
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from k5_2_env_dynamics_probe import (  # noqa: E402
    _build_env,
    hazard_reactive_oracle,
)


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run(seeds: list[int], out_dir: Path, max_steps: int = 1800) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    ep_lengths: list[int] = []
    seed_log: list[int] = []
    env = _build_env(
        observation_mode="state",
        run_dir=out_dir / "godot",
        seed=int(seeds[0]),
        max_steps=max_steps,
        reward_shaping="none",
    )
    try:
        for s in seeds:
            obs, info = env.reset(seed=int(s))
            rs = (info.get("godot_info") or {}).get("reward_state") or {}
            steps = 0
            term = trunc = False
            t0 = time.monotonic()
            while steps < max_steps:
                a = hazard_reactive_oracle(rs)
                X_list.append(np.asarray(obs, dtype=np.float32))
                y_list.append(int(a))
                obs, r, term, trunc, info = env.step(int(a))
                rs = (info.get("godot_info") or {}).get("reward_state") or {}
                steps += 1
                if term or trunc:
                    break
            ep_lengths.append(steps)
            seed_log.append(int(s))
            print(
                f"seed {s}: steps={steps} term={term} trunc={trunc} "
                f"wall={time.monotonic() - t0:.1f}s",
                flush=True,
            )
    finally:
        env.close()

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    out_npz = out_dir / f"dataset_{seeds[0]}_{seeds[-1]}.npz"
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        ep_lengths=np.asarray(ep_lengths, dtype=np.int32),
        seeds=np.asarray(seed_log, dtype=np.int32),
    )
    dist = np.bincount(y, minlength=3) / max(1, len(y))
    print(
        f"WROTE {out_npz} X={X.shape} y={y.shape} "
        f"mean_ep_len={float(np.mean(ep_lengths)):.1f} "
        f"timeout_frac={float(np.mean([1 if e >= max_steps else 0 for e in ep_lengths])):.2f} "
        f"action_dist_LSR={dist.round(3).tolist()}",
        flush=True,
    )
    return out_npz


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_6_bc_dataset")
    p.add_argument("--seeds", default="2000-2039")
    p.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc"
    )
    p.add_argument("--max-steps", type=int, default=1800)
    args = p.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("no seeds parsed")
    run(seeds, args.out.resolve(), max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
