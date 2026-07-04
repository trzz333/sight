"""Imitation reliability on the sd-fast replica, apples-to-apples axis with the
from-scratch arms.

Runs the EXISTING Godot-trained imitation checkpoints (BC bc_policy.pt,
PPO-finetune ppo_ft_policy.pt, both in the BCPolicyNet schema with baked
feat_mean/feat_std) greedily on the held-out replica seed block 5000-5029
(SignalDodgeFast, max_steps 1800), the SAME env/seeds/metric as
tools/sd_fast_reliability.py. Reuses that harness's rliable stat core (IQM,
percentile bootstrap CI) and the on-disk from-scratch cache for POI.

Obs parity is the enabling fact: sd_fast obs is byte-identical to Godot
main.gd _h3_build_observation (docstring + dim-by-dim match of the BC baked
mu/sd: dims 4,7 ~1.0 present-flags, dims 3,6,9 small-negative dy). So a
Godot-trained policy consumes in-distribution inputs here.

METHODOLOGY LIMIT (stated, not hidden): there is ONE imitation training run
per arm (seed 0), so the from-scratch arms' 5-training-seed run-level IQM/CI
is NOT matched. What is computed here is a single-policy EVAL-seed
distribution (30 episodes over seeds 5000-5029) with an eval-seed bootstrap.
POI vs the from-scratch pools is episode-level (Mann-Whitney), which the base
harness also computes at episode granularity, so that comparison IS matched.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = r"C:\Projects\Sight"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from k5_6_bc_train import BCPolicyNet  # noqa: E402
from sight_agent.rl.sd_fast import SignalDodgeFast  # noqa: E402

OUT = os.path.join(ROOT, "runs", "sd_fast")
FROM_SCRATCH_CACHE = os.path.join(OUT, "reliability_eval_cache.json")
IMIT_CACHE = os.path.join(OUT, "imitation_reliability_cache.json")
BAR = 930.27
EVAL_BASE_SEED = 5000
EVAL_N = 30
BOOT = 50000
RNG = np.random.default_rng(12345)

CKPTS = {
    "BC": os.path.join(ROOT, "runs", "phase_k", "k5_6_bc", "bc_policy.pt"),
    "PPO_ft": os.path.join(ROOT, "runs", "phase_k", "k5_6_bc", "ppo_ft_policy.pt"),
}
NONE_RUNS = [f"sd_fast_m21_s{s}_5M" for s in range(5)]
SHAPED_RUNS = [f"sd_fast_m21sh_s{s}_5M" for s in range(5)]


def load_policy(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    m = BCPolicyNet(in_dim=a["in_dim"], hidden=a["hidden"], n_actions=a["n_actions"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    mu = np.asarray(ck["feat_mean"], dtype=np.float32)
    sd = np.asarray(ck["feat_std"], dtype=np.float32)
    sd[sd < 1e-6] = 1.0
    return m, mu, sd


def greedy(m, mu, sd, obs) -> int:
    x = (np.asarray(obs, dtype=np.float32) - mu) / sd
    with torch.no_grad():
        return int(m(torch.from_numpy(x).unsqueeze(0)).argmax(1).item())


def eval_ckpt(path: str) -> list[int]:
    m, mu, sd = load_policy(path)
    env = SignalDodgeFast(max_steps=1800)
    lengths = []
    for s in range(EVAL_N):
        obs, _ = env.reset(seed=EVAL_BASE_SEED + s)
        steps = 0
        while True:
            a = greedy(m, mu, sd, obs)
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    return lengths


def iqm(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def boot_ci(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    b = np.array([iqm(RNG.choice(x, size=len(x), replace=True)) for _ in range(BOOT)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def poi(a, b) -> float:
    """Episode-level P(a beats b), Mann-Whitney with ties=0.5."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def pool_from_cache(runs: list[str]) -> np.ndarray | None:
    if not os.path.exists(FROM_SCRATCH_CACHE):
        return None
    cache = json.load(open(FROM_SCRATCH_CACHE))
    missing = [r for r in runs if r not in cache]
    if missing:
        print(f"  WARN from-scratch cache missing {missing}", flush=True)
        return None
    return np.concatenate([np.asarray(cache[r], dtype=float) for r in runs])


def main() -> None:
    icache = json.load(open(IMIT_CACHE)) if os.path.exists(IMIT_CACHE) else {}
    imit = {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"  SKIP {name}: checkpoint absent {path}", flush=True)
            continue
        if name in icache:
            imit[name] = np.asarray(icache[name], dtype=float)
        else:
            print(f"  eval {name} on seeds 5000-5029 ...", flush=True)
            lens = eval_ckpt(path)
            icache[name] = lens
            imit[name] = np.asarray(lens, dtype=float)
            json.dump(icache, open(IMIT_CACHE, "w"), indent=0)

    none_pool = pool_from_cache(NONE_RUNS)
    shaped_pool = pool_from_cache(SHAPED_RUNS)

    print("\n=== imitation reliability on sd-fast replica (greedy, seeds 5000-5029, 30 eps) ===")
    print(f"{'arm':>8} {'IQM':>8} {'95%CI(eval-seed boot)':>26} {'mean':>8} {'min':>6} {'max':>6} {'clears930':>10}")
    for name, arr in imit.items():
        lo, hi = boot_ci(arr)
        clears = int((arr > BAR).sum())
        print(f"{name:>8} {iqm(arr):>8.1f} [{lo:>10.1f}, {hi:>10.1f}] "
              f"{arr.mean():>8.1f} {arr.min():>6.0f} {arr.max():>6.0f} {clears:>7}/30")

    if none_pool is not None and shaped_pool is not None:
        print("\n=== from-scratch pooled reference (episode-level, same seed block) ===")
        for label, pool in (("none", none_pool), ("shaped", shaped_pool)):
            print(f"{label:>8} IQM={iqm(pool):8.1f} mean={pool.mean():8.1f} "
                  f"n={pool.size} clears930={(pool>BAR).sum()}/{pool.size}")
        print("\n=== episode-level POI (imitation beats from-scratch pool) ===")
        for name, arr in imit.items():
            print(f"  P({name} > none)={poi(arr, none_pool):.3f}   "
                  f"P({name} > shaped)={poi(arr, shaped_pool):.3f}")
    else:
        print("\n(from-scratch cache incomplete; POI skipped. Re-run tools/sd_fast_reliability.py first.)")


if __name__ == "__main__":
    main()
