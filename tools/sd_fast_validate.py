"""Fidelity + throughput validation for SignalDodgeFast.

Fidelity anchor: Godot constant_left mean episode length = 845.7 (K5.2,
measured over held-out seeds 1000-1009). RNG differs between Godot and numpy,
so per-seed match is impossible; the mean over many seeds is the fidelity test.
Also reports constant_stay and constant_right for shape, and steps/sec.
"""
from __future__ import annotations

import time

import numpy as np

from sight_agent.rl.sd_fast import SignalDodgeFast


def run_constant(action: int, n_seeds: int, base_seed: int = 1000):
    env = SignalDodgeFast()
    lengths = []
    for s in range(n_seeds):
        env.reset(seed=base_seed + s)
        steps = 0
        while True:
            _, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    return np.array(lengths, dtype=float)


if __name__ == "__main__":
    # 10-seed slice matching Godot's held-out set, plus a 500-seed mean for a
    # low-noise fidelity estimate.
    for label, a in (("left", 0), ("stay", 1), ("right", 2)):
        l10 = run_constant(a, 10, 1000)
        l500 = run_constant(a, 500, 1000)
        print(
            f"const_{label:5s}  seeds1000-1009 mean={l10.mean():7.1f} "
            f"std={l10.std():6.1f}   500seed mean={l500.mean():7.1f} "
            f"std={l500.std():6.1f}"
        )

    # throughput: random policy
    env = SignalDodgeFast()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    N = 200000
    t0 = time.perf_counter()
    eps = 0
    for _ in range(N):
        _, _, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        if term or trunc:
            eps += 1
            env.reset(seed=eps)
    dt = time.perf_counter() - t0
    print(f"\nTHROUGHPUT random: {N} steps, {eps} eps, {dt:.2f}s, "
          f"{N/dt:,.0f} steps/s (single env)")
