"""Reload sd_fast_m21_s0_5M and dump the per-seed held-out distribution.

Recomputes greedy eval (deterministic, reproducible) on seeds 5000-5029 to
report median and IQM, the robust metrics M2.1 was judged on (IQM 418). The
run summary saved only mean/std/min/max; the mean can be inflated by the 1800
step cap, so median/IQM settle whether the clear is real.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sight_agent.rl.sd_fast import SignalDodgeFast

OUT = r"C:\Projects\Sight\runs\sd_fast"
RUN = "sd_fast_m21_s0_5M"


def make_env():
    return SignalDodgeFast()


def iqm(x: np.ndarray) -> float:
    x = np.sort(x)
    lo, hi = int(len(x) * 0.25), int(np.ceil(len(x) * 0.75))
    return float(x[lo:hi].mean())


def main():
    model = PPO.load(f"{OUT}\\{RUN}.zip", device="cpu")
    vn = VecNormalize.load(f"{OUT}\\{RUN}_vecnormalize.pkl", DummyVecEnv([make_env]))
    vn.training = False
    vn.norm_reward = False
    raw = SignalDodgeFast(max_steps=1800)
    lengths = []
    for s in range(30):
        obs, _ = raw.reset(seed=5000 + s)
        steps = 0
        while True:
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(nobs, deterministic=True)
            obs, _, term, trunc, _ = raw.step(int(a))
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    a = np.array(lengths, dtype=float)
    print("sorted:", [int(v) for v in np.sort(a)])
    print(f"n={len(a)} mean={a.mean():.1f} median={np.median(a):.1f} "
          f"iqm={iqm(a):.1f} std={a.std():.1f} min={a.min():.0f} max={a.max():.0f}")
    print(f"frac_at_cap(1800)={float((a >= 1800).mean()):.3f} "
          f"frac_below_bestconst(746)={float((a < 746.3).mean()):.3f}")
    print(f"beats_bar_930.27_by_mean={a.mean() > 930.27} "
          f"by_median={np.median(a) > 930.27} by_iqm={iqm(a) > 930.27}")


if __name__ == "__main__":
    main()
