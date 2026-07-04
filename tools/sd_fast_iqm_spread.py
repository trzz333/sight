"""Reload the three M2.1-recipe 5M replica seeds and dump the per-seed held-out
IQM spread. Greedy eval on held-out seeds 5000-5029, obs normalized through each
run's saved VecNormalize stats (training=False). Reports mean/median/IQM so the
3-seed reliability spread is anchored to reload-eval, not to the summary means
(which the run wrote as mean/std only)."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sight_agent.rl.sd_fast import SignalDodgeFast

OUT = r"C:\Projects\Sight\runs\sd_fast"
RUNS = ["sd_fast_m21_s0_5M", "sd_fast_m21_s1_5M", "sd_fast_m21_s2_5M"]
BAR = 930.27
BEST_CONST = 746.3


def make_env():
    return SignalDodgeFast()


def iqm(x: np.ndarray) -> float:
    x = np.sort(x)
    lo, hi = int(len(x) * 0.25), int(np.ceil(len(x) * 0.75))
    return float(x[lo:hi].mean())


def eval_run(run: str):
    model = PPO.load(f"{OUT}\\{run}.zip", device="cpu")
    vn = VecNormalize.load(f"{OUT}\\{run}_vecnormalize.pkl", DummyVecEnv([make_env]))
    vn.training = False
    vn.norm_reward = False
    raw = SignalDodgeFast(max_steps=1800)
    lengths, acts = [], np.zeros(3)
    for s in range(30):
        obs, _ = raw.reset(seed=5000 + s)
        steps = 0
        while True:
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(nobs, deterministic=True)
            ai = int(a)
            acts[ai] += 1
            obs, _, term, trunc, _ = raw.step(ai)
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    a = np.array(lengths, dtype=float)
    fr = acts / acts.sum()
    return a, fr


def main():
    print(f"{'seed':>4} {'mean':>7} {'median':>7} {'IQM':>7} {'std':>6} "
          f"{'maxfrac':>7} {'L/S/R':>18} {'cap%':>5} {'clr930(m/md/iqm)':>18}")
    rows = []
    for i, run in enumerate(RUNS):
        a, fr = eval_run(run)
        m, md, q = a.mean(), float(np.median(a)), iqm(a)
        cap = float((a >= 1800).mean())
        lsr = f"{fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}"
        clr = f"{m>BAR}/{md>BAR}/{q>BAR}"
        rows.append((i, m, md, q, a.std(), fr.max(), lsr, cap, clr, a))
        print(f"{i:>4} {m:>7.1f} {md:>7.1f} {q:>7.1f} {a.std():>6.1f} "
              f"{fr.max():>7.2f} {lsr:>18} {cap*100:>4.0f}% {clr:>18}")
    print()
    iqms = np.array([r[3] for r in rows])
    print(f"IQM spread across seeds: min={iqms.min():.1f} max={iqms.max():.1f} "
          f"mean={iqms.mean():.1f} range={iqms.max()-iqms.min():.1f}")
    n_clear = sum(1 for r in rows if r[3] > BAR)
    print(f"seeds clearing 930.27 by IQM: {n_clear}/3")
    for r in rows:
        print(f"seed{r[0]} sorted:", [int(v) for v in np.sort(r[9])])


if __name__ == "__main__":
    main()
