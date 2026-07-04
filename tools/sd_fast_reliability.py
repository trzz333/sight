"""sd-fast none-vs-shaped reliability eval (Agarwal 2021 methodology, numpy port).

Reload-evals all 10 replica models greedily on the held-out seed block
5000-5029 (30 episodes/seed), then aggregates per arm with the rliable
(Agarwal et al. 2021, "Deep RL at the Edge of the Statistical Precipice")
reliability trio: interquartile mean (IQM), a percentile bootstrap 95% CI,
and probability of improvement. rliable's C-extension dep (`arch`) will not
build here without MSVC, so the *methodology* is adopted, not the package.

Single training environment, so rliable's stratified (per-task) bootstrap
collapses to seed-level resampling: the resampling unit is the seed, and the
per-arm score vector is the 5 per-seed held-out means.

Eval is reward-agnostic (episode length only), so the shaped and none arms
use an identical harness; only each run's saved VecNormalize obs-stats differ.

Refuses to run until all 10 models are present on disk. Caches per-model
episode-length arrays so re-runs are instant.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sight_agent.rl.sd_fast import SignalDodgeFast  # noqa: E402

OUT = r"C:\Projects\Sight\runs\sd_fast"
CACHE = os.path.join(OUT, "reliability_eval_cache.json")
BAR = 930.27
EVAL_BASE_SEED = 5000
EVAL_N = 30
BOOT = 50000
RNG = np.random.default_rng(12345)

NONE_RUNS = [f"sd_fast_m21_s{s}_5M" for s in range(5)]
SHAPED_RUNS = [f"sd_fast_m21sh_s{s}_5M" for s in range(5)]
CURR_RUNS = [f"sd_fast_m21curr_s{s}_5M" for s in range(5)]
ALL_RUNS = NONE_RUNS + SHAPED_RUNS + CURR_RUNS


def make_env():
    return SignalDodgeFast()


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% after sorting."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def eval_run(run: str) -> list[int]:
    """Greedy reload-eval on held-out seeds; return the 30 episode lengths."""
    model = PPO.load(f"{OUT}\\{run}.zip", device="cpu")
    vn = VecNormalize.load(f"{OUT}\\{run}_vecnormalize.pkl", DummyVecEnv([make_env]))
    vn.training = False
    vn.norm_reward = False
    raw = SignalDodgeFast(max_steps=1800)
    lengths = []
    for s in range(EVAL_N):
        obs, _ = raw.reset(seed=EVAL_BASE_SEED + s)
        steps = 0
        while True:
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(nobs, deterministic=True)
            obs, _, term, trunc, _ = raw.step(int(a))
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    return lengths


def load_or_eval() -> dict[str, np.ndarray]:
    missing = [r for r in ALL_RUNS
               if not (os.path.exists(f"{OUT}\\{r}.zip")
                       and os.path.exists(f"{OUT}\\{r}_vecnormalize.pkl"))]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} model(s) not on disk yet: {missing}")
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    out = {}
    for r in ALL_RUNS:
        if r in cache:
            out[r] = np.array(cache[r], dtype=float)
        else:
            print(f"  eval {r} ...", flush=True)
            lens = eval_run(r)
            cache[r] = lens
            out[r] = np.array(lens, dtype=float)
            json.dump(cache, open(CACHE, "w"), indent=0)
    return out


def prob_of_improvement(shaped_eps: list[np.ndarray], none_eps: list[np.ndarray]) -> float:
    """rliable POI: mean over run-pairs of P(shaped_run beats none_run),
    each pair scored by Mann-Whitney (ties count 0.5)."""
    vals = []
    for a in shaped_eps:
        for b in none_eps:
            gt = (a[:, None] > b[None, :]).sum()
            eq = (a[:, None] == b[None, :]).sum()
            vals.append((gt + 0.5 * eq) / (a.size * b.size))
    return float(np.mean(vals))


def arm_stats(runs: list[str], data: dict[str, np.ndarray]):
    eps = [data[r] for r in runs]                 # list of 30-length arrays
    seed_means = np.array([e.mean() for e in eps])  # per-seed score (rliable run score)
    point_iqm = iqm(seed_means)
    # percentile bootstrap over seeds (resample the 5 seed-scores w/ replacement)
    boot = np.array([iqm(RNG.choice(seed_means, size=len(seed_means), replace=True))
                     for _ in range(BOOT)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    clears = int((seed_means > BAR).sum())
    return dict(eps=eps, seed_means=seed_means, iqm=point_iqm, ci=ci,
                boot=boot, clears=clears, pooled_mean=float(np.mean(seed_means)))


def main():
    data = load_or_eval()
    none = arm_stats(NONE_RUNS, data)
    shaped = arm_stats(SHAPED_RUNS, data)
    curr = arm_stats(CURR_RUNS, data)

    print("\n=== per-seed held-out means (greedy, seeds 5000-5029) ===")
    print(f"{'seed':>4} {'none':>9} {'shaped':>9} {'curr':>9}")
    for i in range(5):
        print(f"{i:>4} {none['seed_means'][i]:>9.1f} {shaped['seed_means'][i]:>9.1f} "
              f"{curr['seed_means'][i]:>9.1f}")

    print("\n=== per-arm reliability (Agarwal 2021, seed-level bootstrap) ===")
    for name, arm in (("none", none), ("shaped", shaped), ("curr", curr)):
        print(f"{name:>7}: IQM={arm['iqm']:7.1f}  "
              f"95%CI=[{arm['ci'][0]:.1f}, {arm['ci'][1]:.1f}]  "
              f"clears930={arm['clears']}/5  poolMean={arm['pooled_mean']:.1f}")

    def compare(label, exp, base):
        p_iqm = float((exp['boot'] > base['boot']).mean())
        poi = prob_of_improvement(exp['eps'], base['eps'])
        diff = exp['iqm'] - base['iqm']
        print(f"\n=== comparison: {label} ===")
        print(f"IQM({label.split(' vs ')[0]}) - IQM({label.split(' vs ')[1]}) = {diff:+.1f}")
        print(f"P(IQM_exp > IQM_base)  [paired seed bootstrap] = {p_iqm:.3f}")
        print(f"P(improvement) exp>base [rliable POI]          = {poi:.3f}")
        return diff, p_iqm, poi

    # Headline: the curriculum lever is this session's question.
    c_diff, c_p_iqm, c_poi = compare("curr vs none", curr, none)
    curr_better = (c_diff > 0) and (c_p_iqm >= 0.975) and (curr['clears'] >= none['clears'])
    print("\n=== verdict: curriculum ===")
    if curr_better:
        print("CURR > NONE reliably (IQM up, P(IQM)>=0.975, clear-rate >=). "
              "First reproducible from-scratch clear on sd-fast; port the "
              "curriculum recipe to a Godot 5M eval-of-record (bar 930.27).")
    else:
        print("CURR NOT reliably better than NONE at 5-seed rliable. Do not "
              "port yet; inspect per-seed spread and the anneal schedule.")

    # Secondary, kept for the record: shaped vs none (PBRS).
    compare("shaped vs none", shaped, none)


if __name__ == "__main__":
    main()
