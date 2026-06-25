"""
k5_8_reliability_report.py

Reliability report for the K5.8 NoisyNet QR-DQN from-scratch multi-seed sweep on
Signal Dodge. Reports the across-seed DISTRIBUTION, not a single point estimate,
because dependability in deep RL is a property measured across training seeds
(Henderson et al. 2018, "Deep RL that Matters"; Agarwal et al. 2021, "Deep RL at
the Edge of the Statistical Precipice", NeurIPS).

Method is adopted from Agarwal et al. 2021 (the rliable framework): interquartile
mean (IQM), bootstrap percentile confidence intervals, and a performance profile.
The bootstrap CI uses scipy.stats.bootstrap (an off-the-shelf primitive, no new
deps) rather than a hand-rolled resampling loop.

rliable itself is NOT adopted wholesale. It pins arch<8.0, and only arch>=8.0
ships a Python 3.14 (cp314) Windows wheel; installing rliable therefore drags in
arch 7.2.0 as an sdist that needs an MSVC/Cython source build absent on this
machine (verified 2026-06-25 via pip --dry-run). For a single task (Signal
Dodge), stratified bootstrap reduces to ordinary resampling of runs with
replacement, so scipy's bootstrap is faithful to the single-task case and
rliable's cross-task machinery would be overkill regardless.

Score per seed = mean held-out episode length (the eval metric, bar 930.27).
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

BAR = 930.27
BEST_CONSTANT = 845.7
BC = 1737.3
PPO_FT = 1710.5


def collect_scores(run_glob):
    rows = []
    for d in sorted(glob.glob(run_glob)):
        rep = os.path.join(d, "eval_inenv", "noisy_eval_inenv_report.json")
        if not os.path.exists(rep):
            continue
        j = json.load(open(rep))
        rows.append({
            "run": os.path.basename(d),
            "mean_ep_len": float(j["mean_episode_length"]),
            "nondegenerate": bool(j["nondegenerate"]),
            "frac_R": float(j["pooled_action_fractions_LSR"][2]),
            "verdict": j["verdict"],
        })
    return rows


def iqm(x):
    # interquartile mean = mean of middle 50 percent of pooled scores
    return float(stats.trim_mean(x, 0.25))


def _iqm_axis(x, axis):
    # axis-aware IQM so scipy.stats.bootstrap can vectorize the resampling
    return stats.trim_mean(x, 0.25, axis=axis)


def _mean_axis(x, axis):
    return np.mean(x, axis=axis)


def boot_ci(x, stat_axis_fn, reps=10000, alpha=0.05, seed=0):
    # Off-the-shelf bootstrap CI (scipy.stats.bootstrap) in place of a hand-rolled
    # resample loop. Percentile method: matches rliable's single-task CI and stays
    # stable at small n (BCa's jackknife acceleration is unreliable at n=5).
    res = stats.bootstrap(
        (np.asarray(x, float),),
        stat_axis_fn,
        n_resamples=reps,
        confidence_level=1.0 - alpha,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high)


def perf_profile(x, taus):
    x = np.asarray(x, float)
    return [float(np.mean(x >= t)) for t in taus]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=r"C:\Projects\Sight\runs\phase_k\k5_8_noisy_qrdqn*")
    ap.add_argument("--out", default=r"C:\Projects\Sight\runs\phase_k\k5_8_reliability_report.json")
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()

    rows = collect_scores(args.glob)
    if not rows:
        print("NO_RUNS_FOUND for glob", args.glob)
        return
    scores = np.array([r["mean_ep_len"] for r in rows])
    n = len(scores)

    iqm_pt = iqm(scores)
    mean_pt = float(np.mean(scores))
    median_pt = float(np.median(scores))
    iqm_lo, iqm_hi = boot_ci(scores, _iqm_axis, reps=args.reps)
    mean_lo, mean_hi = boot_ci(scores, _mean_axis, reps=args.reps)

    frac_above_bar = float(np.mean(scores >= BAR))
    frac_above_best_const = float(np.mean(scores >= BEST_CONSTANT))
    frac_nondegenerate = float(np.mean([r["nondegenerate"] for r in rows]))

    taus = [600, 700, 800, BEST_CONSTANT, BAR, 1000, 1100]
    profile = perf_profile(scores, taus)

    report = {
        "n_seeds": n,
        "bar": BAR, "best_constant": BEST_CONSTANT, "bc": BC, "ppo_ft": PPO_FT,
        "per_seed": rows,
        "scores": [float(s) for s in scores],
        "iqm": iqm_pt, "iqm_ci95": [iqm_lo, iqm_hi],
        "mean": mean_pt, "mean_ci95": [mean_lo, mean_hi],
        "median": median_pt,
        "min": float(np.min(scores)), "max": float(np.max(scores)),
        "frac_above_bar": frac_above_bar,
        "frac_above_best_constant": frac_above_best_const,
        "frac_nondegenerate": frac_nondegenerate,
        "perf_profile_taus": taus,
        "perf_profile_frac_ge_tau": profile,
        "method": "IQM + percentile bootstrap CI (scipy.stats.bootstrap, 10k resamples) + performance profile; method per Agarwal et al. 2021 (rliable), single-task",
    }
    json.dump(report, open(args.out, "w"), indent=2)

    print(f"N_SEEDS={n}")
    print(f"IQM={iqm_pt:.1f}  95% CI [{iqm_lo:.1f}, {iqm_hi:.1f}]  bar={BAR}")
    print(f"MEAN={mean_pt:.1f}  95% CI [{mean_lo:.1f}, {mean_hi:.1f}]")
    print(f"MEDIAN={median_pt:.1f}  MIN={np.min(scores):.1f}  MAX={np.max(scores):.1f}")
    print(f"FRAC_ABOVE_BAR={frac_above_bar:.2f}  FRAC_ABOVE_BEST_CONST={frac_above_best_const:.2f}  FRAC_NONDEGENERATE={frac_nondegenerate:.2f}")
    print("PERF_PROFILE (frac of seeds with score >= tau):")
    for t, p in zip(taus, profile):
        print(f"  tau={t:>7.1f}  frac={p:.2f}")
    print("REPORT_WRITTEN", args.out)


if __name__ == "__main__":
    main()
