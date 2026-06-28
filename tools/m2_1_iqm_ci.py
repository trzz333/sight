# M2.1 IQM + 95% stratified-bootstrap CI (Agarwal et al. 2021).
# IQM primitive = scipy.stats.trim_mean(x, 0.25), identical to rliable.metrics.aggregate_iqm.
# Stratified bootstrap: resample episodes with replacement WITHIN each seed-stratum,
# pool, recompute IQM. 95% CI = percentile method. rliable not installed on the Py3.14
# global interp; this reproduces its definitions without perturbing the M-phase env.
import json
import numpy as np
from scipy.stats import trim_mean

P = r"C:\Projects\Sight\runs\phase_m\m2_1_eval3\m2_eval_summary.json"
d = json.load(open(P))
bar = float(d["survival_bar"])
labels = [m["label"] for m in d["per_model"]]
strata = [[float(v) for v in m["per_seed_length"].values()] for m in d["per_model"]]
arr = np.array(strata, dtype=float)  # (n_seeds, n_eval_episodes) = (3, 10)
pooled = arr.reshape(-1)

# --- unit-check the IQM primitive against a known case ---
assert abs(trim_mean(np.arange(1, 101), 0.25) - 50.5) < 1e-9, "IQM primitive mismatch"

def iqm(x):
    return float(trim_mean(x, 0.25))

iqm_point = iqm(pooled)

REPS = 50000
rng = np.random.default_rng(0)
n_seeds, n_ep = arr.shape
boot = np.empty(REPS)
for i in range(REPS):
    sample = np.empty_like(arr)
    for s in range(n_seeds):
        idx = rng.integers(0, n_ep, size=n_ep)  # stratified: within-seed resample
        sample[s] = arr[s, idx]
    boot[i] = iqm(sample.reshape(-1))

lo, hi = np.percentile(boot, [2.5, 97.5])
p_exceed = float((boot >= bar).mean())

out = {
    "bar": bar,
    "labels": labels,
    "per_seed_mean": [round(float(x), 1) for x in arr.mean(1)],
    "grand_mean": round(float(pooled.mean()), 2),
    "n_seeds": n_seeds,
    "n_eval_episodes_per_seed": n_ep,
    "n_pooled": int(pooled.size),
    "iqm": round(iqm_point, 2),
    "ci95_lo": round(float(lo), 2),
    "ci95_hi": round(float(hi), 2),
    "bootstrap_reps": REPS,
    "bootstrap_rng_seed": 0,
    "method": "scipy.stats.trim_mean(.,0.25); stratified-by-seed percentile bootstrap",
    "frac_bootstrap_iqm_ge_bar": round(p_exceed, 4),
}
with open(r"C:\Projects\Sight\runs\phase_m\m2_1_eval3\iqm_ci_result.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
