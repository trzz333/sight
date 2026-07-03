"""Phase N / C3 - reward-ranked elite behavioral cloning (self-imitation).

Critic-free Cross-Entropy-Method trained-policy variant. Conceptual
ancestor: SIL (Oh 2018, arXiv 1806.05635) and the Cross-Entropy Method
(De Boer 2005; Lapan's CEM-on-CartPole). This is NOT canonical SIL (no
A2C/PPO base, no value critic - the exact stack Phase M / Phase L already
closed NEGATIVE) and NOT CEM-over-weights (a third black-box weight search,
same family as C1/C2, forbidden by method-fails-twice).

Mechanism (structurally distinct from C1/C2 and M/L):
  1. Roll the CURRENT policy STOCHASTICALLY (softmax sample + small epsilon)
     across a batch of B rotated training seeds. Stochastic rollout is
     load-bearing: argmax rollout + BC-on-own-argmax is an immediate fixed
     point (zero learning signal). Sampling lets some trajectories survive
     longer than the argmax would, and BC pulls the greedy policy toward
     those better-than-typical action choices.
  2. Keep the agent's OWN top-k percent episodes by return (episode length;
     reward_shaping='none' so return == survival steps). No external,
     scripted, or human demos - this is what keeps C3 from-scratch (a clear
     counts as mission success where BC 1737.3 did not).
  3. BC (supervised cross-entropy) the (state, sampled-action) pairs from the
     elite episodes directly into the SB3 actor. Warm-start each iteration
     (the policy is the CEM iterate carried forward).
  4. Repeat.

Wall diagnosis attacked: C1/C2 selected WEIGHT VECTORS on ~2 noisy seeds per
generation (the diagnosed wall). C3 selects EPISODES (dense per-state targets)
and can afford a large seed batch per iteration because each iteration is one
BC fit, not a population eval. B defaults to 48 (24x the 2-seed selection).

Gate-native by construction: reuses c1_es_train's build_policy / flatten_actor
/ predict_action / worker pool, so the emitted best_actor_vec.npy (5059 actor
params, raw obs, argmax rule) runs tools\\c1_es_eval.py UNCHANGED. The value
net is never touched (critic-free).

THROUGHPUT SMOKE-TEST is mandatory before any multi-hour run: --smoke times
ONE realistic iteration (B stochastic rollouts + one BC fit) and reports
iters/hour, then exits without training. Do not remove that gate.

Outputs under --out:
  best_actor_vec.npy   dev-best policy actor vector (noise-robust selection
                       on a fixed dev seed band, NOT the sealed held-out gate)
  best_final_vec.npy   final-iteration policy actor vector (2nd gate eval,
                       mirrors C1/C2 dual-vector discipline)
  c3_history.json      per-iteration elite/dev metrics
  c3_report.json       run summary
  c3_state.pt          resumable checkpoint (policy + optim + bookkeeping)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Same detached-launch user-site fix as c1_es_train (WMI-detached processes
# lose APPDATA and never auto-discover user-site where torch/numpy live).
_USER_SITE = r"C:\Users\maste\AppData\Roaming\Python\Python314\site-packages"
if _USER_SITE not in sys.path and Path(_USER_SITE).is_dir():
    sys.path.insert(0, _USER_SITE)

import numpy as np  # noqa: E402

from c1_es_train import (  # noqa: E402
    ACTION_NAMES,
    DEFAULT_EXE,
    DEFAULT_PROJECT,
    MAX_STEPS,
    SURVIVAL_BAR,
    build_policy,
    build_worker_pool,
    flatten_actor,
    load_actor,
    predict_action,
)


# ---------------------------------------------------------------------------
# Actor training surface. build_policy() returns an SB3 ActorCriticPolicy with
# grad globally disabled and eval mode; C3 re-enables grad on the ACTOR ONLY.
# ---------------------------------------------------------------------------

def build_trainable_policy():
    """Gate-native policy with grad enabled on actor params only (critic-free).

    Returns (pol, actor_keys, actor_params, numel). actor_params is the list
    of torch tensors the BC optimizer updates; value-net params keep
    requires_grad=False so the critic is never fit.
    """
    import torch
    pol, actor_keys, numel = build_policy()
    torch.set_grad_enabled(True)  # build_policy disabled it globally
    actor_key_set = set(actor_keys)
    actor_params = []
    for name, p in pol.named_parameters():
        want = name in actor_key_set
        p.requires_grad_(want)
        if want:
            actor_params.append(p)
    assert numel == 5059, f"actor param drift: {numel}"
    return pol, actor_keys, actor_params, numel


def actor_logits(pol, x):
    """Grad-connected forward to the 3 action logits (same path as predict_action)."""
    feat = pol.extract_features(x)
    latent_pi = pol.mlp_extractor.forward_actor(feat)
    return pol.action_net(latent_pi)


def sample_action(pol, obs, *, temperature: float, epsilon: float, rng) -> int:
    """Stochastic action: softmax(logits/T) with an epsilon uniform mix.

    epsilon keeps all three actions on a small floor in the rollout so the
    elite set can contain lateral moves early (honest epsilon-greedy on the
    agent's own policy; not an external demo, not a gate-targeting term)."""
    import torch
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        logits = actor_logits(pol, x).reshape(-1).cpu().numpy().astype(np.float64)
    logits = logits / max(temperature, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    p = (1.0 - epsilon) * p + epsilon * (np.ones(3) / 3.0)
    p /= p.sum()
    return int(rng.choice(3, p=p))


# ---------------------------------------------------------------------------
# Rollout collection across the persistent worker pool.
# ---------------------------------------------------------------------------

def _train_seeds(iteration: int, b: int) -> list[int]:
    """B distinct training seeds for an iteration, disjoint from held-out
    1000-1009. base = 200000 + iteration*1000 (B<1000 => no cross-iter overlap)."""
    base = 200_000 + iteration * 1000
    return [base + j for j in range(b)]


def collect_rollouts(pol, vec_env, n_workers: int, seeds: list[int], *,
                     temperature: float, epsilon: float, rng, record: bool) -> list[dict]:
    """Roll `pol` stochastically over `seeds`, packed n_workers per wave.

    Each wave resets n_workers envs on distinct seeds, then lockstep-steps
    until every worker's FIRST episode ends (SB3 auto-resets finished envs;
    a per-worker done_mask freezes accounting so only the first episode
    counts, exactly as c1 evaluate_batch does). Returns one dict per seed:
    {seed, length, action_counts, states?, actions?}. states/actions filled
    only when record=True (elite rollouts); dev eval passes record=False."""
    results: list[dict | None] = [None] * len(seeds)
    for w0 in range(0, len(seeds), n_workers):
        chunk = list(range(w0, min(w0 + n_workers, len(seeds))))
        m = len(chunk)
        obs = {}
        for k, gi in enumerate(chunk):
            res = vec_env.env_method("reset", seed=int(seeds[gi]), indices=[k])
            obs_k, _info = res[0]
            obs[k] = np.asarray(obs_k, dtype=np.float32)
        lengths = np.zeros(m, dtype=np.int64)
        counts = np.zeros((m, 3), dtype=np.int64)
        done_mask = np.zeros(m, dtype=bool)
        st_buf = [[] for _ in range(m)]
        ac_buf = [[] for _ in range(m)]
        step = 0
        while step < MAX_STEPS and not done_mask.all():
            acts = np.ones(n_workers, dtype=np.int64)  # full width for vec_env
            for k in range(m):
                if done_mask[k]:
                    continue
                a = sample_action(pol, obs[k], temperature=temperature,
                                  epsilon=epsilon, rng=rng)
                acts[k] = a
                counts[k, a] += 1
                if record:
                    st_buf[k].append(obs[k].copy())
                    ac_buf[k].append(a)
            new_obs, _rews, dones, _infos = vec_env.step(acts)
            step += 1
            for k in range(m):
                if not done_mask[k]:
                    lengths[k] += 1
                    obs[k] = np.asarray(new_obs[k], dtype=np.float32)
                    if bool(np.asarray(dones)[k]):
                        done_mask[k] = True
        for k, gi in enumerate(chunk):
            d = {"seed": int(seeds[gi]), "length": int(lengths[k]),
                 "action_counts": [int(counts[k, a]) for a in (0, 1, 2)]}
            if record:
                d["states"] = (np.asarray(st_buf[k], dtype=np.float32)
                               if st_buf[k] else np.zeros((0, 10), np.float32))
                d["actions"] = np.asarray(ac_buf[k], dtype=np.int64)
            results[gi] = d
    return [r for r in results if r is not None]


def dev_eval(pol, vec_env, n_workers: int, dev_seeds: list[int]) -> dict:
    """Deterministic ARGMAX roll on the fixed dev band (best-vector selection,
    NOT the sealed gate). Mirrors the gate's action rule (predict_action) so
    dev-best tracks the metric the gate will score. Returns mean length,
    aggregate action fractions, and the gate-style pass flag on dev seeds."""
    lengths: list[int] = []
    tot = np.zeros(3, dtype=np.int64)
    for w0 in range(0, len(dev_seeds), n_workers):
        chunk = list(range(w0, min(w0 + n_workers, len(dev_seeds))))
        m = len(chunk)
        obs = {}
        for k, gi in enumerate(chunk):
            res = vec_env.env_method("reset", seed=int(dev_seeds[gi]), indices=[k])
            obs_k, _info = res[0]
            obs[k] = np.asarray(obs_k, dtype=np.float32)
        ln = np.zeros(m, dtype=np.int64)
        done_mask = np.zeros(m, dtype=bool)
        cnt = np.zeros((m, 3), dtype=np.int64)
        step = 0
        while step < MAX_STEPS and not done_mask.all():
            acts = np.ones(n_workers, dtype=np.int64)
            for k in range(m):
                if done_mask[k]:
                    continue
                a = predict_action(pol, obs[k])
                acts[k] = a
                cnt[k, a] += 1
            new_obs, _rews, dones, _infos = vec_env.step(acts)
            step += 1
            for k in range(m):
                if not done_mask[k]:
                    ln[k] += 1
                    obs[k] = np.asarray(new_obs[k], dtype=np.float32)
                    if bool(np.asarray(dones)[k]):
                        done_mask[k] = True
        for k in range(m):
            lengths.append(int(ln[k]))
            tot += cnt[k]
    total = int(tot.sum()) or 1
    frac = {ACTION_NAMES[a]: float(tot[a]) / total for a in (0, 1, 2)}
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    max_frac = max(frac.values())
    passed = (mean_len >= SURVIVAL_BAR and frac["left"] >= 0.03
              and frac["right"] >= 0.03 and max_frac < 0.97)
    return {"mean_length": mean_len,
            "min_length": float(min(lengths)) if lengths else 0.0,
            "max_length": float(max(lengths)) if lengths else 0.0,
            "action_fractions": frac, "max_action_fraction": max_frac,
            "dev_gate_pass": bool(passed),
            "per_seed_length": {str(dev_seeds[i]): lengths[i]
                                for i in range(len(lengths))}}


# ---------------------------------------------------------------------------
# Elite selection + BC fit into the SB3 actor.
# ---------------------------------------------------------------------------

def select_elites(episodes: list[dict], elite_frac: float) -> tuple[list[dict], float]:
    """Top elite_frac episodes by length (return). Returns (elites, threshold)."""
    ordered = sorted(episodes, key=lambda e: e["length"], reverse=True)
    n_elite = max(1, int(round(len(ordered) * elite_frac)))
    elites = ordered[:n_elite]
    thresh = float(elites[-1]["length"]) if elites else 0.0
    return elites, thresh


def bc_fit(pol, actor_params, elites: list[dict], *, epochs: int,
           batch_size: int, lr: float, seed: int) -> dict:
    """Supervised cross-entropy of elite (state, sampled-action) into the actor.

    Warm-starts from the current policy (the CEM iterate). Class-weighted CE
    keeps the two lateral actions from being swamped by 'stay'. Raw obs (no
    normalization) so the trained actor stays gate-native."""
    import torch
    import torch.nn as nn
    Xs = [e["states"] for e in elites if len(e["states"]) > 0]
    Ys = [e["actions"] for e in elites if len(e["actions"]) > 0]
    if not Xs:
        return {"n_samples": 0, "skipped": True}
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(Ys, axis=0).astype(np.int64)
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    class_w = torch.tensor(w, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(actor_params, lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    pol.set_training_mode(True)
    last = 0.0
    for _ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            logits = actor_logits(pol, Xt[b])
            loss = lossf(logits, yt[b])
            loss.backward()
            opt.step()
            last = float(loss.detach())
    pol.set_training_mode(False)
    return {"n_samples": int(n), "final_batch_loss": round(last, 4),
            "elite_action_counts_LSR": counts.astype(int).tolist(),
            "skipped": False}


# ---------------------------------------------------------------------------
# Iteration loop.
# ---------------------------------------------------------------------------

def run_c3(args, vec_env, out: Path) -> dict:
    import torch
    pol, actor_keys, actor_params, numel = build_trainable_policy()
    rng = np.random.default_rng(args.seed)
    dev_seeds = [300_000 + j for j in range(args.dev_seeds)]

    history = []
    best_dev = -1.0
    best_vec = flatten_actor(pol, actor_keys)
    best_iter = -1
    t0 = time.time()
    launch_t0 = time.time()

    for it in range(args.iters):
        if args.max_wall_s > 0 and (time.time() - launch_t0) >= args.max_wall_s:
            print(json.dumps({"wall_budget_stop_s": round(args.max_wall_s, 1),
                              "stopped_before_iter": it}), flush=True)
            break
        seeds = _train_seeds(it, args.batch)
        eps = collect_rollouts(pol, vec_env, args.n_workers, seeds,
                               temperature=args.temperature, epsilon=args.epsilon,
                               rng=rng, record=True)
        elites, thresh = select_elites(eps, args.elite_frac)
        roll_mean = float(np.mean([e["length"] for e in eps]))
        elite_mean = float(np.mean([e["length"] for e in elites]))
        fit = bc_fit(pol, actor_params, elites, epochs=args.bc_epochs,
                     batch_size=args.batch_size, lr=args.lr, seed=args.seed + it)
        dev = dev_eval(pol, vec_env, args.n_workers, dev_seeds)
        # dev-best selection is noise-robust (fixed band, argmax == gate rule).
        if dev["mean_length"] > best_dev:
            best_dev = dev["mean_length"]
            best_vec = flatten_actor(pol, actor_keys)
            best_iter = it
        rec = {"iter": it, "rollout_mean_length": round(roll_mean, 1),
               "elite_threshold": round(thresh, 1),
               "elite_mean_length": round(elite_mean, 1),
               "bc": fit,
               "dev_mean_length": round(dev["mean_length"], 1),
               "dev_action_fractions": dev["action_fractions"],
               "dev_max_frac": round(dev["max_action_fraction"], 3),
               "dev_gate_pass": dev["dev_gate_pass"],
               "running_best_dev": round(best_dev, 1),
               "elapsed_seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        # Persist incrementally so a kill mid-run still yields best + final.
        np.save(str(out / "best_actor_vec.npy"), best_vec)
        np.save(str(out / "best_final_vec.npy"), flatten_actor(pol, actor_keys))
        with (out / "c3_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        torch.save({"policy_state": pol.state_dict(), "iter": it + 1,
                    "best_dev": best_dev, "best_iter": best_iter,
                    "best_vec": best_vec}, out / "c3_state.pt")

    np.save(str(out / "best_actor_vec.npy"), best_vec)
    np.save(str(out / "best_final_vec.npy"), flatten_actor(pol, actor_keys))
    return {"mode": "c3", "iters_run": len(history), "best_iter": best_iter,
            "best_dev_mean_length": round(best_dev, 1),
            "elapsed_seconds": round(time.time() - t0, 1),
            "note": "best_actor_vec = dev-best; best_final_vec = last iter"}


def run_smoke(args, vec_env, out: Path) -> dict:
    """Time ONE realistic iteration (B stochastic rollouts + one BC fit + dev)."""
    pol, actor_keys, actor_params, numel = build_trainable_policy()
    rng = np.random.default_rng(args.seed)
    dev_seeds = [300_000 + j for j in range(args.dev_seeds)]
    seeds = _train_seeds(0, args.batch)
    t0 = time.time()
    eps = collect_rollouts(pol, vec_env, args.n_workers, seeds,
                           temperature=args.temperature, epsilon=args.epsilon,
                           rng=rng, record=True)
    t_roll = time.time() - t0
    elites, thresh = select_elites(eps, args.elite_frac)
    tb = time.time()
    fit = bc_fit(pol, actor_params, elites, epochs=args.bc_epochs,
                 batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    t_bc = time.time() - tb
    td = time.time()
    dev = dev_eval(pol, vec_env, args.n_workers, dev_seeds)
    t_dev = time.time() - td
    elapsed = time.time() - t0
    return {"mode": "smoke", "n_workers": args.n_workers, "batch": args.batch,
            "elite_frac": args.elite_frac, "n_elites": len(elites),
            "rollout_mean_length": round(float(np.mean([e["length"] for e in eps])), 1),
            "elite_threshold": round(thresh, 1),
            "bc_samples": fit.get("n_samples", 0),
            "secs_rollout": round(t_roll, 1), "secs_bc": round(t_bc, 2),
            "secs_dev": round(t_dev, 1), "secs_per_iter": round(elapsed, 1),
            "iters_per_hour": round(3600.0 / max(elapsed, 1e-9), 2),
            "dev_mean_length": round(dev["mean_length"], 1),
            "dev_action_fractions": dev["action_fractions"]}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="c3_elitebc_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--vec", choices=("dummy", "subproc"), default="subproc")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--batch", type=int, default=48,
                   help="episodes rolled per iteration (elite pool). 24x the "
                        "2-seed selection C1/C2 collapsed on.")
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="uniform action mix in rollout (honest epsilon-greedy)")
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dev-seeds", type=int, default=8,
                   help="fixed dev band size (300000+) for best-vector pick")
    p.add_argument("--max-wall-s", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--exe", default=DEFAULT_EXE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_root = out / "worker_envs"
    pool_root.mkdir(parents=True, exist_ok=True)

    vec_env = build_worker_pool(
        n_workers=args.n_workers, base_seed=args.seed, run_root=pool_root,
        exe=args.exe, project=args.project, vec=args.vec,
    )
    try:
        report = run_smoke(args, vec_env, out) if args.smoke else run_c3(args, vec_env, out)
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    report.update({
        "actor_param_count": 5059,
        "survival_bar": SURVIVAL_BAR,
        "seed": args.seed,
        "python": sys.version.split()[0],
        "sb3": __import__("stable_baselines3").__version__,
        "torch": __import__("torch").__version__,
    })
    with (out / "c3_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
