"""Phase N / C1 - Evolution Strategies (separable CMA-ES) on Signal Dodge.

Black-box weight optimization of the SB3 MlpPolicy actor (10->64->64->3,
5059 params). No critic: sidesteps the entire M2/M2.1 critic-fitting saga.
Each candidate weight-vector is evaluated over full episodes, so the
perturbation is consistent within an episode (the K5.8 exploration-
structure lever that produced the only from-scratch seed to ever clear the
bar).

Optimizer: pycma CMA_diagonal=True (separable, O(d) not O(d^2)/O(d^3);
full-covariance is infeasible at d~5k). popsize defaults to cma's
4+3ln(d) ~= 29.

Policy binding: actor-only params (mlp_extractor.policy_net.* +
action_net.*) flattened to a vector and loaded per candidate. Forward
pass argmax over 3 logits == the same deterministic action rule the M2
eval uses, so ES policies are gate-comparable apples-to-apples.

Fitness: mean episode length over FRESH training seeds per candidate
(held-out 1000-1009 stay sealed for the final gate). CMA minimizes, so
reported fitness to cma is NEGATIVE mean length.

Throughput: Godot subprocess spawn is the expensive part, so a pool of
persistent worker envs is created ONCE and reused across all candidates
and generations. Each rollout loads a candidate's weights into a local
policy and steps one env. Workers run via SubprocVecEnv (overlaps the TCP
round-trips across processes); the ES loop dispatches one rollout per
worker per wave.

THROUGHPUT SMOKE-TEST is mandatory before any multi-hour run: --smoke
measures episodes/sec and estimates generations-to-signal, then exits
without optimizing. Do not remove that gate.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Detached-launch env fix: a WMI-detached process runs with a minimal
# environment (no APPDATA), so the user-site dir is not auto-discovered and
# PYTHONPATH from the batch file does not reach it. cma/numpy/torch all live
# in user-site on this machine, so inject it onto sys.path explicitly before
# importing any third-party package. Foreground imports already work; this
# only adds the path when missing.
_USER_SITE = r"C:\Users\maste\AppData\Roaming\Python\Python314\site-packages"
if _USER_SITE not in sys.path and Path(_USER_SITE).is_dir():
    sys.path.insert(0, _USER_SITE)

import numpy as np  # noqa: E402

DEFAULT_EXE = (
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64.exe"
)
DEFAULT_PROJECT = str(_REPO_ROOT / "games" / "signal-dodge")
SURVIVAL_BAR = 930.27
MAX_STEPS = 1800
ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}

# Actor-only parameter keys (verified by c1_policy_probe.py: 5059 params).
ACTOR_PREFIXES = ("mlp_extractor.policy_net", "action_net")


def _alloc_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def build_policy():
    """SB3 ActorCriticPolicy with Signal Dodge spaces, net_arch pi/vf=[64,64].

    Returns (policy, actor_keys, total_actor_numel). Policy is set to eval
    mode; ES never backprops. CPU only.
    """
    import gymnasium as gym
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    def lr_sched(_progress):
        return 3e-4

    pol = ActorCriticPolicy(
        obs_space, act_space, lr_sched,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    pol.set_training_mode(False)
    torch.set_grad_enabled(False)

    actor_keys = [
        name for name, _ in pol.named_parameters()
        if name.startswith(ACTOR_PREFIXES)
    ]
    numel = sum(
        p.numel() for n, p in pol.named_parameters() if n in actor_keys
    )
    return pol, actor_keys, numel


def build_policies(n: int):
    """n independent policy instances sharing the same architecture.

    Candidate-packing evaluates n DIFFERENT candidates in one lockstep wave
    (one per worker env), so each worker needs its own loaded weights. n
    policy objects (~10k params each) is trivial memory and lets us load
    each candidate once per wave instead of per step.
    """
    pols = []
    keys = None
    numel = None
    for _ in range(n):
        p, k, m = build_policy()
        pols.append(p)
        keys, numel = k, m
    return pols, keys, numel


def flatten_actor(pol, actor_keys) -> np.ndarray:
    """Concatenate actor params into a 1-D float64 vector (cma wants float)."""
    import torch
    sd = pol.state_dict()
    parts = [sd[k].detach().cpu().numpy().ravel() for k in actor_keys]
    return np.concatenate(parts).astype(np.float64)


def load_actor(pol, actor_keys, vec: np.ndarray) -> None:
    """Write a flat vector back into the actor params, in actor_keys order."""
    import torch
    sd = pol.state_dict()
    i = 0
    for k in actor_keys:
        t = sd[k]
        n = t.numel()
        chunk = np.asarray(vec[i:i + n], dtype=np.float32).reshape(tuple(t.shape))
        sd[k].copy_(torch.from_numpy(chunk))
        i += n
    # state_dict() returns references to the live tensors for SB3 policies,
    # so copy_ above mutates the policy in place. No load_state_dict needed.


def predict_action(pol, obs: np.ndarray) -> int:
    """Deterministic argmax over the 3 action logits (matches M2 eval)."""
    import torch
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
    feat = pol.extract_features(x)
    latent_pi = pol.mlp_extractor.forward_actor(feat)
    logits = pol.action_net(latent_pi)
    return int(torch.argmax(logits, dim=1).item())


def _build_godot_env(*, port: int, seed: int, run_dir: str, exe: str, project: str):
    """Module-level for SubprocVecEnv pickling under spawn. Mirrors m2."""
    from sight_agent.rl.godot_env import GodotSignalDodgeEnv
    return GodotSignalDodgeEnv(
        godot_executable=exe, project_path=project, run_dir=run_dir,
        max_steps=MAX_STEPS, headless=True, observation_mode="state",
        reward_shaping="none", tcp_port=port, seed=seed,
    )


def build_worker_pool(*, n_workers: int, base_seed: int, run_root: Path,
                      exe: str, project: str, vec: str):
    """One Godot subprocess + one TCP port per worker, reused across candidates.

    Seeds here are placeholders; per-evaluation seeds are set via vec.seed()
    + reset() at each fitness call so a candidate is scored on fresh training
    seeds. vec in {dummy, subproc}.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    fns = []
    for i in range(n_workers):
        port = _alloc_port()
        sub = run_root / f"w{i}"
        sub.mkdir(parents=True, exist_ok=True)
        fns.append(partial(
            _build_godot_env, port=port, seed=base_seed + i,
            run_dir=str(sub), exe=exe, project=project,
        ))
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def evaluate_batch(policies, actor_keys, vec_env, cand_vecs, seed: int) -> list:
    """Score up to n_workers candidates IN ONE WAVE on a shared seed.

    Candidate-packing: cand_vecs[j] is loaded into policies[j] and rolled on
    worker j. All workers reset on the SAME `seed` (common random numbers ->
    fair within-wave ranking, the property CMA needs). Lockstep-step until
    every worker's first episode ends; SB3 auto-resets finished envs but
    done_mask freezes their accounting so only the first episode counts.

    Returns a list aligned to cand_vecs: per-candidate length + action
    fractions on this seed.
    """
    m = len(cand_vecs)
    assert m <= len(policies), f"{m} candidates > {len(policies)} workers"
    for j in range(m):
        load_actor(policies[j], actor_keys, np.asarray(cand_vecs[j]))
    obs_list = []
    for j in range(m):
        res = vec_env.env_method("reset", seed=int(seed), indices=[j])
        obs_j, _info = res[0]
        obs_list.append(np.asarray(obs_j, dtype=np.float32))
    obs = {j: obs_list[j] for j in range(m)}
    lengths = np.zeros(m, dtype=np.int64)
    done_mask = np.zeros(m, dtype=bool)
    counts = np.zeros((m, 3), dtype=np.int64)
    step = 0
    while step < MAX_STEPS and not done_mask.all():
        acts = np.ones(len(policies), dtype=np.int64)  # full-width for vec_env
        for j in range(m):
            if done_mask[j]:
                continue
            a = predict_action(policies[j], obs[j])
            acts[j] = a
            counts[j, a] += 1
        new_obs, _rews, dones, _infos = vec_env.step(acts)
        step += 1
        for j in range(m):
            if not done_mask[j]:
                lengths[j] += 1
                obs[j] = np.asarray(new_obs[j], dtype=np.float32)
                if bool(np.asarray(dones)[j]):
                    done_mask[j] = True
    out = []
    for j in range(m):
        tot = counts[j]
        total = int(tot.sum()) or 1
        frac = {ACTION_NAMES[a]: float(tot[a]) / total for a in (0, 1, 2)}
        out.append({
            "length": int(lengths[j]),
            "action_fractions": frac,
            "max_action_fraction": max(frac.values()),
        })
    return out


def evaluate_generation(policies, actor_keys, vec_env, n_workers: int,
                        cand_vecs, seeds: list[int]) -> list:
    """Score ALL candidates as mean length over `seeds`, packed across workers.

    For each seed (common random numbers across the whole population), the
    candidates are chunked into waves of n_workers and each wave scored by
    evaluate_batch. Per-candidate results are aggregated to mean length and
    summed action counts across seeds. Returns a list aligned to cand_vecs.
    """
    pop = len(cand_vecs)
    acc_len = np.zeros((pop, len(seeds)), dtype=np.float64)
    acc_frac = [np.zeros(3, dtype=np.float64) for _ in range(pop)]
    for si, seed in enumerate(seeds):
        for w0 in range(0, pop, n_workers):
            chunk = cand_vecs[w0:w0 + n_workers]
            res = evaluate_batch(policies, actor_keys, vec_env, chunk, seed)
            for k, r in enumerate(res):
                idx = w0 + k
                acc_len[idx, si] = r["length"]
                fr = r["action_fractions"]
                acc_frac[idx] += np.array(
                    [fr["left"], fr["stay"], fr["right"]], dtype=np.float64
                )
    out = []
    for idx in range(pop):
        mean_len = float(acc_len[idx].mean())
        fsum = acc_frac[idx]
        ftot = float(fsum.sum()) or 1.0
        frac = {
            "left": float(fsum[0] / ftot),
            "stay": float(fsum[1] / ftot),
            "right": float(fsum[2] / ftot),
        }
        out.append({
            "mean_length": mean_len,
            "min_length": float(acc_len[idx].min()),
            "max_length": float(acc_len[idx].max()),
            "per_seed_length": {str(seeds[s]): float(acc_len[idx, s])
                                for s in range(len(seeds))},
            "action_fractions": frac,
            "max_action_fraction": max(frac.values()),
            "fitness": shaped_fitness(mean_len, frac),
        })
    return out


# Diversity floor below the gate's 0.97 collapse line. Past this, fitness is
# discounted hard so CMA is steered away from single-action survivors that
# would fail the eval gate's max(frac)<0.97 clause. Also requires the two
# lateral actions to each clear the gate's 0.03 floor, with margin.
COLLAPSE_SOFT = 0.90
MIN_LATERAL = 0.05  # > gate's 0.03 so the screen optimizes with headroom


def shaped_fitness(mean_len: float, frac: dict) -> float:
    """Length, penalized toward the gate's action-diversity requirements.

    Raw mean_length rewards a single-action policy that survives by luck (seen
    at gen 0: left=1.0 surviving 1203). The eval gate rejects max(frac)>=0.97
    and frac_left/right<0.03. Optimizing raw length steers CMA toward gate-
    FAILING policies. This bends fitness toward gate-PASSING behavior:
      - multiplicative penalty ramping in as max_frac exceeds COLLAPSE_SOFT,
        reaching ~0 at full collapse (1.0);
      - penalty if either lateral action is below MIN_LATERAL.
    A policy that survives long AND moves both directions keeps near-full
    length; a collapsed one is driven down regardless of its lucky length.
    """
    mf = max(frac.values())
    if mf <= COLLAPSE_SOFT:
        collapse_mult = 1.0
    else:
        # 1.0 at COLLAPSE_SOFT, 0.0 at 1.0, linear.
        collapse_mult = max(0.0, (1.0 - mf) / (1.0 - COLLAPSE_SOFT))
    lateral = min(frac.get("left", 0.0), frac.get("right", 0.0))
    if lateral >= MIN_LATERAL:
        lateral_mult = 1.0
    else:
        lateral_mult = lateral / MIN_LATERAL  # 0 at no lateral, 1 at floor
    return float(mean_len) * collapse_mult * lateral_mult


def fresh_training_seeds(gen: int, k: int) -> list[int]:
    """Distinct training seeds for a generation's fitness, disjoint from the
    held-out eval band 1000-1009. Uses a high base so we never collide with
    eval seeds even across thousands of generations.
    """
    base = 200_000 + gen * 97
    return [base + j for j in range(k)]


def run_smoke(args, vec_env, policies, actor_keys, numel) -> dict:
    """Throughput smoke-test: time ONE realistic packed generation.

    Times popsize candidates packed n_workers-per-wave over seeds_per_gen
    seeds. Reports episodes/sec, secs/generation, gens/hour. Does NOT
    optimize. Mandatory gate before any long run.
    """
    rng = np.random.RandomState(args.seed)
    n = args.n_workers
    popsize = args.popsize if args.popsize > 0 else int(4 + 3 * np.log(numel))
    seeds = fresh_training_seeds(gen=0, k=args.seeds_per_gen)
    # Smoke: time ONE realistic generation (popsize candidates, packed
    # n-per-wave, over seeds_per_gen seeds). This is the true cost unit.
    cands = [rng.randn(numel).astype(np.float64) * args.sigma0
             for _ in range(popsize)]
    t0 = time.time()
    res = evaluate_generation(policies, actor_keys, vec_env, n, cands, seeds)
    elapsed = time.time() - t0
    total_eps = popsize * len(seeds)
    eps_per_sec = total_eps / max(elapsed, 1e-9)
    secs_per_gen = elapsed
    gens_per_hour = 3600.0 / max(secs_per_gen, 1e-9)
    mean_len_all = float(np.mean([r["mean_length"] for r in res]))
    best_len = max(r["mean_length"] for r in res)
    return {
        "mode": "smoke",
        "n_workers": n,
        "vec": args.vec,
        "popsize": popsize,
        "seeds_per_gen": args.seeds_per_gen,
        "episodes_this_gen": total_eps,
        "elapsed_seconds": round(elapsed, 2),
        "episodes_per_sec": round(eps_per_sec, 3),
        "secs_per_generation": round(secs_per_gen, 1),
        "generations_per_hour": round(gens_per_hour, 2),
        "random_policy_mean_length": round(mean_len_all, 1),
        "random_policy_best_length": round(best_len, 1),
    }


def run_es(args, vec_env, policies, actor_keys, numel, out: Path) -> dict:
    """Separable CMA-ES optimization loop. Saves best actor vector + report."""
    import cma

    popsize = args.popsize if args.popsize > 0 else int(4 + 3 * np.log(numel))
    x0 = flatten_actor(policies[0], actor_keys)  # SB3 default init as ES mean
    es = cma.CMAEvolutionStrategy(
        list(x0), args.sigma0,
        {"CMA_diagonal": True, "popsize": popsize, "seed": args.seed,
         "maxiter": args.gens, "verbose": -9},
    )
    history = []
    best_fit = -1.0
    best_mean_len = -1.0
    best_vec = x0.copy()
    t0 = time.time()
    gen = 0
    while not es.stop() and gen < args.gens:
        cands = es.ask()
        # Fresh seeds per generation (common random numbers within the gen:
        # all candidates face the same seeds -> fair ranking for CMA).
        seeds = fresh_training_seeds(gen=gen, k=args.seeds_per_gen)
        res = evaluate_generation(policies, actor_keys, vec_env,
                                  args.n_workers, [np.asarray(c) for c in cands],
                                  seeds)
        fitnesses = [-r["fitness"] for r in res]  # cma minimizes shaped fitness
        gi = int(np.argmax([r["fitness"] for r in res]))
        gen_best_fit = res[gi]["fitness"]
        gen_best = res[gi]["mean_length"]
        gen_best_detail = res[gi]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_mean_len = gen_best
            best_vec = np.asarray(cands[gi]).copy()
        es.tell(cands, fitnesses)
        elapsed = time.time() - t0
        rec = {
            "gen": gen,
            "gen_best_fitness": round(gen_best_fit, 1),
            "gen_best_mean_length": round(gen_best, 1),
            "gen_best_action_fractions": gen_best_detail["action_fractions"],
            "gen_best_max_frac": round(gen_best_detail["max_action_fraction"], 3),
            "running_best_mean_length": round(best_mean_len, 1),
            "elapsed_seconds": round(elapsed, 1),
        }
        history.append(rec)
        print(json.dumps(rec))
        # Persist incrementally so a kill mid-run still yields the best vector.
        # Save BOTH the best sampled candidate (noise-biased: argmax of a noisy
        # 2-seed fitness over-selects lucky seeds) and the CMA distribution mean
        # (es.result.xfavorite), which is pycma's noise-robust recommendation
        # and the one the gate eval should trust under noisy fitness.
        np.save(str(out / "best_actor_vec.npy"), best_vec)
        np.save(str(out / "best_mean_vec.npy"), np.asarray(es.result.xfavorite, dtype=np.float64))
        with (out / "es_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        gen += 1

    np.save(str(out / "best_actor_vec.npy"), best_vec)
    np.save(str(out / "best_mean_vec.npy"), np.asarray(es.result.xfavorite, dtype=np.float64))
    load_actor(policies[0], actor_keys, best_vec)
    return {
        "mode": "es",
        "generations_run": gen,
        "popsize": popsize,
        "seeds_per_gen": args.seeds_per_gen,
        "best_mean_length_train_seeds": round(best_mean_len, 1),
        "elapsed_seconds": round(time.time() - t0, 1),
        "cma_stop": dict(es.stop()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="c1_es_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8,
                   help="parallel Godot subprocs = candidates scored per wave")
    p.add_argument("--vec", choices=("dummy", "subproc"), default="subproc")
    p.add_argument("--sigma0", type=float, default=0.1,
                   help="CMA initial step size in weight space")
    p.add_argument("--popsize", type=int, default=0,
                   help="0 = cma default 4+3ln(d) ~= 29")
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--seeds-per-gen", type=int, default=1,
                   help="seeds each candidate is averaged over per generation "
                        "(common random numbers within a gen). 1 = Salimans-"
                        "style single rollout; higher = lower-variance fitness, "
                        "linearly slower.")
    p.add_argument("--smoke", action="store_true",
                   help="throughput smoke-test only; times one packed gen")
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

    # One policy per worker for candidate-packing (n different candidates
    # rolled in one lockstep wave).
    policies, actor_keys, numel = build_policies(args.n_workers)
    assert numel == 5059, f"actor param count drift: {numel}"

    vec_env = build_worker_pool(
        n_workers=args.n_workers, base_seed=args.seed, run_root=pool_root,
        exe=args.exe, project=args.project, vec=args.vec,
    )
    try:
        if args.smoke:
            report = run_smoke(args, vec_env, policies, actor_keys, numel)
        else:
            report = run_es(args, vec_env, policies, actor_keys, numel, out)
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    report.update({
        "actor_param_count": numel,
        "survival_bar": SURVIVAL_BAR,
        "seed": args.seed,
        "python": sys.version.split()[0],
        "cma": __import__("cma").__version__,
        "sb3": __import__("stable_baselines3").__version__,
    })
    with (out / "c1_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
