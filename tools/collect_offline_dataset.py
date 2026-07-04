"""Collect a mixed-quality offline-RL dataset from Signal Dodge behaviors.

Phase L offline-RL pivot. Rolls a mix of behavior policies through the
production GodotSignalDodgeEnv, logs full transitions (obs, action,
reward, terminated, truncated) to an npz, then orchestrates offline
training (DiscreteCQL + filtered DiscreteBC) inside .venv-d3rlpy and an
in-env greedy eval of the resulting TorchScript policies vs the 930.27 bar.

Interpreter boundary (locked):
  - THIS script runs in the GLOBAL interpreter (SB3 + gymnasium 1.2.3 +
    the Godot env). It owns rollouts, the npz, and the in-env eval.
  - tools\\d3rlpy_offline_train.py runs in C:\\Projects\\Sight\\.venv-d3rlpy
    (d3rlpy 2.8.1). It loads the npz, builds an MDPDataset, fits the algos,
    and exports TorchScript policies. d3rlpy is NEVER imported here.

Behavior mix (the point): uniform-random (floor) + saved QR-DQN
checkpoints at several training stages (competence gradient) + the BC
policy (expert). Mixed quality is the precondition for value-based
offline RL to beat the behavior policy via stitching; an all-expert set
reduces to BC. The BC demo npz is deliberately NOT a source.

Collection env seeds default to 3000+ so dataset episodes stay disjoint
from the held-out eval seeds 1000-1009.

Usage (cmd, SIGHT_GODOT_EXE inline), smoke:
    python tools\\collect_offline_dataset.py --smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# In-repo prior art reused wholesale (found-art verdict: ADOPT).
from k5_2_env_dynamics_probe import _build_env  # noqa: E402
from k5_6_bc_eval_inenv import (  # noqa: E402
    load_policy as load_bc_policy,
    greedy_action as bc_greedy_action,
)

BAR = 930.27
BEST_CONSTANT = 845.7
BC_MEAN = 1737.3
VENV_PY = REPO_ROOT / ".venv-d3rlpy" / "Scripts" / "python.exe"
PHASE_K = REPO_ROOT / "runs" / "phase_k"


class RandomBehavior:
    """Uniform-random over Discrete(3). The dataset quality floor."""

    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, obs) -> int:
        return int(self.rng.integers(0, 3))


class QRDQNBehavior:
    """A saved DynQRDQN checkpoint at one training stage. Greedy, eval-mode.

    Loads the SB3 zip and its matching VecNormalize stats; normalizes raw
    env obs exactly as the K6 eval tool does before greedy predict. The
    auxiliary dynamics head is unused at action selection.
    """

    def __init__(self, name: str, ckpt_zip: Path, vecnorm_pkl: Path) -> None:
        from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy
        from sight_agent.rl.dyn_encoder import DynEncoder  # noqa: F401
        from sight_agent.rl.dyn_qrdqn import DynQRDQN
        from k6_dyn_eval_inenv import load_obs_stats, norm_obs

        self.name = name
        self._norm = norm_obs
        self._stats = load_obs_stats(vecnorm_pkl)
        self.model = DynQRDQN.load(
            str(ckpt_zip), device="cpu",
            custom_objects={"policy_class": NoisyQRDQNPolicy},
        )
        self.model.policy.set_training_mode(False)

    def act(self, obs) -> int:
        x = self._norm(obs, *self._stats)
        a, _ = self.model.predict(x, deterministic=True)
        return int(np.asarray(a).reshape(-1)[0])


class BCBehavior:
    """The behavioral-cloning policy. The dataset's expert end."""

    name = "bc"

    def __init__(self, ckpt: Path) -> None:
        self.model, self.mu, self.sd = load_bc_policy(ckpt)

    def act(self, obs) -> int:
        return bc_greedy_action(self.model, self.mu, self.sd, obs)


def collect(env, plan, max_steps: int) -> tuple[dict, dict]:
    """Roll every (behavior, seeds) pair through one env; log transitions.

    observations[t] is the state in which actions[t] was taken; rewards[t]
    the resulting reward; terminals[t] whether the next state is a
    collision; timeouts[t] whether the episode truncated at the step cap.
    """
    obs_l, act_l, rew_l, term_l, trunc_l, epidx_l = [], [], [], [], [], []
    ep_beh, ep_ret, ep_len = [], [], []
    epi = 0
    for source, seeds in plan:
        for s in seeds:
            obs, info = env.reset(seed=int(s))
            steps = 0
            ret = 0.0
            term = trunc = False
            while steps < max_steps:
                a = source.act(obs)
                nobs, r, term, trunc, info = env.step(a)
                obs_l.append(np.asarray(obs, dtype=np.float32))
                act_l.append(int(a))
                rew_l.append(float(r))
                term_l.append(bool(term))
                trunc_l.append(bool(trunc))
                epidx_l.append(epi)
                ret += float(r)
                steps += 1
                obs = nobs
                if term or trunc:
                    break
            ep_beh.append(source.name)
            ep_ret.append(float(ret))
            ep_len.append(int(steps))
            print(
                f"[collect] ep{epi:03d} {source.name:>10} seed={s} "
                f"len={steps} ret={ret:.0f} reason={info.get('terminal_reason','')}",
                flush=True,
            )
            epi += 1
    data = dict(
        observations=np.asarray(obs_l, dtype=np.float32),
        actions=np.asarray(act_l, dtype=np.int64),
        rewards=np.asarray(rew_l, dtype=np.float32),
        terminals=np.asarray(term_l, dtype=bool),
        timeouts=np.asarray(trunc_l, dtype=bool),
        episode_index=np.asarray(epidx_l, dtype=np.int32),
        episode_return=np.asarray(ep_ret, dtype=np.float32),
        episode_length=np.asarray(ep_len, dtype=np.int32),
    )
    meta = dict(
        n_transitions=len(act_l),
        n_episodes=epi,
        episode_behavior=ep_beh,
        episode_return=ep_ret,
        episode_length=ep_len,
        max_steps=max_steps,
    )
    return data, meta


def eval_torchscript(env, policy_pt: Path, seeds, max_steps: int) -> dict:
    """Greedy in-env eval of an exported TorchScript policy.

    Robust to either output convention: a length-1 action id or a length-3
    Q/logit vector (argmax). Same bar (930.27) and report fields as the
    K5.8/K6 eval tools so numbers are directly comparable.
    """
    import torch

    pol = torch.jit.load(str(policy_pt), map_location="cpu")
    pol.eval()
    episodes = []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        steps = 0
        term = trunc = False
        reason = ""
        counts = [0, 0, 0]
        while steps < max_steps:
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                out = pol(x)
            arr = np.asarray(out.detach().cpu()).reshape(-1)
            a = int(round(float(arr[0]))) if arr.size == 1 else int(arr.argmax())
            a = max(0, min(2, a))
            counts[a] += 1
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                reason = info.get("terminal_reason", "")
                break
        episodes.append(dict(seed=int(s), steps=steps,
                             terminal_reason=reason, action_counts_LSR=counts))
        print(f"[eval] seed={s} len={steps} reason={reason} acts={counts}",
              flush=True)
    lengths = [e["steps"] for e in episodes]
    mean_len = float(np.mean(lengths))
    n = len(episodes)
    coll = sum(1 for e in episodes if e["terminal_reason"] == "collision") / n
    tmo = sum(1 for e in episodes if e["terminal_reason"] == "timeout") / n
    pooled = [sum(e["action_counts_LSR"][k] for e in episodes) for k in range(3)]
    tot = sum(pooled)
    frac = [round(c / tot, 3) for c in pooled] if tot else [0, 0, 0]
    verdict = "PASS" if mean_len >= BAR else "FAIL"
    return dict(
        policy=str(policy_pt), eval_seeds=[int(s) for s in seeds],
        max_steps=max_steps, per_seed=episodes,
        mean_episode_length=round(mean_len, 2),
        min_episode_length=int(min(lengths)),
        max_episode_length=int(max(lengths)),
        collision_rate=round(coll, 3), timeout_rate=round(tmo, 3),
        pooled_action_fractions_LSR=frac,
        bar=BAR, best_constant=BEST_CONSTANT, bc_mean=BC_MEAN,
        delta_vs_bar=round(mean_len - BAR, 2),
        delta_vs_bc=round(mean_len - BC_MEAN, 2),
        verdict=verdict,
    )


def seed_range(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def build_plan(args, rng) -> list:
    """Assemble the (behavior, seeds) rollout plan from CLI knobs."""
    plan = []
    cur = args.seed_start
    if args.random_eps > 0:
        plan.append((RandomBehavior(rng), seed_range(cur, args.random_eps)))
        cur += 1000
    for spec in args.qrdqn_stages.split(","):
        spec = spec.strip()
        if not spec:
            continue
        run_name, step = spec.split(":")
        step = int(step)
        ckpt = PHASE_K / run_name / "ckpts" / f"qrdqn_noisy_{step}.zip"
        vnorm = PHASE_K / run_name / "ckpts" / f"vecnorm_{step}.pkl"
        if not ckpt.exists() or not vnorm.exists():
            raise SystemExit(f"missing QR-DQN stage artifact: {ckpt} / {vnorm}")
        label = f"qrdqn_{run_name}_{step}"
        plan.append((QRDQNBehavior(label, ckpt, vnorm),
                     seed_range(cur, args.qrdqn_eps)))
        cur += 1000
    if args.bc_eps > 0:
        bc_ckpt = PHASE_K / "k5_6_bc" / "bc_policy.pt"
        if not bc_ckpt.exists():
            raise SystemExit(f"missing BC checkpoint: {bc_ckpt}")
        plan.append((BCBehavior(bc_ckpt), seed_range(cur, args.bc_eps)))
        cur += 1000
    if not plan:
        raise SystemExit("empty rollout plan")
    return plan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="collect_offline_dataset")
    p.add_argument("--out", type=Path, default=PHASE_K / "k7_offline")
    p.add_argument("--seed-start", type=int, default=3000)
    p.add_argument("--random-eps", type=int, default=8)
    p.add_argument("--qrdqn-eps", type=int, default=4)
    p.add_argument("--qrdqn-stages", default="k6_dyn_off_s0:50000,k6_dyn_off_s0:200000")
    p.add_argument("--bc-eps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--cql-steps", type=int, default=20000)
    p.add_argument("--bc-steps", type=int, default=20000)
    p.add_argument("--filter-frac", type=float, default=0.25)
    p.add_argument("--eval-seeds", default="1000-1009")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end pipe check: few eps, short train, 3 eval seeds")
    args = p.parse_args(argv)

    if args.smoke:
        args.random_eps = 4
        args.qrdqn_eps = 2
        args.bc_eps = 2
        args.cql_steps = 2000
        args.bc_steps = 2000
        args.eval_seeds = "1000-1002"

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "offline_dataset.npz"
    meta_path = out / "offline_dataset_meta.json"
    eval_seeds = _parse_seeds(args.eval_seeds)

    rng = np.random.default_rng(0)
    t0 = time.monotonic()
    env = _build_env(observation_mode="state", run_dir=out / "collect_godot",
                     seed=args.seed_start, max_steps=args.max_steps,
                     reward_shaping="none")
    report: dict = {"args": vars_str(args)}
    try:
        plan = build_plan(args, rng)
        data, meta = collect(env, plan, args.max_steps)
        np.savez_compressed(npz_path, **data)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[collect] wrote {npz_path} ({meta['n_transitions']} transitions, "
              f"{meta['n_episodes']} episodes) in {time.monotonic()-t0:.1f}s",
              flush=True)
        report["dataset"] = meta

        if not args.skip_train:
            train_report = run_trainer(npz_path, out, args)
            report["train"] = train_report

        if not args.skip_eval:
            report["eval"] = {}
            for name, pt in (("cql", out / "cql_policy.pt"),
                             ("filtered_bc", out / "fbc_policy.pt")):
                if pt.exists():
                    print(f"[eval] {name}: {pt.name}", flush=True)
                    report["eval"][name] = eval_torchscript(
                        env, pt, eval_seeds, args.max_steps)
                else:
                    report["eval"][name] = {"error": f"missing {pt}"}
    finally:
        env.close()

    (out / "k7_offline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT", json.dumps(report.get("eval", {})), flush=True)
    _print_verdict(report)
    return 0


def _parse_seeds(spec: str) -> list[int]:
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


def vars_str(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def run_trainer(npz_path: Path, out: Path, args) -> dict:
    """Subprocess the d3rlpy trainer in .venv-d3rlpy. d3rlpy stays out of
    the global interpreter entirely."""
    if not VENV_PY.exists():
        return {"error": f"venv python missing: {VENV_PY}"}
    trainer = REPO_ROOT / "tools" / "d3rlpy_offline_train.py"
    cmd = [
        str(VENV_PY), str(trainer),
        "--npz", str(npz_path), "--out", str(out),
        "--cql-steps", str(args.cql_steps),
        "--bc-steps", str(args.bc_steps),
        "--filter-frac", str(args.filter_frac),
    ]
    print(f"[train] -> {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(proc.stdout[-4000:] if proc.stdout else "", flush=True)
    if proc.returncode != 0:
        print("[train] STDERR", proc.stderr[-4000:], flush=True)
        return {"error": "trainer failed", "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    rep_path = out / "train_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    rep["wall_s"] = round(time.monotonic() - t0, 1)
    return rep


def _print_verdict(report: dict) -> None:
    ev = report.get("eval", {})
    line = []
    for name in ("cql", "filtered_bc"):
        r = ev.get(name, {})
        if "mean_episode_length" in r:
            line.append(f"{name} mean={r['mean_episode_length']} "
                        f"d_bar={r['delta_vs_bar']} {r['verdict']}")
        else:
            line.append(f"{name} {r.get('error','n/a')}")
    print("VERDICT " + " | ".join(line) + f" | bar={BAR} bc={BC_MEAN}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
