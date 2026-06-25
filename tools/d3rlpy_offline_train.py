"""Offline-RL trainer for the Signal Dodge mixed dataset. RUNS IN .venv-d3rlpy.

Loads offline_dataset.npz, builds a d3rlpy MDPDataset (terminals=collision,
timeouts=truncation), fits DiscreteCQL on the full mixed dataset and a
filtered DiscreteBC (%BC: top-return-fraction whole episodes) as the
comparator, and exports each as a TorchScript policy the global-interpreter
eval loads. Writes train_report.json.

NEVER run this in the global interpreter: d3rlpy pins gymnasium 1.0.0 and
would shadow the SB3 stack's gymnasium 1.2.3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig
from d3rlpy.preprocessing import StandardObservationScaler


def make_dataset(obs, act, rew, term, timeout) -> MDPDataset:
    return MDPDataset(
        observations=obs.astype(np.float32),
        actions=act.astype(np.int64),
        rewards=rew.astype(np.float32),
        terminals=term.astype(np.float32),
        timeouts=timeout.astype(np.float32),
        action_space=d3rlpy.ActionSpace.DISCRETE,
        action_size=3,
    )


def _logger_adapter(root: Path):
    """Direct d3rlpy's run logs under the run dir; fall back to default."""
    try:
        return d3rlpy.logging.FileAdapterFactory(root_dir=str(root))
    except Exception:
        return None


def fit_algo(algo, dataset, n_steps: int, name: str, out: Path):
    n_per_epoch = max(1, min(1000, n_steps))
    kwargs = dict(
        dataset=dataset, n_steps=n_steps, n_steps_per_epoch=n_per_epoch,
        experiment_name=name, with_timestamp=False, show_progress=False,
        save_interval=10_000_000,
    )
    adapter = _logger_adapter(out / "d3rlpy_logs")
    if adapter is not None:
        kwargs["logger_adapter"] = adapter
    history = algo.fit(**kwargs)
    last = {}
    try:
        if history:
            last = history[-1][1]
    except Exception:
        last = {}
    return {k: float(v) for k, v in last.items()
            if isinstance(v, (int, float))}


def filter_top_episodes(epidx, epret, frac: float):
    """Indices of transitions belonging to the top-`frac` episodes by return."""
    n_ep = len(epret)
    if n_ep == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(round(frac * n_ep)))
    thresh = np.sort(epret)[::-1][k - 1]
    keep_ep = set(int(i) for i in np.where(epret >= thresh)[0])
    mask = np.array([int(e) in keep_ep for e in epidx], dtype=bool)
    return mask, sorted(keep_ep)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="d3rlpy_offline_train")
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cql-steps", type=int, default=20000)
    p.add_argument("--bc-steps", type=int, default=20000)
    p.add_argument("--filter-frac", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.npz)
    obs, act = z["observations"], z["actions"]
    rew, term, timeout = z["rewards"], z["terminals"], z["timeouts"]
    epidx, epret = z["episode_index"], z["episode_return"]
    report: dict = {
        "n_transitions": int(obs.shape[0]),
        "n_episodes": int(epret.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "reward_sum": float(rew.sum()),
        "terminal_count": int(term.sum()),
        "timeout_count": int(timeout.sum()),
        "return_min": float(epret.min()), "return_max": float(epret.max()),
        "return_mean": float(epret.mean()),
    }
    print("[train] dataset", json.dumps(report), flush=True)

    # --- DiscreteCQL on the full mixed dataset ---
    full = make_dataset(obs, act, rew, term, timeout)
    cql = DiscreteCQLConfig(
        observation_scaler=StandardObservationScaler()
    ).create(device=args.device)
    report["cql_final_metrics"] = fit_algo(cql, full, args.cql_steps, "cql", out)
    cql_pt = out / "cql_policy.pt"
    cql.save_policy(str(cql_pt))
    report["cql_policy"] = str(cql_pt)
    print(f"[train] saved {cql_pt}", flush=True)

    # --- filtered DiscreteBC (%BC) on top-return whole episodes ---
    mask, keep_ep = filter_top_episodes(epidx, epret, args.filter_frac)
    report["filtered_bc"] = {
        "filter_frac": args.filter_frac,
        "kept_episodes": len(keep_ep),
        "kept_transitions": int(mask.sum()),
        "return_threshold": float(np.sort(epret)[::-1][
            max(1, int(round(args.filter_frac * len(epret)))) - 1]),
    }
    fobs, fact = obs[mask], act[mask]
    frew, fterm, ftimeout = rew[mask], term[mask], timeout[mask]
    fbc_ds = make_dataset(fobs, fact, frew, fterm, ftimeout)
    fbc = DiscreteBCConfig(
        observation_scaler=StandardObservationScaler()
    ).create(device=args.device)
    report["filtered_bc"]["final_metrics"] = fit_algo(
        fbc, fbc_ds, args.bc_steps, "filtered_bc", out)
    fbc_pt = out / "fbc_policy.pt"
    fbc.save_policy(str(fbc_pt))
    report["filtered_bc"]["policy"] = str(fbc_pt)
    print(f"[train] saved {fbc_pt}", flush=True)

    (out / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("TRAIN_REPORT", json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
