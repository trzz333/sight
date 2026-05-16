"""Phase I activation comparator (docs-and-tools-only slice).

Loads multiple SB3 PPO CnnPolicy checkpoints, drives the Godot Signal
Dodge env once with a fixed behavior tape, and queries every loaded
model on every observation along the trajectory at five extraction
points: features extractor output, mlp_extractor.latent_pi,
mlp_extractor.latent_vf, action_net raw logits, and value_net output.
Additionally computes manual softmax(raw_logits) and SB3
get_distribution probs as a consistency check.

Per-model summaries quantify whether each layer's activations vary
across observations (per-dimension std, adjacent-step L2, first-vs-last
L2, effective rank via PCA, top-k explained variance). Raw action
logits are reported alongside the action_net weight row norms,
biases, bias gaps, and the ratio between projection std and bias gap.
That ratio is the decisive scalar: if std(W @ latent_pi) is small
relative to the bias gap between the top-bias action and the second-
bias action, the action head behaves as a constant classifier and is
the proximate locus of the observation-insensitive constant-action
policies surfaced in Phase H.

No training. No env code change. No config change. Reuses the
fingerprinting, model-loading verification, tape parser, and obs
hashing from tools/h5_logit_compare.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_TOOLS = _REPO_ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.factories import make_env  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402

import h5_logit_compare as h5lc  # noqa: E402

ACTION_NAMES = h5lc.ACTION_NAMES


def actor_path_forward(model: Any, obs_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the SB3 PPO actor path step by step.

    Returns:
      features, latent_pi, latent_vf, raw_logits, value, probs_manual,
      probs_sb3, logits_sb3.
    """
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        me_out = model.policy.mlp_extractor(features)
        if isinstance(me_out, (tuple, list)):
            latent_pi, latent_vf = me_out[0], me_out[1]
        else:
            latent_pi = latent_vf = me_out
        raw_logits = model.policy.action_net(latent_pi)
        value = model.policy.value_net(latent_vf)
        probs_manual = torch.softmax(raw_logits, dim=-1)
        dist = model.policy.get_distribution(obs_tensor)
        cat = dist.distribution
        probs_sb3 = cat.probs
        logits_sb3 = cat.logits
    return {
        "features": features,
        "latent_pi": latent_pi,
        "latent_vf": latent_vf,
        "raw_logits": raw_logits,
        "value": value,
        "probs_manual": probs_manual,
        "probs_sb3": probs_sb3,
        "logits_sb3": logits_sb3,
    }


def summarize_layer(arr: np.ndarray) -> dict[str, Any]:
    """Compute the per-layer activation summary.

    arr shape (n_steps, d). Reports norm, per-dim std, adjacent-step
    L2, first-vs-last L2, covariance trace, effective rank
    (participation ratio of singular value squares), and top-k PCA
    explained variance fractions.
    """
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    n, d = arr.shape
    per_dim_std = arr.std(axis=0)
    norms = np.linalg.norm(arr, axis=1)
    if n > 1:
        adj_l2 = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        first_last_l2 = float(np.linalg.norm(arr[-1] - arr[0]))
    else:
        adj_l2 = np.array([0.0])
        first_last_l2 = 0.0
    if n > 1:
        cov_trace = float(np.sum(per_dim_std ** 2 * (n - 1) / max(n, 1)))
    else:
        cov_trace = 0.0
    eff_rank = 0.0
    top1 = top3 = top10 = 0.0
    if n > 1 and d > 0:
        centered = arr - arr.mean(axis=0, keepdims=True)
        try:
            s = np.linalg.svd(centered, compute_uv=False)
            s2 = (s * s).astype(np.float64)
            total = float(s2.sum())
            if total > 1e-20:
                eff_rank = float((total * total) / float(np.sum(s2 * s2)))
                top1 = float(s2[0] / total)
                k3 = min(3, len(s2))
                top3 = float(s2[:k3].sum() / total)
                k10 = min(10, len(s2))
                top10 = float(s2[:k10].sum() / total)
        except np.linalg.LinAlgError:
            pass
    thresholds = (1e-6, 1e-5, 1e-4)
    frac = {
        f"frac_dim_std_above_{thr}": float((per_dim_std > thr).mean())
        for thr in thresholds
    }
    return {
        "shape": [int(n), int(d)],
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "per_dim_std": {
            "mean": float(per_dim_std.mean()),
            "median": float(np.median(per_dim_std)),
            "p95": float(np.percentile(per_dim_std, 95.0)),
            "max": float(per_dim_std.max()),
            "min": float(per_dim_std.min()),
        },
        **frac,
        "adjacent_step_l2": {
            "mean": float(adj_l2.mean()),
            "median": float(np.median(adj_l2)),
            "p95": float(np.percentile(adj_l2, 95.0)),
        },
        "first_vs_last_l2": first_last_l2,
        "cov_trace": cov_trace,
        "effective_rank": eff_rank,
        "pca_explained_variance": {
            "top1": top1,
            "top3": top3,
            "top10": top10,
        },
    }


def summarize_raw_logits(
    raw_logits: np.ndarray, probs_sb3: np.ndarray, probs_manual: np.ndarray,
) -> dict[str, Any]:
    """Per-action raw-logit stats plus the SB3 consistency check."""
    n, d = raw_logits.shape
    per_action = {}
    for a in range(d):
        col = raw_logits[:, a]
        per_action[ACTION_NAMES[a]] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
        }
    sorted_logits = np.sort(raw_logits, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    argmax = raw_logits.argmax(axis=1)
    eps = 1e-12
    ent = -np.sum(probs_sb3 * np.log(np.clip(probs_sb3, eps, 1.0)), axis=1)
    err = float(np.max(np.abs(probs_manual - probs_sb3)))
    return {
        "shape": [int(n), int(d)],
        "per_action": per_action,
        "raw_top1_top2_margin": {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "min": float(margin.min()),
            "p95": float(np.percentile(margin, 95.0)),
            "max": float(margin.max()),
        },
        "argmax_counts": {
            ACTION_NAMES[a]: int((argmax == a).sum()) for a in range(d)
        },
        "argmax_fractions": {
            ACTION_NAMES[a]: float((argmax == a).mean()) for a in range(d)
        },
        "probability_entropy": {
            "mean": float(ent.mean()),
            "min": float(ent.min()),
            "max": float(ent.max()),
        },
        "manual_vs_sb3_probs_max_abs_err": err,
    }


def inspect_action_net(model: Any, latent_pi: np.ndarray) -> dict[str, Any]:
    """Inspect the action_net Linear layer weights, biases, and projection.

    Reports:
      weight shape, per-action weight row norms
      bias values, all pairwise bias gaps, bias-only argmax
      W @ latent_pi statistics per action (mean, std, min, max)
      the ratio std(W @ latent_pi)[top_bias_action] / |bias_gap_top|
    A small ratio is the signature of a constant-classifier action head.
    """
    an = model.policy.action_net
    W = an.weight.detach().cpu().numpy().astype(np.float64)
    b = an.bias.detach().cpu().numpy().astype(np.float64)
    n_actions, latent_dim = W.shape
    row_norms = np.linalg.norm(W, axis=1)
    proj = latent_pi @ W.T  # (n_steps, n_actions)
    proj_mean = proj.mean(axis=0)
    proj_std = proj.std(axis=0)
    proj_min = proj.min(axis=0)
    proj_max = proj.max(axis=0)
    bias_gap = {}
    for i in range(n_actions):
        for j in range(n_actions):
            if i == j:
                continue
            bias_gap[f"{ACTION_NAMES[i]}_minus_{ACTION_NAMES[j]}"] = float(b[i] - b[j])
    sorted_bias_idx = np.argsort(b)[::-1]
    top_idx = int(sorted_bias_idx[0])
    second_idx = int(sorted_bias_idx[1])
    bias_gap_top = float(b[top_idx] - b[second_idx])
    proj_std_top = float(proj_std[top_idx])
    ratio = (
        proj_std_top / abs(bias_gap_top)
        if abs(bias_gap_top) > 1e-12
        else float("inf")
    )
    # raw-logit margin from W @ latent_pi + b: max margin between top1 and top2 raw_logit
    raw_logits_from_proj = proj + b[np.newaxis, :]
    sorted_proj_logits = np.sort(raw_logits_from_proj, axis=1)
    raw_margin = sorted_proj_logits[:, -1] - sorted_proj_logits[:, -2]
    return {
        "weight_shape": [int(n_actions), int(latent_dim)],
        "per_action_weight_row_norm": {
            ACTION_NAMES[a]: float(row_norms[a]) for a in range(n_actions)
        },
        "bias": {ACTION_NAMES[a]: float(b[a]) for a in range(n_actions)},
        "bias_gap": bias_gap,
        "bias_only_argmax": ACTION_NAMES[int(np.argmax(b))],
        "top_bias_action": ACTION_NAMES[top_idx],
        "second_bias_action": ACTION_NAMES[second_idx],
        "bias_gap_top_minus_second": bias_gap_top,
        "projection_W_dot_latent_per_action": {
            ACTION_NAMES[a]: {
                "mean": float(proj_mean[a]),
                "std": float(proj_std[a]),
                "min": float(proj_min[a]),
                "max": float(proj_max[a]),
            }
            for a in range(n_actions)
        },
        "proj_std_top_bias_action": proj_std_top,
        "ratio_proj_std_over_bias_gap": ratio,
        "raw_logit_margin_reconstructed": {
            "mean": float(raw_margin.mean()),
            "min": float(raw_margin.min()),
            "p95": float(np.percentile(raw_margin, 95.0)),
        },
    }


def drive_env_once(
    cfg: dict[str, Any],
    behavior_tape: str,
    eval_seed: int,
    max_steps: int,
    out_dir: Path,
) -> tuple[list[np.ndarray], list[str]]:
    """Drive the Godot env once with the fixed tape and record per-step obs.

    Returns the recorded obs sequence and the per-step obs hashes. The
    env is deterministic at fixed eval seed and fixed tape, so a single
    pass produces the same trajectory every model would otherwise see;
    we feed those obs to all models offline.
    """
    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)
    eval_run_dir = out_dir / f"activation_{behavior_tape}_godot"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    if godot_extra:
        env = make_env(
            env_id, n_envs=1, seed=base_seed, mode="eval",
            run_dir=str(eval_run_dir), **godot_extra,
        )
    else:
        env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval")
    tape = h5lc.parse_tape(behavior_tape, max_steps)
    obs_seq: list[np.ndarray] = []
    try:
        try:
            env.seed(int(eval_seed))
        except (AttributeError, TypeError):
            pass
        obs = env.reset()
        for t in range(int(max_steps)):
            obs_seq.append(np.asarray(obs, dtype=np.uint8).copy())
            tape_action = int(tape[t])
            obs, _r, dones, _i = env.step(np.asarray([tape_action]))
            if bool(np.asarray(dones).any()):
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    hashes = [h5lc.obs_hash(o) for o in obs_seq]
    return obs_seq, hashes


def collect_model_activations(
    model: Any, obs_sequence: list[np.ndarray],
) -> dict[str, Any]:
    """Run the actor path on every recorded obs for one loaded model."""
    feats_list: list[np.ndarray] = []
    latent_pi_list: list[np.ndarray] = []
    latent_vf_list: list[np.ndarray] = []
    raw_logits_list: list[np.ndarray] = []
    values_list: list[float] = []
    probs_manual_list: list[np.ndarray] = []
    probs_sb3_list: list[np.ndarray] = []
    consistency_max_err = 0.0
    for obs_arr in obs_sequence:
        obs_tensor, _ = model.policy.obs_to_tensor(obs_arr)
        layers = actor_path_forward(model, obs_tensor)
        feats_list.append(
            layers["features"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        latent_pi_list.append(
            layers["latent_pi"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        latent_vf_list.append(
            layers["latent_vf"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        raw_logits_list.append(
            layers["raw_logits"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        )
        values_list.append(
            float(layers["value"].detach().cpu().numpy().reshape(-1)[0])
        )
        pm = layers["probs_manual"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        ps = layers["probs_sb3"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        probs_manual_list.append(pm)
        probs_sb3_list.append(ps)
        err = float(np.max(np.abs(pm - ps)))
        if err > consistency_max_err:
            consistency_max_err = err
    return {
        "n_steps": len(feats_list),
        "consistency_max_err": consistency_max_err,
        "features": np.asarray(feats_list, dtype=np.float32),
        "latent_pi": np.asarray(latent_pi_list, dtype=np.float32),
        "latent_vf": np.asarray(latent_vf_list, dtype=np.float32),
        "raw_logits": np.asarray(raw_logits_list, dtype=np.float64),
        "values": np.asarray(values_list, dtype=np.float64),
        "probs_manual": np.asarray(probs_manual_list, dtype=np.float64),
        "probs_sb3": np.asarray(probs_sb3_list, dtype=np.float64),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="h5_activation_compare",
        description="Phase I docs-and-tools-only activation comparator.",
    )
    p.add_argument("--config", required=True, help="Path to YAML env config.")
    p.add_argument(
        "--models", required=True,
        help="Comma-separated label=train_run_dir pairs.",
    )
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument(
        "--behavior-tape", required=True,
        help="Keyword (stay|left|right) or comma-separated action ints.",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for the per-tape summary JSON and Godot sidecar.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models_map = h5lc.parse_models_arg(args.models)
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[str, Any, dict[str, Any]]] = []
    seen_sha: dict[str, str] = {}
    for label, run_dir in models_map.items():
        model, fp = h5lc.load_and_fingerprint(run_dir)
        if fp["sha256"] in seen_sha:
            raise RuntimeError(
                f"models {label!r} and {seen_sha[fp['sha256']]!r} share "
                "identical sha256; model-loading bug suspected. Aborting."
            )
        seen_sha[fp["sha256"]] = label
        loaded.append((label, model, fp))
    labels = [l for l, _, _ in loaded]
    fingerprints = {l: fp for l, _, fp in loaded}

    print(f"phase_i_activation tape={args.behavior_tape} models={labels}")
    obs_sequence, obs_hashes = drive_env_once(
        cfg, args.behavior_tape, args.eval_seed, args.max_steps, out_dir,
    )
    n_steps = len(obs_sequence)
    obs_unique = len(set(obs_hashes))
    print(f"  obs collected: n_steps={n_steps} unique={obs_unique}")

    started_at = time.time()
    per_model_results: dict[str, Any] = {}
    for label, model, _ in loaded:
        coll = collect_model_activations(model, obs_sequence)
        feats_summary = summarize_layer(coll["features"])
        latent_pi_summary = summarize_layer(coll["latent_pi"])
        latent_vf_summary = summarize_layer(coll["latent_vf"])
        raw_logits_summary = summarize_raw_logits(
            coll["raw_logits"], coll["probs_sb3"], coll["probs_manual"],
        )
        action_net_summary = inspect_action_net(model, coll["latent_pi"])
        value_summary = {
            "mean": float(coll["values"].mean()),
            "std": float(coll["values"].std()),
            "min": float(coll["values"].min()),
            "max": float(coll["values"].max()),
        }
        per_model_results[label] = {
            "n_steps": int(coll["n_steps"]),
            "consistency_max_err": float(coll["consistency_max_err"]),
            "features": feats_summary,
            "latent_pi": latent_pi_summary,
            "latent_vf": latent_vf_summary,
            "raw_logits": raw_logits_summary,
            "value": value_summary,
            "action_net": action_net_summary,
        }
    elapsed = time.time() - started_at

    out = {
        "_header": {
            "tool": "tools/h5_activation_compare.py",
            "config": str(args.config),
            "eval_seed": int(args.eval_seed),
            "max_steps": int(args.max_steps),
            "behavior_tape": args.behavior_tape,
            "n_steps": int(n_steps),
            "elapsed_seconds": float(elapsed),
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": fingerprints,
        },
        "obs": {
            "n_steps": int(n_steps),
            "unique_hash_count": int(obs_unique),
            "all_distinct": (obs_unique == n_steps),
        },
        "per_model": per_model_results,
    }
    summary_path = out_dir / f"activation_compare_{args.behavior_tape}.summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"phase_i_activation tape={args.behavior_tape} steps={n_steps} elapsed={elapsed:.1f}s")
    for label in labels:
        r = per_model_results[label]
        f = r["features"]; lp = r["latent_pi"]; rl = r["raw_logits"]; an = r["action_net"]
        print(f"  {label}: cons_err={r['consistency_max_err']:.2e}")
        print(
            f"    features  norm_mean={f['norm_mean']:.4f} norm_std={f['norm_std']:.4f} "
            f"adj_l2_mean={f['adjacent_step_l2']['mean']:.4f} "
            f"eff_rank={f['effective_rank']:.2f} "
            f"top1pca={f['pca_explained_variance']['top1']:.4f} "
            f"frac_dim_std_gt_1e-4={f['frac_dim_std_above_0.0001']:.4f}"
        )
        print(
            f"    latent_pi norm_mean={lp['norm_mean']:.4f} norm_std={lp['norm_std']:.4f} "
            f"adj_l2_mean={lp['adjacent_step_l2']['mean']:.4f} "
            f"eff_rank={lp['effective_rank']:.2f}"
        )
        per_action_std = {
            a: round(float(rl["per_action"][a]["std"]), 4)
            for a in ("left", "stay", "right")
        }
        print(
            f"    raw_logits argmax={rl['argmax_fractions']} "
            f"margin_mean={rl['raw_top1_top2_margin']['mean']:.4f} "
            f"margin_min={rl['raw_top1_top2_margin']['min']:.4f} "
            f"per_action_std={per_action_std}"
        )
        print(
            f"    action_net top_bias={an['top_bias_action']} "
            f"bias_gap={an['bias_gap_top_minus_second']:.4f} "
            f"proj_std_top={an['proj_std_top_bias_action']:.4f} "
            f"ratio={an['ratio_proj_std_over_bias_gap']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
