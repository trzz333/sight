"""Phase J stochastic-action eval ablation tool.

Loads one or more trained SB3 PPO CnnPolicy checkpoints, drives the
Godot Signal Dodge env under ``model.predict(obs, deterministic=False)``,
and records per-step sampled actions, deterministic argmax for the same
observation, action probabilities, sampled-differs-from-argmax flags,
and per-episode outcome classification.

Two-stage workflow per GPT scope note: J0 pilot is phase_e_seed2 only,
10 eval seeds, 5 replicates per seed = 50 episodes. J1 expansion adds
phase_g_seed2 and optional Class A controls only if J0 crosses the
trajectory-outcome threshold.

Replicate seeding: ``policy_sample_seed = eval_seed + 10_000_000 +
replicate_index``. Python random, NumPy, and Torch are reseeded at
the start of every episode so five replicates on the same env seed
sample independently.

No training. No env code change. No config change. No reward variant.
Reuses fingerprinting, env factory wiring, obs hashing from
tools/h5_logit_compare.py.
"""

from __future__ import annotations

import argparse
import json
import random
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

ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}
POLICY_SAMPLE_SEED_OFFSET = 10_000_000


def seed_everything(seed: int) -> None:
    """Reseed Python random, NumPy, and Torch for one episode."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_models_arg_min1(models_str: str) -> dict[str, Path]:
    """Like h5_logit_compare.parse_models_arg but accepts >=1 model.

    Phase J J0 runs a single model (phase_e_seed2). Phase J J1 runs two
    models with optional Class A controls.
    """
    out: dict[str, Path] = {}
    for tok in models_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--models token missing '=': {tok!r}")
        label, path = tok.split("=", 1)
        label = label.strip(); path = path.strip()
        if not label or not path:
            raise ValueError(f"--models token has empty label or path: {tok!r}")
        if label in out:
            raise ValueError(f"--models label repeated: {label!r}")
        out[label] = Path(path)
    if len(out) < 1:
        raise ValueError("at least one --models entry required")
    return out


def parse_seed_range(spec: str) -> list[int]:
    """Parse '1000,1001' or '1000-1009' or mixes."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("seeds must be non-empty")
    out: list[int] = []
    for raw_tok in spec.split(","):
        tok = raw_tok.strip()
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) != 2:
                raise ValueError(f"bad range token: {tok!r}")
            lo = int(parts[0]); hi = int(parts[1])
            if hi < lo:
                raise ValueError(f"hi<lo: {tok!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(tok))
    return out


def run_one_episode(
    model: Any,
    env: Any,
    eval_seed: int,
    replicate_idx: int,
    policy_sample_seed: int,
    max_steps: int,
    episode_id: int,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one stochastic episode and return (episode_summary, per_step_rows)."""
    seed_everything(policy_sample_seed)
    try:
        env.seed(int(eval_seed))
    except (AttributeError, TypeError):
        pass
    obs = env.reset()
    rows: list[dict[str, Any]] = []
    ep_reward = 0.0
    ep_len = 0
    final_info: dict[str, Any] = {}
    done_flag = False
    sampled_counts = {0: 0, 1: 0, 2: 0}
    argmax_counts = {0: 0, 1: 0, 2: 0}
    differs = 0
    first_non_left_step: int | None = None
    player_xs: list[float] = []
    t0 = time.time()
    while ep_len < max_steps:
        obs_arr = np.asarray(obs)
        obs_tensor, _ = model.policy.obs_to_tensor(obs_arr)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
            argmax_action = int(np.argmax(probs))
        action_arr, _ = model.predict(obs_arr, deterministic=False)
        sampled = int(np.asarray(action_arr).reshape(-1)[0])
        sampled_counts[sampled] += 1
        argmax_counts[argmax_action] += 1
        is_diff = sampled != argmax_action
        if is_diff:
            differs += 1
        if first_non_left_step is None and sampled != 0:
            first_non_left_step = ep_len
        obs, reward, dones, infos = env.step(np.asarray([sampled]))
        ep_reward += float(np.asarray(reward).sum())
        # Try to extract player_x from env info (works only when reward_shaping
        # is enabled in the config); skip gracefully when absent.
        player_x_val: float | None = None
        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
            rs = infos[0].get("reward_state")
            if isinstance(rs, dict) and "player_x" in rs:
                try:
                    player_x_val = float(rs["player_x"])
                    player_xs.append(player_x_val)
                except (TypeError, ValueError):
                    pass
        rows.append({
            "label": label,
            "episode_id": int(episode_id),
            "eval_seed": int(eval_seed),
            "replicate_idx": int(replicate_idx),
            "step": int(ep_len),
            "sampled_action": int(sampled),
            "deterministic_argmax": int(argmax_action),
            "differs": bool(is_diff),
            "p_left": float(probs[0]),
            "p_stay": float(probs[1]),
            "p_right": float(probs[2]),
            "reward": float(np.asarray(reward).sum()),
            "player_x": player_x_val,
        })
        ep_len += 1
        done_flag = bool(np.asarray(dones).any())
        if done_flag:
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                final_info = dict(infos[0])
            break
    elapsed = time.time() - t0
    if not done_flag:
        collision = False
        timeout = True
    else:
        truncated = bool(final_info.get("TimeLimit.truncated", False))
        collision = not truncated
        timeout = truncated
    # Classification: matches the behavior-audit "wall_hugging_into_collision"
    # and "survived_to_timeout" labels where possible. We rely on action
    # distribution rather than player_x because player_x is only present
    # in info when reward_shaping is enabled.
    left_frac = sampled_counts[0] / ep_len if ep_len else 0.0
    if timeout:
        classification = "survived_to_timeout"
    elif collision and left_frac >= 0.8:
        classification = "wall_hugging_into_collision"
    elif collision:
        classification = "non_wall_hugging_collision"
    else:
        classification = "other"
    summary = {
        "label": label,
        "episode_id": int(episode_id),
        "eval_seed": int(eval_seed),
        "replicate_idx": int(replicate_idx),
        "policy_sample_seed": int(policy_sample_seed),
        "episode_length": int(ep_len),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "total_reward": float(ep_reward),
        "elapsed_seconds": float(elapsed),
        "sampled_action_counts": {ACTION_NAMES[a]: int(sampled_counts[a]) for a in (0, 1, 2)},
        "sampled_action_fractions": {
            ACTION_NAMES[a]: float(sampled_counts[a] / ep_len if ep_len else 0.0)
            for a in (0, 1, 2)
        },
        "argmax_action_counts": {ACTION_NAMES[a]: int(argmax_counts[a]) for a in (0, 1, 2)},
        "sampled_differs_from_argmax_count": int(differs),
        "sampled_differs_from_argmax_fraction": float(differs / ep_len if ep_len else 0.0),
        "first_non_left_step": (
            int(first_non_left_step) if first_non_left_step is not None else None
        ),
        "classification": classification,
        "player_x_available": len(player_xs) > 0,
        "player_x": (
            {
                "mean": float(np.mean(player_xs)),
                "min": float(np.min(player_xs)),
                "max": float(np.max(player_xs)),
                "le16_fraction": float(np.mean(np.asarray(player_xs) <= 16.0)),
                "n_samples": int(len(player_xs)),
            } if player_xs else None
        ),
    }
    return summary, rows


# ---------------------------------------------------------------------------
# K5.3 step-weighted aggregation + classification
# ---------------------------------------------------------------------------

# K5.3 classification constants. best_constant_mean_episode_length anchors
# `constant_left` from K5.2 layer 6 (845.7 mean frames over 10 seeds).
# materiality_threshold is 10% of that baseline; material_survival_bar is
# baseline + threshold = 930.27. Both Grok GREEN reply and GPT K5.3 packet
# adopt these exact numbers.
K5_3_BEST_CONSTANT_MEAN_EPISODE_LENGTH: float = 845.7
K5_3_MATERIALITY_THRESHOLD: float = 84.57
K5_3_MATERIAL_SURVIVAL_BAR: float = (
    K5_3_BEST_CONSTANT_MEAN_EPISODE_LENGTH + K5_3_MATERIALITY_THRESHOLD
)


def _stat_block(arr) -> dict:
    """Mean/std/min/p05/p50/p95/max plus n. Empty-safe."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _compute_k5_3_block(
    step_p_left,
    step_p_stay,
    step_p_right,
    step_sampled,
    step_argmax,
    sampled_mean_episode_length,
) -> dict:
    """Step-weighted K5.3 aggregates + classification + overlay.

    Primary bucket is exclusive and ordered: ARGMAX-ARTIFACT first (requires
    both diversity and material survival gain), then POLICY-DIST-COLLAPSE,
    then SOFT-BAD-POLICY, then UNKNOWN. The NEAR-TIE-BIAS overlay is
    independent of the primary bucket and reported separately.
    """
    n_steps = len(step_sampled)
    sampled_arr = np.asarray(step_sampled, dtype=np.int64)
    argmax_arr = np.asarray(step_argmax, dtype=np.int64)
    pl_arr = np.asarray(step_p_left, dtype=np.float64)
    ps_arr = np.asarray(step_p_stay, dtype=np.float64)
    pr_arr = np.asarray(step_p_right, dtype=np.float64)
    if n_steps == 0:
        return {
            "n_steps": 0,
            "constants": {
                "best_constant_mean_episode_length": K5_3_BEST_CONSTANT_MEAN_EPISODE_LENGTH,
                "materiality_threshold": K5_3_MATERIALITY_THRESHOLD,
                "material_survival_bar": K5_3_MATERIAL_SURVIVAL_BAR,
            },
            "k5_3_classification": {
                "primary_bucket": "UNKNOWN",
                "near_tie_bias_overlay": False,
                "reason": "no_steps_recorded",
            },
        }
    eps = 1e-12
    pl_safe = np.maximum(pl_arr, eps)
    ps_safe = np.maximum(ps_arr, eps)
    pr_safe = np.maximum(pr_arr, eps)
    entropy_arr = -(
        pl_safe * np.log(pl_safe)
        + ps_safe * np.log(ps_safe)
        + pr_safe * np.log(pr_safe)
    )
    stacked = np.stack([pl_arr, ps_arr, pr_arr], axis=1)
    sorted_desc = np.sort(stacked, axis=1)[:, ::-1]
    margin_arr = sorted_desc[:, 0] - sorted_desc[:, 1]
    diff_mask = sampled_arr != argmax_arr
    diff_step_fraction = float(diff_mask.mean())

    def _action_fracs(arr):
        return {
            ACTION_NAMES[a]: float((arr == a).mean()) for a in (0, 1, 2)
        }

    sampled_action_fractions = _action_fracs(sampled_arr)
    argmax_action_fractions = _action_fracs(argmax_arr)
    entropy_stats = _stat_block(entropy_arr)
    margin_stats = _stat_block(margin_arr)
    margin_lt_005_fraction = float((margin_arr < 0.05).mean())
    per_action_probability = {
        "left": _stat_block(pl_arr),
        "stay": _stat_block(ps_arr),
        "right": _stat_block(pr_arr),
    }
    argmax_counts = {a: int(np.sum(argmax_arr == a)) for a in (0, 1, 2)}
    argmax_concentrated_on_single_action = bool(
        max(argmax_counts.values()) == n_steps
    )
    argmax_dominant_action_name = None
    for a, c in argmax_counts.items():
        if c == n_steps:
            argmax_dominant_action_name = ACTION_NAMES[a]
            break

    sampled_mean_ep_len = float(sampled_mean_episode_length)
    entropy_mean = float(entropy_stats.get("mean", 0.0))
    if (
        diff_step_fraction > 0.05
        and sampled_mean_ep_len > K5_3_MATERIAL_SURVIVAL_BAR
    ):
        primary_bucket = "ARGMAX-ARTIFACT"
    elif diff_step_fraction <= 0.05 or entropy_mean < 0.1:
        primary_bucket = "POLICY-DIST-COLLAPSE"
    elif (
        diff_step_fraction > 0.05
        and sampled_mean_ep_len <= K5_3_MATERIAL_SURVIVAL_BAR
    ):
        primary_bucket = "SOFT-BAD-POLICY"
    else:
        primary_bucket = "UNKNOWN"

    near_tie_bias_overlay = bool(
        margin_lt_005_fraction > 0.50
        and argmax_concentrated_on_single_action
    )

    return {
        "n_steps": int(n_steps),
        "constants": {
            "best_constant_mean_episode_length": K5_3_BEST_CONSTANT_MEAN_EPISODE_LENGTH,
            "materiality_threshold": K5_3_MATERIALITY_THRESHOLD,
            "material_survival_bar": K5_3_MATERIAL_SURVIVAL_BAR,
        },
        "step_weighted_sampled_action_fractions": sampled_action_fractions,
        "step_weighted_argmax_action_fractions": argmax_action_fractions,
        "sampled_differs_from_argmax_step_fraction": diff_step_fraction,
        "entropy_nats": entropy_stats,
        "top1_top2_probability_margin": margin_stats,
        "top1_top2_probability_margin_lt_0p05_fraction": margin_lt_005_fraction,
        "per_action_probability": per_action_probability,
        "argmax_action_step_counts": {ACTION_NAMES[a]: int(c) for a, c in argmax_counts.items()},
        "argmax_concentrated_on_single_action": argmax_concentrated_on_single_action,
        "argmax_dominant_action": argmax_dominant_action_name,
        "k5_3_classification": {
            "primary_bucket": primary_bucket,
            "near_tie_bias_overlay": near_tie_bias_overlay,
            "inputs_used": {
                "sampled_differs_from_argmax_step_fraction": diff_step_fraction,
                "sampled_mean_episode_length": sampled_mean_ep_len,
                "material_survival_bar": K5_3_MATERIAL_SURVIVAL_BAR,
                "entropy_nats_mean": entropy_mean,
                "top1_top2_probability_margin_lt_0p05_fraction": margin_lt_005_fraction,
                "argmax_concentrated_on_single_action": argmax_concentrated_on_single_action,
            },
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="h5_stochastic_eval",
        description="Phase J docs-and-tools-only stochastic-action eval ablation.",
    )
    p.add_argument("--config", required=True, help="Path to YAML env config.")
    p.add_argument(
        "--models", required=True,
        help="Comma-separated label=train_run_dir pairs.",
    )
    p.add_argument("--seeds", required=True, help="Eval seeds, e.g. '1000-1009'.")
    p.add_argument(
        "--replicates", type=int, default=5,
        help="Stochastic replicates per eval seed (default 5).",
    )
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--out-dir", required=True, help="Output directory under runs/.")
    p.add_argument(
        "--label-suffix", default="",
        help="Optional suffix appended to output filenames.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models_map = parse_models_arg_min1(args.models)
    cfg = load_config(args.config)
    seeds = parse_seed_range(args.seeds)
    if not seeds:
        raise ValueError("no eval seeds parsed")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        ("_" + args.label_suffix) if args.label_suffix else ""
    )

    loaded: list[tuple[str, Any, dict[str, Any]]] = []
    seen_sha: dict[str, str] = {}
    for label, run_dir in models_map.items():
        model, fp = h5lc.load_and_fingerprint(run_dir)
        if fp["sha256"] in seen_sha:
            raise RuntimeError(
                f"models {label!r} and {seen_sha[fp['sha256']]!r} share "
                "identical sha256; model-loading bug suspected."
            )
        seen_sha[fp["sha256"]] = label
        loaded.append((label, model, fp))
    fingerprints = {l: fp for l, _, fp in loaded}

    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)
    started_at = time.time()

    per_episode_summaries: list[dict[str, Any]] = []
    step_rows_out_path = out_dir / f"stochastic_eval{suffix}.ndjson"
    summary_path = out_dir / f"stochastic_eval{suffix}.summary.json"
    # Open NDJSON output with one header row, then episode rows interleaved
    header = {
        "_header": True,
        "tool": "tools/h5_stochastic_eval.py",
        "config": str(args.config),
        "seeds": seeds,
        "replicates": int(args.replicates),
        "max_steps": int(args.max_steps),
        "deterministic": False,
        "policy_sample_seed_rule": (
            "eval_seed + 10_000_000 + replicate_idx"
        ),
        "models": fingerprints,
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fh = step_rows_out_path.open("w", encoding="utf-8", newline="")
    try:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        episode_id = 0
        # K5.3 per-label step-level accumulators. Populated from step_rows
        # as they stream out so the per-label classifier can use vectorised
        # step-weighted statistics rather than per-episode means.
        per_label_step_data: dict[str, dict[str, list]] = {
            label: {
                "p_left": [],
                "p_stay": [],
                "p_right": [],
                "sampled": [],
                "argmax": [],
            }
            for label, _, _ in loaded
        }
        for label, model, _ in loaded:
            eval_run_dir = out_dir / f"godot_{label}{suffix}"
            eval_run_dir.mkdir(parents=True, exist_ok=True)
            if godot_extra:
                env = make_env(
                    env_id, n_envs=1, seed=base_seed, mode="eval",
                    run_dir=str(eval_run_dir), **godot_extra,
                )
            else:
                env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval")
            try:
                for eval_seed in seeds:
                    for rep in range(int(args.replicates)):
                        policy_sample_seed = (
                            int(eval_seed) + POLICY_SAMPLE_SEED_OFFSET + int(rep)
                        )
                        ep_summary, step_rows = run_one_episode(
                            model=model,
                            env=env,
                            eval_seed=int(eval_seed),
                            replicate_idx=int(rep),
                            policy_sample_seed=policy_sample_seed,
                            max_steps=int(args.max_steps),
                            episode_id=episode_id,
                            label=label,
                        )
                        per_episode_summaries.append(ep_summary)
                        for sr in step_rows:
                            fh.write(json.dumps(sr, separators=(",", ":")) + "\n")
                            d = per_label_step_data[sr["label"]]
                            d["p_left"].append(float(sr["p_left"]))
                            d["p_stay"].append(float(sr["p_stay"]))
                            d["p_right"].append(float(sr["p_right"]))
                            d["sampled"].append(int(sr["sampled_action"]))
                            d["argmax"].append(int(sr["deterministic_argmax"]))
                        episode_id += 1
                        print(
                            f"  ep {episode_id - 1} {label} seed={eval_seed} rep={rep} "
                            f"len={ep_summary['episode_length']} "
                            f"coll={ep_summary['collision']} "
                            f"to={ep_summary['timeout']} "
                            f"diff_frac={ep_summary['sampled_differs_from_argmax_fraction']:.3f} "
                            f"left_frac={ep_summary['sampled_action_fractions']['left']:.3f} "
                            f"cls={ep_summary['classification']}"
                        )
            finally:
                try:
                    env.close()
                except Exception:
                    pass
    finally:
        fh.close()

    elapsed = time.time() - started_at
    # Aggregate by label
    per_label: dict[str, Any] = {}
    for label, _, fp in loaded:
        eps = [e for e in per_episode_summaries if e["label"] == label]
        lengths = [e["episode_length"] for e in eps]
        rewards = [e["total_reward"] for e in eps]
        diffs = [e["sampled_differs_from_argmax_fraction"] for e in eps]
        left_fracs = [e["sampled_action_fractions"]["left"] for e in eps]
        stay_fracs = [e["sampled_action_fractions"]["stay"] for e in eps]
        right_fracs = [e["sampled_action_fractions"]["right"] for e in eps]
        collisions = sum(1 for e in eps if e["collision"])
        timeouts = sum(1 for e in eps if e["timeout"])
        n = len(eps)
        # Per-eval-seed cross-replicate stats
        seeds_seen = sorted({e["eval_seed"] for e in eps})
        per_seed: list[dict[str, Any]] = []
        any_seed_differs_terminal = 0
        det_baselines = DETERMINISTIC_BASELINES.get(label, {})
        for s in seeds_seen:
            reps_for_seed = [e for e in eps if e["eval_seed"] == s]
            seed_lengths = [e["episode_length"] for e in reps_for_seed]
            seed_colls = [e["collision"] for e in reps_for_seed]
            seed_to = [e["timeout"] for e in reps_for_seed]
            # Did any replicate differ in terminal outcome vs deterministic baseline?
            terminal_differs_count = 0
            det = det_baselines.get(s)
            if det is not None:
                for e in reps_for_seed:
                    if (
                        e["collision"] != det["collision"]
                        or e["timeout"] != det["timeout"]
                    ):
                        terminal_differs_count += 1
                if terminal_differs_count > 0:
                    any_seed_differs_terminal += 1
            per_seed.append({
                "eval_seed": int(s),
                "n_replicates": int(len(reps_for_seed)),
                "length_mean": float(np.mean(seed_lengths)),
                "length_min": int(min(seed_lengths)),
                "length_max": int(max(seed_lengths)),
                "length_std": float(np.std(seed_lengths)),
                "collision_rate": float(np.mean(seed_colls)),
                "timeout_rate": float(np.mean(seed_to)),
                "deterministic_baseline": det,
                "terminal_differs_count": int(terminal_differs_count),
            })
        cls_counts: dict[str, int] = {}
        for e in eps:
            cls_counts[e["classification"]] = cls_counts.get(e["classification"], 0) + 1
        per_label[label] = {
            "n_episodes": int(n),
            "n_eval_seeds": int(len(seeds_seen)),
            "n_replicates_per_seed": int(args.replicates),
            "episode_length": {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "min": int(min(lengths)),
                "max": int(max(lengths)),
                "std": float(np.std(lengths)),
            },
            "total_reward_mean": float(np.mean(rewards)),
            "collision_rate": float(collisions / n),
            "timeout_rate": float(timeouts / n),
            "sampled_differs_from_argmax_fraction": {
                "mean": float(np.mean(diffs)),
                "min": float(min(diffs)),
                "max": float(max(diffs)),
            },
            "action_fractions_mean": {
                "left": float(np.mean(left_fracs)),
                "stay": float(np.mean(stay_fracs)),
                "right": float(np.mean(right_fracs)),
            },
            "classification_counts": cls_counts,
            "seeds_with_any_terminal_diff": int(any_seed_differs_terminal),
            "per_seed": per_seed,
        }
        # K5.3 step-weighted block + classification + overlay flag.
        step_data = per_label_step_data.get(label, {})
        per_label[label]["k5_3"] = _compute_k5_3_block(
            step_p_left=step_data.get("p_left", []),
            step_p_stay=step_data.get("p_stay", []),
            step_p_right=step_data.get("p_right", []),
            step_sampled=step_data.get("sampled", []),
            step_argmax=step_data.get("argmax", []),
            sampled_mean_episode_length=per_label[label]["episode_length"]["mean"],
        )

    summary = {
        "_header": header,
        "elapsed_seconds": float(elapsed),
        "n_episodes": int(len(per_episode_summaries)),
        "per_label": per_label,
        "episodes": per_episode_summaries,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"phase_j_stochastic done: episodes={len(per_episode_summaries)} "
        f"elapsed={elapsed:.1f}s out={summary_path}"
    )
    for label, agg in per_label.items():
        print(
            f"  {label}: n={agg['n_episodes']} "
            f"len_mean={agg['episode_length']['mean']:.1f} "
            f"coll_rate={agg['collision_rate']:.2f} "
            f"to_rate={agg['timeout_rate']:.2f} "
            f"diff_frac_mean={agg['sampled_differs_from_argmax_fraction']['mean']:.3f} "
            f"left_frac_mean={agg['action_fractions_mean']['left']:.3f} "
            f"seeds_with_term_diff={agg['seeds_with_any_terminal_diff']}"
        )
        k5_3 = agg.get("k5_3", {})
        cls = k5_3.get("k5_3_classification", {})
        print(
            f"    k5_3 verdict={cls.get('primary_bucket', 'UNKNOWN')} "
            f"near_tie_bias={cls.get('near_tie_bias_overlay', False)} "
            f"diff_step_frac={k5_3.get('sampled_differs_from_argmax_step_fraction', 0.0):.3f} "
            f"ent_mean={k5_3.get('entropy_nats', {}).get('mean', 0.0):.3f} "
            f"margin_lt_005_frac={k5_3.get('top1_top2_probability_margin_lt_0p05_fraction', 0.0):.3f}"
        )
    return 0


# Known deterministic baselines (from on-disk eval summaries). Phase J reads
# these for the per-seed terminal-differs-from-deterministic count.
DETERMINISTIC_BASELINES: dict[str, dict[int, dict[str, Any]]] = {
    "phase_e_seed2": {
        1000: {"episode_length": 1383, "collision": True, "timeout": False},
        1001: {"episode_length": 483, "collision": True, "timeout": False},
        1002: {"episode_length": 1293, "collision": True, "timeout": False},
        1003: {"episode_length": 603, "collision": True, "timeout": False},
        1004: {"episode_length": 1443, "collision": True, "timeout": False},
        1005: {"episode_length": 363, "collision": True, "timeout": False},
        1006: {"episode_length": 573, "collision": True, "timeout": False},
        1007: {"episode_length": 273, "collision": True, "timeout": False},
        1008: {"episode_length": 1800, "collision": False, "timeout": True},
        1009: {"episode_length": 243, "collision": True, "timeout": False},
    },
    # phase_g_seed2 eval-trajectory data is byte-identical to phase_e_seed2 per
    # docs/h5-phase-g-shaped-evidence.md (six trained networks reduce to two
    # eval-trajectory equivalence classes keyed by train_seed only).
    "phase_g_seed2": {
        1000: {"episode_length": 1383, "collision": True, "timeout": False},
        1001: {"episode_length": 483, "collision": True, "timeout": False},
        1002: {"episode_length": 1293, "collision": True, "timeout": False},
        1003: {"episode_length": 603, "collision": True, "timeout": False},
        1004: {"episode_length": 1443, "collision": True, "timeout": False},
        1005: {"episode_length": 363, "collision": True, "timeout": False},
        1006: {"episode_length": 573, "collision": True, "timeout": False},
        1007: {"episode_length": 273, "collision": True, "timeout": False},
        1008: {"episode_length": 1800, "collision": False, "timeout": True},
        1009: {"episode_length": 243, "collision": True, "timeout": False},
    },
    # K5.1 alpha=0.30 seed=0 10k-step shaped checkpoint. Per-seed deterministic
    # eval lengths supplied by GPT K5.3 execution packet. All 10 seeds collide
    # under deterministic argmax (handoff records 100% stay over 6060 reached
    # eval steps, player_x held at 360.0 with zero lateral motion).
    "k5_1_alpha030_seed0_10k": {
        1000: {"episode_length": 333, "collision": True, "timeout": False},
        1001: {"episode_length": 273, "collision": True, "timeout": False},
        1002: {"episode_length": 843, "collision": True, "timeout": False},
        1003: {"episode_length": 963, "collision": True, "timeout": False},
        1004: {"episode_length": 1203, "collision": True, "timeout": False},
        1005: {"episode_length": 1263, "collision": True, "timeout": False},
        1006: {"episode_length": 543, "collision": True, "timeout": False},
        1007: {"episode_length": 183, "collision": True, "timeout": False},
        1008: {"episode_length": 183, "collision": True, "timeout": False},
        1009: {"episode_length": 273, "collision": True, "timeout": False},
    },
}


if __name__ == "__main__":
    raise SystemExit(main())
