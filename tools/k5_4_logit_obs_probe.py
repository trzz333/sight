"""K5.4 replay-derived logit/obs oracle-alignment probe.

Drives the production Godot H3 pixel env (the same env used in K5.1
training and K5.3 stochastic eval) with multiple collector policies,
labels every aligned post-step reward_state with the K5.2 hazard-
reactive 1-step oracle, queries the K5.1 alpha=0.30 CnnPolicy on the
post-step pixel observation, and emits per-step records of logits /
probabilities / entropy / argmax / oracle alignment plus per-geometry
bucket summary statistics and a K5.4 classification verdict.

Per the K5.4 execution packet:
- replay-derived, NOT synthetic. Uses real Godot viewport pixels.
- no training, no env code edit, no frame_stack / capacity / budget
  change, no reward-shape revision.
- reuses helpers from tools/h5_logit_compare.py (load_and_fingerprint,
  get_action_logits, obs_hash) and tools/k5_2_env_dynamics_probe.py
  (hazard_reactive_oracle, screen / kinematic constants).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

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
from k5_2_env_dynamics_probe import (  # noqa: E402
    hazard_reactive_oracle,
    HAZARD_SPEED_PX_STEP,
)

ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}
DEFAULT_COLLECTORS = ("stay", "left", "right", "oracle", "seeded_random", "sweep")

# Sweep pattern per K5.4 packet:
#   left 60, stay 15, right 120, stay 15, left 120, cycling.
SWEEP_PATTERN: tuple[int, ...] = tuple(
    [0] * 60 + [1] * 15 + [2] * 120 + [1] * 15 + [0] * 120
)
SWEEP_PERIOD = len(SWEEP_PATTERN)


# ---------------------------------------------------------------------------
# Collector policies
# ---------------------------------------------------------------------------


class CollectorPolicy:
    """Action source for one replay episode. Each instance is per-episode."""

    def __init__(self, name: str, seed: int) -> None:
        if name not in DEFAULT_COLLECTORS:
            raise ValueError(f"unknown collector: {name!r}")
        self.name = name
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.step = 0

    def act(self, last_reward_state: dict[str, Any] | None) -> int:
        """Return wire action 0/1/2 for this step."""
        n = self.name
        if n == "stay":
            a = 1
        elif n == "left":
            a = 0
        elif n == "right":
            a = 2
        elif n == "oracle":
            if last_reward_state is None:
                a = 1
            else:
                a = int(hazard_reactive_oracle(last_reward_state))
        elif n == "seeded_random":
            a = int(self.rng.randint(0, 2))
        elif n == "sweep":
            a = int(SWEEP_PATTERN[self.step % SWEEP_PERIOD])
        else:
            raise ValueError(self.name)
        self.step += 1
        return a


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _player_x_bucket(x: float) -> str:
    if x <= 60.0:
        return "left_wall"
    if 300.0 <= x <= 420.0:
        return "center"
    if x >= 660.0:
        return "right_wall"
    return "other"


def _arrival_bucket(arrival_steps: float | None) -> str:
    if arrival_steps is None:
        return "none"
    if arrival_steps <= 15.0:
        return "le15"
    if arrival_steps <= 30.0:
        return "le30"
    if arrival_steps <= 60.0:
        return "le60"
    return "gt60"


def _dx_bucket(dx: float | None) -> str:
    if dx is None:
        return "none"
    if dx < -40.0:
        return "left"
    if abs(dx) <= 40.0:
        return "centerline"
    return "right"


def compute_geometry_fields(reward_state: dict[str, Any]) -> dict[str, Any]:
    """Pull player_x, player_x_bucket, nearest hazard fields from reward_state.

    Mirrors the K5.2 oracle's view: only hazards above the player with
    arrival_steps <= 60 count as imminent. Tied to HAZARD_SPEED_PX_STEP
    from tools/k5_2_env_dynamics_probe.py so the geometry definitions
    here and in the oracle stay in lock-step.
    """
    px = float(reward_state.get("player_x", 360.0))
    py = float(reward_state.get("player_y", 500.0))
    hazards = reward_state.get("hazards_above", []) or []
    above: list[tuple[float, float, float]] = []  # (hx, hy, arrival_steps)
    for h in hazards:
        try:
            hy = float(h["y"])
            hx = float(h["x"])
        except (KeyError, TypeError, ValueError):
            continue
        vd = py - hy
        if vd <= 0:
            continue
        s = vd / HAZARD_SPEED_PX_STEP
        above.append((hx, hy, s))
    above.sort(key=lambda t: t[2])
    imminent = [t for t in above if t[2] <= 60.0]
    if imminent:
        nx, _ny, arrival_steps = imminent[0]
        nearest_dx = float(nx - px)
        nearest_abs_dx = float(abs(nearest_dx))
        imminent_threat = True
    else:
        nearest_dx = None
        nearest_abs_dx = None
        arrival_steps = None
        imminent_threat = False
    return {
        "player_x": float(px),
        "player_y": float(py),
        "player_x_bucket": _player_x_bucket(px),
        "imminent_threat": bool(imminent_threat),
        "nearest_dx": nearest_dx,
        "nearest_abs_dx": nearest_abs_dx,
        "arrival_steps": (
            float(arrival_steps) if arrival_steps is not None else None
        ),
        "arrival_bucket": _arrival_bucket(arrival_steps),
        "nearest_dx_bucket": _dx_bucket(nearest_dx),
        "hazard_count_above": int(len(above)),
        "imminent_hazard_count_le60": int(len(imminent)),
    }


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------


def _extract_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
        return dict(infos[0])
    return {}


def _query_logits(
    model: Any, obs_arr: np.ndarray, oracle_label: int,
) -> dict[str, Any]:
    """Run K5.1 policy on one obs and produce alignment fields."""
    logits, probs, entropy_v, argmax = h5lc.get_action_logits(model, obs_arr)
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    sorted_idx = np.argsort(probs_arr)[::-1]
    sorted_probs = probs_arr[sorted_idx]
    margin = float(sorted_probs[0] - sorted_probs[1])
    rank = int(np.where(sorted_idx == int(oracle_label))[0][0]) + 1
    p_oracle = float(probs_arr[int(oracle_label)])
    non_oracle = [
        float(probs_arr[i]) for i in (0, 1, 2) if i != int(oracle_label)
    ]
    p_oracle_minus_best_wrong = float(p_oracle - max(non_oracle))
    return {
        "logits": [float(x) for x in logits.tolist()],
        "probs": [float(x) for x in probs_arr.tolist()],
        "entropy": float(entropy_v),
        "argmax": int(argmax),
        "argmax_matches_oracle": bool(int(argmax) == int(oracle_label)),
        "top1_top2_margin": margin,
        "oracle_rank": rank,
        "p_oracle": p_oracle,
        "p_oracle_minus_best_wrong": p_oracle_minus_best_wrong,
    }


def run_one_episode(
    model: Any,
    env: Any,
    eval_seed: int,
    collector_name: str,
    max_steps: int,
    episode_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one episode under one collector policy and return (rows, summary)."""
    pol_seed = int(eval_seed) * 1000 + DEFAULT_COLLECTORS.index(collector_name)
    pol = CollectorPolicy(collector_name, pol_seed)
    try:
        env.seed(int(eval_seed))
    except (AttributeError, TypeError):
        pass
    obs = env.reset()
    last_reward_state: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    step = 0
    collision = False
    timeout = False
    t0 = time.time()
    while step < int(max_steps):
        action = pol.act(last_reward_state)
        obs, _rewards, dones, infos = env.step(np.asarray([int(action)]))
        info = _extract_info(infos)
        # Godot wire info (which contains ``reward_state``) is forwarded
        # under ``info["godot_info"]`` per src/sight_agent/rl/godot_env.py
        # _build_info. Read the nested path; the flat ``info.reward_state``
        # path that K5.3 used opportunistically does not exist.
        godot_info = info.get("godot_info") if isinstance(info, dict) else None
        reward_state = (
            godot_info.get("reward_state") if isinstance(godot_info, dict) else None
        )
        if isinstance(reward_state, dict):
            obs_arr = np.asarray(obs)
            oracle_label = int(hazard_reactive_oracle(reward_state))
            q = _query_logits(model, obs_arr, oracle_label)
            geom = compute_geometry_fields(reward_state)
            rows.append({
                "collector": collector_name,
                "eval_seed": int(eval_seed),
                "episode_id": int(episode_id),
                "step": int(step),
                "action_taken": int(action),
                "obs_hash": h5lc.obs_hash(obs_arr),
                "oracle_label": int(oracle_label),
                **q,
                **geom,
            })
            last_reward_state = reward_state
        step += 1
        if bool(np.asarray(dones).any()):
            truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
            collision = not truncated
            timeout = truncated
            break
    else:
        # Loop completed without done.
        timeout = True
    elapsed = time.time() - t0
    return rows, {
        "collector": collector_name,
        "eval_seed": int(eval_seed),
        "episode_id": int(episode_id),
        "policy_seed": int(pol_seed),
        "episode_length": int(step),
        "rows_recorded": int(len(rows)),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Aggregation and K5.4 classification
# ---------------------------------------------------------------------------


def _np_stats(vals: list[float]) -> dict[str, float | int]:
    a = np.asarray(vals, dtype=np.float64)
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


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(
        sum(1 for r in rows if r["argmax_matches_oracle"]) / len(rows)
    )


def _bucket_stats(
    rows: list[dict[str, Any]], key: str, ordered_values: list[str],
) -> dict[str, Any]:
    """Per-bucket accuracy, p_oracle stats, entropy stats, count."""
    out: dict[str, Any] = {}
    for v in ordered_values:
        sub = [r for r in rows if r.get(key) == v]
        if not sub:
            out[v] = {"n": 0}
            continue
        out[v] = {
            "n": int(len(sub)),
            "oracle_top1_accuracy": _accuracy(sub),
            "mean_oracle_rank": float(np.mean([r["oracle_rank"] for r in sub])),
            "p_oracle": _np_stats([r["p_oracle"] for r in sub]),
            "p_oracle_minus_best_wrong": _np_stats(
                [r["p_oracle_minus_best_wrong"] for r in sub]
            ),
            "entropy": _np_stats([r["entropy"] for r in sub]),
            "top1_top2_margin": _np_stats(
                [r["top1_top2_margin"] for r in sub]
            ),
            "argmax_fractions": {
                ACTION_NAMES[a]: float(
                    sum(1 for r in sub if r["argmax"] == a) / len(sub)
                ) for a in (0, 1, 2)
            },
            "oracle_label_counts": {
                ACTION_NAMES[a]: int(
                    sum(1 for r in sub if r["oracle_label"] == a)
                ) for a in (0, 1, 2)
            },
        }
    return out


def classify_k5_4(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the K5.4 packet decision rules. Ordered primary buckets."""
    n_total = int(summary.get("n_samples", 0))
    label_counts = summary.get("oracle_label_counts", {})
    n_left = int(label_counts.get("left", 0))
    n_stay = int(label_counts.get("stay", 0))
    n_right = int(label_counts.get("right", 0))
    overall_acc = float(summary.get("overall_oracle_top1_accuracy", 0.0))
    acc_by_label = summary.get("oracle_top1_accuracy_by_oracle_label", {})
    acc_left = acc_by_label.get("left", None)
    acc_stay = acc_by_label.get("stay", None)
    acc_right = acc_by_label.get("right", None)
    argmax_fractions = summary.get("argmax_fractions_overall", {})
    stay_argmax_frac = float(argmax_fractions.get("stay", 0.0))

    insufficient = (
        n_total < 3000
        or n_left < 200
        or n_stay < 200
        or n_right < 200
    )
    inputs = {
        "n_total": n_total,
        "n_left": n_left,
        "n_stay": n_stay,
        "n_right": n_right,
        "overall_oracle_top1_accuracy": overall_acc,
        "acc_left": acc_left,
        "acc_stay": acc_stay,
        "acc_right": acc_right,
        "stay_argmax_fraction": stay_argmax_frac,
    }
    if insufficient:
        return {
            "primary_bucket": "INSUFFICIENT-COVERAGE",
            "inputs_used": inputs,
            "reason": "n_total<3000 or any oracle label count <200",
        }

    al = float(acc_left) if acc_left is not None else 0.0
    ar = float(acc_right) if acc_right is not None else 0.0
    # LOGIT-GEOMETRY-ALIGNED
    if overall_acc >= 0.60 and al >= 0.50 and ar >= 0.50:
        return {
            "primary_bucket": "LOGIT-GEOMETRY-ALIGNED",
            "inputs_used": inputs,
            "reason": "overall>=0.60 and per-side>=0.50",
        }
    # STAY-BIASED-MISRANKING
    if stay_argmax_frac >= 0.90 and al < 0.30 and ar < 0.30:
        return {
            "primary_bucket": "STAY-BIASED-MISRANKING",
            "inputs_used": inputs,
            "reason": "argmax stay fraction>=0.90 and non-stay oracle accs<0.30",
        }
    # DIRECTIONAL-MISRANKING
    if (al >= 0.50 and ar < 0.30) or (ar >= 0.50 and al < 0.30):
        return {
            "primary_bucket": "DIRECTIONAL-MISRANKING",
            "inputs_used": inputs,
            "reason": "one non-stay label >=0.50, the other <0.30",
        }
    # NO-GEOMETRY-CORRELATION
    if overall_acc <= 0.40:
        return {
            "primary_bucket": "NO-GEOMETRY-CORRELATION",
            "inputs_used": inputs,
            "reason": "overall oracle_top1_accuracy<=0.40 with sufficient coverage",
        }
    return {
        "primary_bucket": "UNKNOWN",
        "inputs_used": inputs,
        "reason": "no rule matched (mixed signal)",
    }


def build_summary(
    rows: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    header: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Aggregate per-step rows into a K5.4 summary block + classification."""
    n_total = len(rows)
    by_collector = Counter(r["collector"] for r in rows)
    by_oracle = Counter(int(r["oracle_label"]) for r in rows)
    by_argmax = Counter(int(r["argmax"]) for r in rows)
    by_px = Counter(r["player_x_bucket"] for r in rows)
    by_arrival = Counter(r["arrival_bucket"] for r in rows)
    by_dx = Counter(r["nearest_dx_bucket"] for r in rows)
    by_imminent = Counter(bool(r["imminent_threat"]) for r in rows)

    overall_acc = _accuracy(rows) if rows else None
    acc_by_oracle: dict[str, float | None] = {}
    for a in (0, 1, 2):
        sub = [r for r in rows if int(r["oracle_label"]) == a]
        acc_by_oracle[ACTION_NAMES[a]] = _accuracy(sub)
    mean_oracle_rank_by_label: dict[str, float | None] = {}
    mean_p_oracle_by_label: dict[str, float | None] = {}
    for a in (0, 1, 2):
        sub = [r for r in rows if int(r["oracle_label"]) == a]
        if sub:
            mean_oracle_rank_by_label[ACTION_NAMES[a]] = float(
                np.mean([r["oracle_rank"] for r in sub])
            )
            mean_p_oracle_by_label[ACTION_NAMES[a]] = float(
                np.mean([r["p_oracle"] for r in sub])
            )
        else:
            mean_oracle_rank_by_label[ACTION_NAMES[a]] = None
            mean_p_oracle_by_label[ACTION_NAMES[a]] = None

    confusion: dict[str, int] = {}
    for r in rows:
        key = (
            f"oracle_{ACTION_NAMES[int(r['oracle_label'])]}"
            f"__argmax_{ACTION_NAMES[int(r['argmax'])]}"
        )
        confusion[key] = confusion.get(key, 0) + 1

    argmax_fractions_overall = {
        ACTION_NAMES[a]: float(by_argmax.get(a, 0) / n_total if n_total else 0.0)
        for a in (0, 1, 2)
    }
    entropy_overall = (
        _np_stats([r["entropy"] for r in rows]) if rows else {"n": 0}
    )
    margin_overall = (
        _np_stats([r["top1_top2_margin"] for r in rows]) if rows else {"n": 0}
    )
    p_oracle_overall = (
        _np_stats([r["p_oracle"] for r in rows]) if rows else {"n": 0}
    )
    p_oracle_minus_best_wrong_overall = (
        _np_stats([r["p_oracle_minus_best_wrong"] for r in rows])
        if rows else {"n": 0}
    )


    px_bucket_stats = _bucket_stats(
        rows, "player_x_bucket", ["left_wall", "center", "right_wall", "other"],
    )
    arrival_bucket_stats = _bucket_stats(
        rows, "arrival_bucket", ["none", "le15", "le30", "le60", "gt60"],
    )
    dx_bucket_stats = _bucket_stats(
        rows, "nearest_dx_bucket", ["none", "left", "centerline", "right"],
    )

    summary: dict[str, Any] = {
        "_header": header,
        "elapsed_seconds": float(elapsed_seconds),
        "n_episodes": int(len(episode_summaries)),
        "n_samples": int(n_total),
        "samples_by_collector": {k: int(v) for k, v in by_collector.items()},
        "samples_by_oracle_label": {
            ACTION_NAMES[a]: int(by_oracle.get(a, 0)) for a in (0, 1, 2)
        },
        "samples_by_player_x_bucket": {
            k: int(v) for k, v in by_px.items()
        },
        "samples_by_arrival_bucket": {
            k: int(v) for k, v in by_arrival.items()
        },
        "samples_by_nearest_dx_bucket": {
            k: int(v) for k, v in by_dx.items()
        },
        "samples_by_imminent_threat": {
            str(k): int(v) for k, v in by_imminent.items()
        },
        "oracle_label_counts": {
            ACTION_NAMES[a]: int(by_oracle.get(a, 0)) for a in (0, 1, 2)
        },
        "argmax_counts_overall": {
            ACTION_NAMES[a]: int(by_argmax.get(a, 0)) for a in (0, 1, 2)
        },
        "argmax_fractions_overall": argmax_fractions_overall,
        "overall_oracle_top1_accuracy": overall_acc,
        "oracle_top1_accuracy_by_oracle_label": acc_by_oracle,
        "mean_oracle_rank_by_label": mean_oracle_rank_by_label,
        "mean_p_oracle_by_label": mean_p_oracle_by_label,
        "confusion_matrix_oracle_x_argmax": confusion,
        "entropy_overall": entropy_overall,
        "top1_top2_margin_overall": margin_overall,
        "p_oracle_overall": p_oracle_overall,
        "p_oracle_minus_best_wrong_overall": p_oracle_minus_best_wrong_overall,
        "by_player_x_bucket": px_bucket_stats,
        "by_arrival_bucket": arrival_bucket_stats,
        "by_nearest_dx_bucket": dx_bucket_stats,
        "episodes": episode_summaries,
    }
    summary["k5_4_classification"] = classify_k5_4(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def parse_seed_range(spec: str) -> list[int]:
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


def parse_collectors(spec: str) -> list[str]:
    out: list[str] = []
    for tok in spec.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in DEFAULT_COLLECTORS:
            raise ValueError(
                f"unknown collector {t!r}, must be one of {DEFAULT_COLLECTORS}"
            )
        if t in out:
            raise ValueError(f"collector repeated: {t!r}")
        out.append(t)
    if not out:
        raise ValueError("at least one collector required")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="k5_4_logit_obs_probe",
        description=(
            "K5.4 replay-derived logit/obs oracle-alignment probe on the "
            "K5.1 alpha=0.30 CnnPolicy."
        ),
    )
    p.add_argument("--config", required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument(
        "--collector-policies", default=",".join(DEFAULT_COLLECTORS),
    )
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label-suffix", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    seeds = parse_seed_range(args.seeds)
    if not seeds:
        raise ValueError("no eval seeds parsed")
    collectors = parse_collectors(args.collector_policies)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + args.label_suffix) if args.label_suffix else ""

    model, fp = h5lc.load_and_fingerprint(Path(args.model_dir))

    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)
    started_at = time.time()

    eval_run_dir = out_dir / f"godot_{args.model_label}{suffix}"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    if godot_extra:
        env = make_env(
            env_id, n_envs=1, seed=base_seed, mode="eval",
            run_dir=str(eval_run_dir), **godot_extra,
        )
    else:
        env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval")

    header = {
        "_header": True,
        "tool": "tools/k5_4_logit_obs_probe.py",
        "config": str(args.config),
        "model": {args.model_label: fp},
        "seeds": seeds,
        "collectors": collectors,
        "max_steps": int(args.max_steps),
        "label_suffix": args.label_suffix,
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ndjson_path = out_dir / f"k5_4_logit_obs_probe{suffix}.ndjson"
    summary_path = out_dir / f"k5_4_logit_obs_probe{suffix}.summary.json"

    all_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    episode_id = 0
    fh = ndjson_path.open("w", encoding="utf-8", newline="")
    try:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for collector_name in collectors:
            for eval_seed in seeds:
                rows, ep = run_one_episode(
                    model=model,
                    env=env,
                    eval_seed=int(eval_seed),
                    collector_name=collector_name,
                    max_steps=int(args.max_steps),
                    episode_id=episode_id,
                )
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                all_rows.extend(rows)
                episode_summaries.append(ep)
                episode_id += 1
                print(
                    f"  ep {ep['episode_id']} {collector_name} "
                    f"seed={ep['eval_seed']} "
                    f"len={ep['episode_length']} "
                    f"rows={ep['rows_recorded']} "
                    f"coll={ep['collision']} "
                    f"to={ep['timeout']}"
                )
    finally:
        try:
            env.close()
        except Exception:
            pass
        fh.close()

    elapsed = time.time() - started_at
    summary = build_summary(all_rows, episode_summaries, header, elapsed)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cls = summary.get("k5_4_classification", {})
    print(
        f"k5_4 done: episodes={len(episode_summaries)} "
        f"samples={len(all_rows)} elapsed={elapsed:.1f}s "
        f"verdict={cls.get('primary_bucket', 'UNKNOWN')} "
        f"acc_overall={summary.get('overall_oracle_top1_accuracy')} "
        f"out={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
