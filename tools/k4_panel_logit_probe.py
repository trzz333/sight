"""Phase K K4.0 panel-logit mechanism diagnostic.

Probes three policy states on the same fixed observation-conditioning
panel and records row-level raw logits, softmax probs, deterministic
argmax actions, and top1-top2 margins to localize the within-regime
deterministic-argmax invariance documented in K3.5c.

Per GPT K4.0 scope:
- No training. Use existing K3.5c checkpoints on disk.
- Probe fresh_seed0_init (zero-step CnnPolicy under H5 entropy config),
  k3_5c_2048 model.zip, k3_5c_10000 model.zip.
- Reuse the fixed observation-conditioning panel machinery from
  tools/h5_training_entropy_probe.py (build_fixed_observation_panel,
  snapshot_policy_state, snapshot_action_net).
- Record row-level panel data AND per-model summaries.

Classification (per GPT K4.0 scope):
- K4-A feature extractor uniformity: raw obs diverse, CNN features
  near-uniform, logits nearly identical.
- K4-B action-head near-tie / deterministic tie-break: CNN/latent_pi
  diverse, logits nearly identical, margins tiny.
- K4-C action-head decision boundary pinned: CNN/latent_pi diverse,
  logits differ across rows, same argmax dominates with meaningful
  margins.
- K4-D eval-invisible logit drift: 2048 and 10000 logits differ
  materially but argmax rows identical.
- K4-E panel not representative: panel argmax differs between 2048
  and 10000 even though external eval is bit-identical.
- K4-F init-surface persistence: fresh init already has same argmax
  as both K3.5c checkpoints.
- K4-G early training lands on stable deterministic surface: fresh
  init differs, 2048 and 10000 match each other.

Outputs:
- runs/phase_k/k4_panel_logit_probe.json
- runs/phase_k/k4_panel_logit_probe_panel_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_TOOLS = _REPO_ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402

from h5_training_entropy_probe import (  # noqa: E402
    ACTION_NAMES,
    build_fixed_observation_panel,
    snapshot_action_net,
    snapshot_policy_state,
)


# Classification thresholds for K4-A through K4-G. Diagnostic, not tuned.
# Values report alongside the verdict; the verdict is a heuristic over the
# row-level data, the substantive evidence is the per-row logits in the CSV.
DIM_STD_NEAR_UNIFORM = 1e-3
DIM_STD_DIVERSE = 1e-2
LOGIT_NEAR_UNIFORM_RANGE_ACROSS_ROWS = 0.01
MARGIN_TINY = 0.1
INTER_MODEL_LOGIT_LINF_NEGLIGIBLE = 1e-6
INTER_MODEL_LOGIT_LINF_MATERIAL = 0.1
ARGMAX_MATCH_TIGHT = 0.999

def _sha256_of_file(path: Path) -> str:
    """Return uppercase SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def snapshot_per_row(
    policy: Any,
    obs_tensor: th.Tensor,
    panel_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per-row diagnostic forward pass on the fixed panel.

    Mirrors the actor-path used by snapshot_policy_state but emits a
    per-row record so K4 classification can localize where (if anywhere)
    diversity dies and where logits move without crossing argmax
    boundaries. Pure forward pass, no grad, does not affect optimizer
    state.
    """
    with th.no_grad():
        features = policy.extract_features(obs_tensor)
        latent_pi, _latent_vf = policy.mlp_extractor(features)
        raw_logits = policy.action_net(latent_pi)
        probs = th.softmax(raw_logits, dim=-1)
        argmax = raw_logits.argmax(dim=-1)
        top2 = th.topk(raw_logits, k=2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
    logits_np = raw_logits.detach().cpu().numpy().astype(np.float64)
    probs_np = probs.detach().cpu().numpy().astype(np.float64)
    argmax_np = argmax.detach().cpu().numpy().astype(np.int64)
    margins_np = margins.detach().cpu().numpy().astype(np.float64)

    rows: list[dict[str, Any]] = []
    items = panel_metadata.get("items") or []
    for i in range(logits_np.shape[0]):
        meta = items[i] if i < len(items) else {}
        rows.append({
            "row_idx": int(i),
            "panel_seed": int(meta.get("panel_seed", -1)),
            "prefix_name": str(meta.get("prefix_name", f"unknown_{i}")),
            "prefix_length": int(meta.get("prefix_length", -1)),
            "raw_logit_left": float(logits_np[i, 0]),
            "raw_logit_stay": float(logits_np[i, 1]),
            "raw_logit_right": float(logits_np[i, 2]),
            "prob_left": float(probs_np[i, 0]),
            "prob_stay": float(probs_np[i, 1]),
            "prob_right": float(probs_np[i, 2]),
            "det_argmax_action": ACTION_NAMES[int(argmax_np[i])],
            "det_argmax_idx": int(argmax_np[i]),
            "margin_top1_top2": float(margins_np[i]),
        })
    return rows

def probe_one_policy(
    policy: Any,
    obs_tensor: th.Tensor,
    panel_metadata: dict[str, Any],
    name: str,
    model_zip_path: Path | None = None,
) -> dict[str, Any]:
    """Full per-model probe: action-net snapshot + summary + row-level."""
    rows = snapshot_per_row(policy, obs_tensor, panel_metadata)
    state = snapshot_policy_state(policy, obs_tensor)
    action_net = snapshot_action_net(policy)

    model_sha256 = None
    model_path_str = None
    if model_zip_path is not None:
        path = Path(model_zip_path)
        if path.exists():
            model_sha256 = _sha256_of_file(path)
            model_path_str = str(path)

    # Per-action raw-logit range across panel rows. Cheap row aggregate
    # to characterize how much logits vary across panel input on this
    # model.
    logit_range_per_action: dict[str, float] = {}
    for action_name in ACTION_NAMES.values():
        vals = [r[f"raw_logit_{action_name}"] for r in rows]
        logit_range_per_action[action_name] = float(max(vals) - min(vals))
    margins = [r["margin_top1_top2"] for r in rows]
    mean_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "name": name,
        "model_sha256": model_sha256,
        "model_path": model_path_str,
        "panel_top_argmax_action": state["top_argmax_action"],
        "panel_top_argmax_fraction": state["top_argmax_fraction"],
        "panel_num_det_actions": state["num_det_actions"],
        "panel_det_argmax_counts": state["det_argmax_counts"],
        "mean_logits": state["mean_logits"],
        "logit_std": state["logit_std"],
        "mean_probs": state["mean_probs"],
        "prob_ranges": state["prob_ranges"],
        "logit_range_per_action_across_rows": logit_range_per_action,
        "mean_logit_range_across_rows": float(
            np.mean(list(logit_range_per_action.values()))
        ),
        "mean_margin_top1_top2": mean_margin,
        "feature_chain_diversity": state["feature_chain_diversity"],
        "action_net_row_norms": action_net["row_norms"],
        "action_net_biases": action_net["biases"],
        "action_net_hash_blake2b16": action_net["blake2b16"],
        "panel_rows": rows,
    }

def compare_models(
    probes: dict[str, dict[str, Any]],
    pair_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    """Pairwise per-row comparisons across the probed models.

    For each ordered pair (a, b), zips panel rows by row_idx (panel
    construction is deterministic so row alignment is by index), and
    computes:
    - argmax match count and fraction (raw deterministic argmax equality)
    - per-row raw-logit L1 and Linf delta (a - b across left/stay/right)

    The Linf max across rows is the load-bearing inter-model statistic
    for K4-D (eval-invisible logit drift): nonzero argmax match while
    Linf_max above MATERIAL threshold means logits move but never cross
    a decision boundary.
    """
    pair_summary: dict[str, Any] = {}
    for a, b in pair_keys:
        rows_a = probes[a]["panel_rows"]
        rows_b = probes[b]["panel_rows"]
        if len(rows_a) != len(rows_b):
            raise RuntimeError(
                f"row count mismatch between {a} ({len(rows_a)}) and "
                f"{b} ({len(rows_b)}); panel construction was non-deterministic"
            )
        n_rows = len(rows_a)
        argmax_match = 0
        logit_l1: list[float] = []
        logit_linf: list[float] = []
        per_row_deltas: list[dict[str, Any]] = []
        for ra, rb in zip(rows_a, rows_b):
            if ra["row_idx"] != rb["row_idx"]:
                raise RuntimeError(
                    f"row_idx mismatch between {a} and {b}: "
                    f"{ra['row_idx']} vs {rb['row_idx']}"
                )
            same_argmax = bool(ra["det_argmax_idx"] == rb["det_argmax_idx"])
            if same_argmax:
                argmax_match += 1
            la = np.array(
                [ra["raw_logit_left"], ra["raw_logit_stay"], ra["raw_logit_right"]]
            )
            lb = np.array(
                [rb["raw_logit_left"], rb["raw_logit_stay"], rb["raw_logit_right"]]
            )
            diff = la - lb
            logit_l1.append(float(np.sum(np.abs(diff))))
            logit_linf.append(float(np.max(np.abs(diff))))
            per_row_deltas.append({
                "row_idx": int(ra["row_idx"]),
                "panel_seed": int(ra["panel_seed"]),
                "prefix_name": str(ra["prefix_name"]),
                "argmax_match": same_argmax,
                "argmax_a": ra["det_argmax_action"],
                "argmax_b": rb["det_argmax_action"],
                "delta_logit_left": float(diff[0]),
                "delta_logit_stay": float(diff[1]),
                "delta_logit_right": float(diff[2]),
                "logit_l1": logit_l1[-1],
                "logit_linf": logit_linf[-1],
            })
        pair_summary[f"{a}_vs_{b}"] = {
            "a": a,
            "b": b,
            "n_rows": n_rows,
            "argmax_match_count": argmax_match,
            "argmax_match_fraction": (
                float(argmax_match) / n_rows if n_rows else 0.0
            ),
            "logit_l1_mean": float(np.mean(logit_l1)) if logit_l1 else 0.0,
            "logit_l1_max": float(np.max(logit_l1)) if logit_l1 else 0.0,
            "logit_linf_mean": float(np.mean(logit_linf)) if logit_linf else 0.0,
            "logit_linf_max": float(np.max(logit_linf)) if logit_linf else 0.0,
            "per_row_deltas": per_row_deltas,
        }
    return {"pairs": pair_summary}

def _within_model_signature(probe: dict[str, Any]) -> dict[str, Any]:
    """Within-model K4-A / K4-B / K4-C signature on a single probe.

    Returns the diagnostic flags used by classify_k4 to localize where
    diversity dies inside the actor-critic forward pass.
    """
    fcd = probe["feature_chain_diversity"]
    return {
        "cnn_features_dim_std_mean": fcd["cnn_features"]["dim_std_mean"],
        "latent_pi_dim_std_mean": fcd["latent_pi"]["dim_std_mean"],
        "raw_obs_dim_std_mean": fcd["raw_obs"]["dim_std_mean"],
        "mean_logit_range_across_rows": probe["mean_logit_range_across_rows"],
        "mean_margin_top1_top2": probe["mean_margin_top1_top2"],
        "panel_top_argmax_fraction": probe["panel_top_argmax_fraction"],
        "panel_num_det_actions": probe["panel_num_det_actions"],
        "raw_obs_diverse": bool(
            fcd["raw_obs"]["dim_std_mean"] >= DIM_STD_DIVERSE
        ),
        "cnn_features_near_uniform": bool(
            fcd["cnn_features"]["dim_std_mean"] < DIM_STD_NEAR_UNIFORM
        ),
        "cnn_features_diverse": bool(
            fcd["cnn_features"]["dim_std_mean"] >= DIM_STD_DIVERSE
        ),
        "latent_pi_diverse": bool(
            fcd["latent_pi"]["dim_std_mean"] >= DIM_STD_DIVERSE
        ),
        "logits_near_uniform_across_rows": bool(
            probe["mean_logit_range_across_rows"]
            < LOGIT_NEAR_UNIFORM_RANGE_ACROSS_ROWS
        ),
        "margins_tiny": bool(
            probe["mean_margin_top1_top2"] < MARGIN_TINY
        ),
    }


def _classify_within(sig: dict[str, Any]) -> tuple[str, str]:
    """Map a within-model signature to K4-A / K4-B / K4-C or unclassified."""
    if sig["cnn_features_near_uniform"]:
        return (
            "K4-A",
            (
                f"cnn_features dim_std_mean="
                f"{sig['cnn_features_dim_std_mean']:.6g} < "
                f"{DIM_STD_NEAR_UNIFORM}; feature extractor near-uniform on "
                "panel; logits cannot diversify regardless of action-head."
            ),
        )
    if sig["latent_pi_diverse"] and sig["logits_near_uniform_across_rows"]:
        return (
            "K4-B",
            (
                f"latent_pi dim_std_mean="
                f"{sig['latent_pi_dim_std_mean']:.6g} (diverse, >= "
                f"{DIM_STD_DIVERSE}), but logit range across rows="
                f"{sig['mean_logit_range_across_rows']:.6g} < "
                f"{LOGIT_NEAR_UNIFORM_RANGE_ACROSS_ROWS}, mean margin="
                f"{sig['mean_margin_top1_top2']:.6g} "
                f"(tiny={sig['margins_tiny']}); action-head produces "
                "near-uniform logits across diverse latent_pi inputs."
            ),
        )
    if sig["latent_pi_diverse"] and not sig["logits_near_uniform_across_rows"]:
        return (
            "K4-C",
            (
                f"latent_pi diverse, logit range across rows="
                f"{sig['mean_logit_range_across_rows']:.6g}, mean margin="
                f"{sig['mean_margin_top1_top2']:.6g}; logits differ across "
                "panel rows yet a single argmax dominates "
                f"({sig['panel_top_argmax_fraction']:.3f} fraction, "
                f"num_det_actions={sig['panel_num_det_actions']}); "
                "action-head decision boundary is pinned to one action."
            ),
        )
    return (
        "K4-unclassified-within",
        (
            f"signature did not match A/B/C: cnn_uniform="
            f"{sig['cnn_features_near_uniform']} latent_pi_diverse="
            f"{sig['latent_pi_diverse']} logits_near_uniform="
            f"{sig['logits_near_uniform_across_rows']}"
        ),
    )

def classify_k4(
    probes: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Map probe + comparison data to K4-A through K4-G.

    Verdict shape: "{cross_model}+{within_model}" where cross_model is
    K4-D, K4-E, K4-F, or K4-G, and within_model is K4-A, K4-B, or K4-C
    based on the K3.5c 10000 model (the latest training state). If
    cross_model is K4-E (panel non-representative), within_model is
    still reported on K3.5c 10000 but flagged as advisory.
    """
    fresh_sig = _within_model_signature(probes["fresh_seed0_init"])
    p2048_sig = _within_model_signature(probes["k3_5c_2048"])
    p10000_sig = _within_model_signature(probes["k3_5c_10000"])

    pair_2048_10000 = comparison["pairs"]["k3_5c_2048_vs_k3_5c_10000"]
    pair_fresh_2048 = comparison["pairs"]["fresh_seed0_init_vs_k3_5c_2048"]
    pair_fresh_10000 = (
        comparison["pairs"]["fresh_seed0_init_vs_k3_5c_10000"]
    )

    flags = {
        "argmax_2048_eq_10000": bool(
            pair_2048_10000["argmax_match_fraction"] >= ARGMAX_MATCH_TIGHT
        ),
        "argmax_fresh_eq_2048": bool(
            pair_fresh_2048["argmax_match_fraction"] >= ARGMAX_MATCH_TIGHT
        ),
        "argmax_fresh_eq_10000": bool(
            pair_fresh_10000["argmax_match_fraction"] >= ARGMAX_MATCH_TIGHT
        ),
        "logits_2048_vs_10000_linf_max": pair_2048_10000["logit_linf_max"],
        "logits_2048_vs_10000_material_drift": bool(
            pair_2048_10000["logit_linf_max"]
            >= INTER_MODEL_LOGIT_LINF_MATERIAL
        ),
        "logits_2048_vs_10000_negligible": bool(
            pair_2048_10000["logit_linf_max"]
            < INTER_MODEL_LOGIT_LINF_NEGLIGIBLE
        ),
    }

    # Cross-model classification.
    cross_verdict: str
    cross_rationale: str
    if not flags["argmax_2048_eq_10000"]:
        cross_verdict = "K4-E"
        cross_rationale = (
            f"K3.5c 2048 vs 10000 panel argmax match fraction="
            f"{pair_2048_10000['argmax_match_fraction']:.3f} < "
            f"{ARGMAX_MATCH_TIGHT} while external eval is bit-identical; "
            "panel is not representative of the eval rollout."
        )
    elif (
        flags["argmax_fresh_eq_2048"] and flags["argmax_fresh_eq_10000"]
    ):
        cross_verdict = "K4-F"
        cross_rationale = (
            f"fresh_seed0_init panel argmax matches both K3.5c 2048 "
            f"(fraction={pair_fresh_2048['argmax_match_fraction']:.3f}) "
            f"and K3.5c 10000 "
            f"(fraction={pair_fresh_10000['argmax_match_fraction']:.3f}); "
            "deterministic-argmax surface is fixed at init and persists "
            "through all K3.5c training."
        )
    elif flags["logits_2048_vs_10000_material_drift"]:
        cross_verdict = "K4-D"
        cross_rationale = (
            f"K3.5c 2048 and 10000 panel argmax match "
            f"(fraction={pair_2048_10000['argmax_match_fraction']:.3f}) "
            f"but raw logits drift materially (linf_max="
            f"{pair_2048_10000['logit_linf_max']:.6g} >= "
            f"{INTER_MODEL_LOGIT_LINF_MATERIAL}); training continues to "
            "move logits but never crosses a deterministic decision "
            "boundary on the panel; eval-invisible logit drift."
        )
    else:
        cross_verdict = "K4-G"
        cross_rationale = (
            f"fresh_seed0_init differs from K3.5c 2048 (argmax_match="
            f"{pair_fresh_2048['argmax_match_fraction']:.3f}) and "
            f"K3.5c 10000 (argmax_match="
            f"{pair_fresh_10000['argmax_match_fraction']:.3f}), but "
            f"2048 and 10000 match each other (argmax_match="
            f"{pair_2048_10000['argmax_match_fraction']:.3f}, logit "
            f"linf_max={pair_2048_10000['logit_linf_max']:.6g}); early "
            "training lands on a stable deterministic surface that is "
            "frozen by update 8 and unchanged through update 156."
        )

    within_verdict, within_rationale = _classify_within(p10000_sig)

    verdict = f"{cross_verdict}+{within_verdict}"
    return {
        "verdict": verdict,
        "cross_model_verdict": cross_verdict,
        "cross_model_rationale": cross_rationale,
        "within_model_verdict": within_verdict,
        "within_model_rationale": within_rationale,
        "flags": flags,
        "signatures": {
            "fresh_seed0_init": fresh_sig,
            "k3_5c_2048": p2048_sig,
            "k3_5c_10000": p10000_sig,
        },
        "thresholds": {
            "DIM_STD_NEAR_UNIFORM": DIM_STD_NEAR_UNIFORM,
            "DIM_STD_DIVERSE": DIM_STD_DIVERSE,
            "LOGIT_NEAR_UNIFORM_RANGE_ACROSS_ROWS": (
                LOGIT_NEAR_UNIFORM_RANGE_ACROSS_ROWS
            ),
            "MARGIN_TINY": MARGIN_TINY,
            "INTER_MODEL_LOGIT_LINF_NEGLIGIBLE": (
                INTER_MODEL_LOGIT_LINF_NEGLIGIBLE
            ),
            "INTER_MODEL_LOGIT_LINF_MATERIAL": (
                INTER_MODEL_LOGIT_LINF_MATERIAL
            ),
            "ARGMAX_MATCH_TIGHT": ARGMAX_MATCH_TIGHT,
        },
    }

class _PanelSpaceEnv(gym.Env):
    """Minimal stub env used to construct a fresh SB3 PPO model.

    SB3 PPO.__init__ requires a VecEnv to read observation_space and
    action_space. The policy weights are initialized inside
    _setup_model -> set_random_seed(seed) -> policy_class(obs_space,
    act_space, ...), so once the seed is fixed and the spaces match
    GodotSignalDodgeEnv's H5 pixel contract, fresh init weights are
    identical to what the production trainer produces at update 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_n: int,
        obs_dtype: np.dtype,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(int(action_n))
        # Match GodotSignalDodgeEnv H5 pixel contract: uint8, range [0, 255].
        if np.issubdtype(obs_dtype, np.unsignedinteger):
            low, high = 0, 255
        else:
            low, high = 0.0, 1.0
        self.observation_space = spaces.Box(
            low=low, high=high,
            shape=tuple(int(x) for x in observation_shape),
            dtype=obs_dtype,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return (
            self.observation_space.sample(),
            0.0,
            False,
            False,
            {},
        )


def build_panel_space_vec_env(
    observation_shape: tuple[int, ...],
    action_n: int,
    obs_dtype: np.dtype,
) -> DummyVecEnv:
    """Wrap _PanelSpaceEnv in DummyVecEnv for PPO construction."""
    return DummyVecEnv(
        [lambda: _PanelSpaceEnv(observation_shape, action_n, obs_dtype)]
    )

def write_json_artifact(
    out_dir: Path,
    label: str,
    config_path: str,
    seed: int,
    panel_metadata: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    classification: dict[str, Any],
) -> Path:
    """Write the full K4.0 probe payload as JSON."""
    payload = {
        "_header": True,
        "tool": "tools/k4_panel_logit_probe.py",
        "phase": "H5-K-K4.0",
        "config_path": str(config_path),
        "seed": int(seed),
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panel_metadata": panel_metadata,
        "models": probes,
        "comparison": comparison,
        "classification": classification,
    }
    json_path = out_dir / f"{label}.json"
    with json_path.open("w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2)
    return json_path


def write_csv_artifact(
    out_dir: Path,
    label: str,
    probes: dict[str, dict[str, Any]],
) -> Path:
    """Write per-row panel data as CSV with model_name first column."""
    csv_path = out_dir / f"{label}_panel_rows.csv"
    fieldnames = [
        "model_name", "model_sha256",
        "row_idx", "panel_seed", "prefix_name", "prefix_length",
        "raw_logit_left", "raw_logit_stay", "raw_logit_right",
        "prob_left", "prob_stay", "prob_right",
        "det_argmax_action", "det_argmax_idx", "margin_top1_top2",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, probe in probes.items():
            sha = probe.get("model_sha256") or ""
            for row in probe["panel_rows"]:
                rec = {"model_name": name, "model_sha256": sha}
                for k in fieldnames:
                    if k in ("model_name", "model_sha256"):
                        continue
                    rec[k] = row[k]
                writer.writerow(rec)
    return csv_path

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="k4_panel_logit_probe",
        description=(
            "Phase K K4.0 panel-logit mechanism diagnostic. Probes three "
            "policy states (fresh seed-0 init, K3.5c 2048, K3.5c 10000) "
            "on the same fixed observation-conditioning panel and emits "
            "row-level logits plus per-model summaries."
        ),
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML env+algo config (use H5 entropy config).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the fresh PPO model and for panel-env construction.",
    )
    p.add_argument(
        "--out-dir", default="runs/phase_k",
        help="Output directory under runs/.",
    )
    p.add_argument(
        "--label", default="k4_panel_logit_probe",
        help="Output filename stem.",
    )
    p.add_argument(
        "--k3-5c-2048", required=True,
        help="Path to K3.5c 2048 checkpoint model.zip.",
    )
    p.add_argument(
        "--k3-5c-10000", required=True,
        help="Path to K3.5c 10000 checkpoint model.zip.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    godot_extra = resolve_godot_kwargs(cfg)

    # Step 1: build the fixed observation-conditioning panel via the same
    # machinery the training probe uses. Panel env is opened, panel obs
    # captured under fixed seeds and scripted prefixes, then panel env is
    # closed inside the helper.
    print(
        "[k4_panel_logit_probe] building fixed observation panel "
        "(Godot env, closes before continuing)...",
        flush=True,
    )
    fixed_panel_obs, fixed_panel_metadata = build_fixed_observation_panel(
        cfg=cfg,
        godot_extra=godot_extra,
        train_seed=int(args.seed),
        out_dir=out_dir,
        label=args.label,
    )
    print(
        f"[k4_panel_logit_probe] panel captured "
        f"{fixed_panel_obs.shape[0]} obs (shape={list(fixed_panel_obs.shape)} "
        f"dtype={fixed_panel_obs.dtype})",
        flush=True,
    )

    obs_shape = tuple(int(x) for x in fixed_panel_obs.shape[1:])
    obs_dtype = fixed_panel_obs.dtype
    algo_cfg = cfg["algo"]
    hyperparams = dict(algo_cfg.get("hyperparams") or {})
    policy_class = algo_cfg.get("policy", "CnnPolicy")
    device = "cpu"

    # Action-space cardinality is fixed to GodotSignalDodgeEnv's discrete
    # {left, stay, right} contract, matching ACTION_NAMES.
    action_n = 3

    # Step 2: construct fresh seed-0 PPO using the panel-space stub env.
    # SB3 seeds inside _setup_model so fresh weights match what a
    # production train at seed 0 produces at update 0.
    print(
        "[k4_panel_logit_probe] constructing fresh CnnPolicy at seed 0 "
        "via panel-space stub env (no Godot, no training)...",
        flush=True,
    )
    stub_vec = build_panel_space_vec_env(obs_shape, action_n, obs_dtype)
    fresh_model = PPO(
        policy=policy_class,
        env=stub_vec,
        seed=int(args.seed),
        device=device,
        **hyperparams,
    )

    panel_tensor = th.as_tensor(fixed_panel_obs).to(fresh_model.device)

    print("[k4_panel_logit_probe] probing fresh_seed0_init...", flush=True)
    probes: dict[str, dict[str, Any]] = {}
    probes["fresh_seed0_init"] = probe_one_policy(
        fresh_model.policy,
        panel_tensor,
        fixed_panel_metadata,
        name="fresh_seed0_init",
        model_zip_path=None,
    )
    try:
        stub_vec.close()
    except Exception:
        pass

    # Step 3: load K3.5c checkpoints and probe each on the same panel.
    checkpoint_paths = {
        "k3_5c_2048": Path(args.k3_5c_2048),
        "k3_5c_10000": Path(args.k3_5c_10000),
    }
    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found for {name}: {path}"
            )
        print(
            f"[k4_panel_logit_probe] loading {name} from {path}...",
            flush=True,
        )
        loaded = PPO.load(str(path), device=device)
        panel_tensor_dev = panel_tensor.to(loaded.device)
        print(f"[k4_panel_logit_probe] probing {name}...", flush=True)
        probes[name] = probe_one_policy(
            loaded.policy,
            panel_tensor_dev,
            fixed_panel_metadata,
            name=name,
            model_zip_path=path,
        )

    # Step 4: cross-model comparisons. Order pairs deterministically:
    # fresh vs 2048, fresh vs 10000, 2048 vs 10000.
    pair_keys: list[tuple[str, str]] = [
        ("fresh_seed0_init", "k3_5c_2048"),
        ("fresh_seed0_init", "k3_5c_10000"),
        ("k3_5c_2048", "k3_5c_10000"),
    ]
    print("[k4_panel_logit_probe] computing pairwise comparisons...", flush=True)
    comparison = compare_models(probes, pair_keys)

    # Step 5: K4 classification (cross-model + within-model).
    classification = classify_k4(probes, comparison)

    # Step 6: write artifacts.
    json_path = write_json_artifact(
        out_dir=out_dir,
        label=args.label,
        config_path=args.config,
        seed=int(args.seed),
        panel_metadata=fixed_panel_metadata,
        probes=probes,
        comparison=comparison,
        classification=classification,
    )
    csv_path = write_csv_artifact(
        out_dir=out_dir,
        label=args.label,
        probes=probes,
    )

    print(
        f"[k4_panel_logit_probe] DONE verdict={classification['verdict']}",
        flush=True,
    )
    print(
        f"[k4_panel_logit_probe] cross_model: "
        f"{classification['cross_model_verdict']}: "
        f"{classification['cross_model_rationale']}",
        flush=True,
    )
    print(
        f"[k4_panel_logit_probe] within_model: "
        f"{classification['within_model_verdict']}: "
        f"{classification['within_model_rationale']}",
        flush=True,
    )
    print(f"[k4_panel_logit_probe] wrote {json_path}", flush=True)
    print(f"[k4_panel_logit_probe] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
