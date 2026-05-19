"""Phase K K4.1 eval-observation panel-logit mechanism diagnostic.

K4.0 falsified easy explanations on a narrow scripted-prefix panel but
left the eval-distribution layer untested: the 32-row scripted panel
collapsed to only 5-8 effectively-distinct logit signatures, so the
"action head pins 'left' 100%" finding could not explain why two
K3.5c checkpoints with distinct SHA-256 hashes (E14D1A12... and
5664A12E...) produce bit-identical per-seed eval lengths 243-1800
across 10 seeds.

K4.1 swaps the scripted-prefix panel for actual K3.5c trained-only
eval-rollout observations on seeds 1000-1009, then probes both
checkpoints on the captured obs sequences.

Methodology:

Pass 1 (actor = K3.5c 10000): open one Godot env, loop seeds 1000-1009,
  per seed: env.seed + reset, rollout with deterministic argmax from
  K3.5c 10000. Per step capture: obs, obs_sha256, action taken,
  K3.5c 10000 logits/probs/argmax/margin (self-probe),
  K3.5c 2048 logits/probs/argmax/margin on the SAME obs (cross-probe).

Pass 2 (actor = K3.5c 2048): open one Godot env, loop seeds 1000-1009
  same way but use K3.5c 2048's argmax as the actor. Per step capture:
  obs_sha256, action taken. Minimal record for trajectory parity check.

Cross-check (K4.1-E reproducibility):
- per-seed episode length match
- per-seed termination flag match
- per-(seed, step) obs_sha256 match
- per-(seed, step) action match

Compare on Pass 1 captures (the load-bearing K4.1 question):
- per-(seed, step) argmax match between K3.5c 2048 and K3.5c 10000 on
  the eval-obs distribution
- first step of divergence, if any
- logit drift stats (mean/max L1, L_inf per row)
- margin distribution per checkpoint

Classification:
- K4.1-A full eval pin: 100% argmax match on every reached eval obs,
  K3.5c 2048 cross-probe matches Pass 2 actor actions, lengths and
  obs sequences identical. Mechanistic explanation for bit-identical
  eval: every eval obs maps to the same action on both checkpoints,
  so trajectories coincide step by step.
- K4.1-B boundary-equivalent drift: like K4.1-A but raw logits differ
  materially (L_inf max above threshold) and margins differ. Same
  practical explanation; intervention to change eval requires crossing
  decision boundaries.
- K4.1-C panel failure only: cross-probe argmax differs on some eval
  obs, AND Pass 2 actor actions also differ at those steps, yet
  trajectories still coincide. Would imply env or step-timing
  coincidence (does not match K3.5c bit-identical claim).
- K4.1-D hidden divergence: cross-probe argmax differs but Pass 2
  actor actions match Pass 1 actor actions exactly (eval-capture
  alignment defect or obs aliasing). Requires defect inspection.
- K4.1-E reproducibility failure: Pass 1 and Pass 2 do not reproduce
  the K3.5c eval lengths or terminations. Stop and fix eval-capture
  parity.

Outputs:
- runs/phase_k/k4_1_eval_obs_logit_probe.json
- runs/phase_k/k4_1_eval_obs_logit_probe_rows.csv

No training. No env code change. No config change.
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

import numpy as np
import torch as th

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_TOOLS = _REPO_ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from stable_baselines3 import PPO  # noqa: E402

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.factories import make_env  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402

ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}

# Classification thresholds (diagnostic, not tuned).
INTER_MODEL_LOGIT_LINF_MATERIAL = 0.1
ARGMAX_MATCH_TIGHT = 0.999


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _sha256_of_obs(obs: np.ndarray) -> str:
    """Hash a single observation for trajectory parity checks."""
    arr = np.ascontiguousarray(obs)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _forward_pass(policy: Any, obs_tensor: th.Tensor) -> dict[str, np.ndarray]:
    """Forward pass capturing raw logits, probs, argmax, top1-top2 margin.

    Mirrors the actor path used by snapshot_policy_state in
    tools/h5_training_entropy_probe.py: extract_features -> mlp_extractor
    -> action_net. No grad, no env interaction.
    """
    with th.no_grad():
        features = policy.extract_features(obs_tensor)
        latent_pi, _ = policy.mlp_extractor(features)
        raw_logits = policy.action_net(latent_pi)
        probs = th.softmax(raw_logits, dim=-1)
        argmax = raw_logits.argmax(dim=-1)
        top2 = th.topk(raw_logits, k=2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
    return {
        "logits": raw_logits.detach().cpu().numpy().astype(np.float64),
        "probs": probs.detach().cpu().numpy().astype(np.float64),
        "argmax": argmax.detach().cpu().numpy().astype(np.int64),
        "margins": margins.detach().cpu().numpy().astype(np.float64),
    }

def build_eval_env(cfg: dict[str, Any], seed: int, run_dir: Path) -> Any:
    """Construct a Godot eval env via the standard factories path."""
    env_id = cfg["env"]["id"]
    godot_extra = resolve_godot_kwargs(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if godot_extra:
        return make_env(
            env_id, n_envs=1, seed=int(seed), mode="eval",
            run_dir=str(run_dir), **godot_extra,
        )
    return make_env(env_id, n_envs=1, seed=int(seed), mode="eval")


def rollout_with_full_capture(
    env: Any,
    actor_model: Any,
    cross_model: Any,
    actor_label: str,
    cross_label: str,
    eval_seed: int,
    episode_id: int,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One episode: actor drives the env, both checkpoints get a forward pass.

    Returns (episode_summary, per_step_rows). actor_label is the
    checkpoint whose argmax is taken; cross_label is the other
    checkpoint, probed on the same obs but its argmax is NOT used to
    step the env.
    """
    try:
        env.seed(int(eval_seed))
    except (AttributeError, TypeError):
        pass
    obs = env.reset()
    rows: list[dict[str, Any]] = []
    ep_len = 0
    ep_reward = 0.0
    final_info: dict[str, Any] = {}
    done_flag = False
    obs_sha_first = None
    obs_sha_last = None
    cross_argmax_diffs = 0
    first_diff_step: int | None = None
    t0 = time.time()
    while ep_len < max_steps:
        obs_arr = np.asarray(obs)
        # obs_arr shape (1, C, H, W); single-batch row.
        obs_for_hash = obs_arr[0] if obs_arr.ndim == 4 else obs_arr
        obs_sha = _sha256_of_obs(obs_for_hash)
        if ep_len == 0:
            obs_sha_first = obs_sha
        obs_sha_last = obs_sha
        # Build single tensor for both forward passes; same device as
        # actor model. Both checkpoints live on cpu in this tool.
        obs_tensor = th.as_tensor(obs_arr).to(actor_model.device)
        actor_fp = _forward_pass(actor_model.policy, obs_tensor)
        cross_fp = _forward_pass(cross_model.policy, obs_tensor)

        actor_action = int(actor_fp["argmax"][0])
        cross_action = int(cross_fp["argmax"][0])
        if actor_action != cross_action:
            cross_argmax_diffs += 1
            if first_diff_step is None:
                first_diff_step = ep_len

        # Per-step row. Logits stored as native floats to keep CSV small.
        rows.append({
            "eval_seed": int(eval_seed),
            "episode_id": int(episode_id),
            "step": int(ep_len),
            "obs_sha256": obs_sha,
            "actor_label": actor_label,
            "actor_action": ACTION_NAMES[actor_action],
            "actor_action_idx": int(actor_action),
            f"argmax_{actor_label}": ACTION_NAMES[actor_action],
            f"argmax_{cross_label}": ACTION_NAMES[cross_action],
            "argmax_match": bool(actor_action == cross_action),
            f"logit_left_{actor_label}": float(actor_fp["logits"][0, 0]),
            f"logit_stay_{actor_label}": float(actor_fp["logits"][0, 1]),
            f"logit_right_{actor_label}": float(actor_fp["logits"][0, 2]),
            f"logit_left_{cross_label}": float(cross_fp["logits"][0, 0]),
            f"logit_stay_{cross_label}": float(cross_fp["logits"][0, 1]),
            f"logit_right_{cross_label}": float(cross_fp["logits"][0, 2]),
            f"prob_left_{actor_label}": float(actor_fp["probs"][0, 0]),
            f"prob_stay_{actor_label}": float(actor_fp["probs"][0, 1]),
            f"prob_right_{actor_label}": float(actor_fp["probs"][0, 2]),
            f"prob_left_{cross_label}": float(cross_fp["probs"][0, 0]),
            f"prob_stay_{cross_label}": float(cross_fp["probs"][0, 1]),
            f"prob_right_{cross_label}": float(cross_fp["probs"][0, 2]),
            f"margin_{actor_label}": float(actor_fp["margins"][0]),
            f"margin_{cross_label}": float(cross_fp["margins"][0]),
        })

        obs, reward, dones, infos = env.step(np.asarray([actor_action]))
        ep_reward += float(np.asarray(reward).sum())
        ep_len += 1
        done_flag = bool(np.asarray(dones).any())
        if done_flag:
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                final_info = dict(infos[0])
            break

    elapsed = time.time() - t0
    timeout = bool(final_info.get("TimeLimit.truncated", False)) if done_flag else (ep_len >= max_steps)
    collision = bool(done_flag and not timeout)
    return ({
        "eval_seed": int(eval_seed),
        "episode_id": int(episode_id),
        "actor_label": actor_label,
        "cross_label": cross_label,
        "episode_length": int(ep_len),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "total_reward": float(ep_reward),
        "elapsed_seconds": float(elapsed),
        "obs_sha_first": obs_sha_first,
        "obs_sha_last": obs_sha_last,
        "cross_argmax_diffs": int(cross_argmax_diffs),
        "first_diff_step": first_diff_step,
        "action_trace": [r["actor_action_idx"] for r in rows],
        "obs_sha_trace": [r["obs_sha256"] for r in rows],
        "cross_argmax_trace": [
            int({"left": 0, "stay": 1, "right": 2}[r[f"argmax_{cross_label}"]])
            for r in rows
        ],
    }, rows)

def rollout_minimal(
    env: Any,
    actor_model: Any,
    eval_seed: int,
    episode_id: int,
    max_steps: int,
    actor_label: str,
) -> dict[str, Any]:
    """One episode: only capture obs hashes and actions.

    Used in Pass 2 for trajectory parity verification against Pass 1.
    """
    try:
        env.seed(int(eval_seed))
    except (AttributeError, TypeError):
        pass
    obs = env.reset()
    obs_sha_trace: list[str] = []
    action_trace: list[int] = []
    ep_reward = 0.0
    ep_len = 0
    final_info: dict[str, Any] = {}
    done_flag = False
    t0 = time.time()
    while ep_len < max_steps:
        obs_arr = np.asarray(obs)
        obs_for_hash = obs_arr[0] if obs_arr.ndim == 4 else obs_arr
        obs_sha_trace.append(_sha256_of_obs(obs_for_hash))
        obs_tensor = th.as_tensor(obs_arr).to(actor_model.device)
        with th.no_grad():
            features = actor_model.policy.extract_features(obs_tensor)
            latent_pi, _ = actor_model.policy.mlp_extractor(features)
            raw_logits = actor_model.policy.action_net(latent_pi)
            action = int(raw_logits.argmax(dim=-1).item())
        action_trace.append(action)
        obs, reward, dones, infos = env.step(np.asarray([action]))
        ep_reward += float(np.asarray(reward).sum())
        ep_len += 1
        done_flag = bool(np.asarray(dones).any())
        if done_flag:
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                final_info = dict(infos[0])
            break
    elapsed = time.time() - t0
    timeout = bool(final_info.get("TimeLimit.truncated", False)) if done_flag else (ep_len >= max_steps)
    collision = bool(done_flag and not timeout)
    return {
        "eval_seed": int(eval_seed),
        "episode_id": int(episode_id),
        "actor_label": actor_label,
        "episode_length": int(ep_len),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "total_reward": float(ep_reward),
        "elapsed_seconds": float(elapsed),
        "obs_sha_trace": obs_sha_trace,
        "action_trace": action_trace,
    }


def compare_trajectories(
    pass1_summaries: list[dict[str, Any]],
    pass2_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-seed parity check between Pass 1 (actor=10000) and Pass 2 (actor=2048)."""
    by_seed1 = {s["eval_seed"]: s for s in pass1_summaries}
    by_seed2 = {s["eval_seed"]: s for s in pass2_summaries}
    seeds = sorted(set(by_seed1) | set(by_seed2))
    per_seed_records: list[dict[str, Any]] = []
    length_matches = 0
    termination_matches = 0
    full_obs_matches = 0
    full_action_matches = 0
    for s in seeds:
        s1 = by_seed1.get(s); s2 = by_seed2.get(s)
        if s1 is None or s2 is None:
            per_seed_records.append({
                "eval_seed": int(s),
                "present_in_pass1": s1 is not None,
                "present_in_pass2": s2 is not None,
            })
            continue
        len_eq = bool(s1["episode_length"] == s2["episode_length"])
        term_eq = bool(s1["collision"] == s2["collision"] and s1["timeout"] == s2["timeout"])
        # Compare obs and actions over the overlapping prefix.
        n_overlap = min(len(s1["obs_sha_trace"]), len(s2["obs_sha_trace"]))
        obs_match_count = sum(
            1 for i in range(n_overlap)
            if s1["obs_sha_trace"][i] == s2["obs_sha_trace"][i]
        )
        act_match_count = sum(
            1 for i in range(n_overlap)
            if s1["action_trace"][i] == s2["action_trace"][i]
        )
        first_obs_diff = next(
            (i for i in range(n_overlap)
             if s1["obs_sha_trace"][i] != s2["obs_sha_trace"][i]),
            None,
        )
        first_act_diff = next(
            (i for i in range(n_overlap)
             if s1["action_trace"][i] != s2["action_trace"][i]),
            None,
        )
        obs_full = bool(len_eq and obs_match_count == n_overlap == s1["episode_length"])
        act_full = bool(len_eq and act_match_count == n_overlap == s1["episode_length"])
        if len_eq: length_matches += 1
        if term_eq: termination_matches += 1
        if obs_full: full_obs_matches += 1
        if act_full: full_action_matches += 1
        per_seed_records.append({
            "eval_seed": int(s),
            "pass1_length": int(s1["episode_length"]),
            "pass2_length": int(s2["episode_length"]),
            "length_match": len_eq,
            "pass1_collision": bool(s1["collision"]),
            "pass2_collision": bool(s2["collision"]),
            "pass1_timeout": bool(s1["timeout"]),
            "pass2_timeout": bool(s2["timeout"]),
            "termination_match": term_eq,
            "overlap_steps": int(n_overlap),
            "obs_match_count": int(obs_match_count),
            "action_match_count": int(act_match_count),
            "first_obs_diff_step": first_obs_diff,
            "first_action_diff_step": first_act_diff,
            "full_obs_match": obs_full,
            "full_action_match": act_full,
        })
    return {
        "n_seeds": int(len(seeds)),
        "length_matches": int(length_matches),
        "termination_matches": int(termination_matches),
        "full_obs_matches": int(full_obs_matches),
        "full_action_matches": int(full_action_matches),
        "per_seed": per_seed_records,
    }

def aggregate_logit_stats(
    pass1_rows: list[dict[str, Any]],
    actor_label: str,
    cross_label: str,
) -> dict[str, Any]:
    """Aggregate per-step logit / argmax comparison over Pass 1 captures."""
    if not pass1_rows:
        return {"n_rows": 0}
    n = len(pass1_rows)
    argmax_match = sum(1 for r in pass1_rows if r["argmax_match"])
    linf_per_row: list[float] = []
    l1_per_row: list[float] = []
    margins_actor: list[float] = []
    margins_cross: list[float] = []
    per_seed_first_diff: dict[int, int | None] = {}
    seeds_seen: set[int] = set()
    for r in pass1_rows:
        seeds_seen.add(int(r["eval_seed"]))
        la = np.array([
            r[f"logit_left_{actor_label}"],
            r[f"logit_stay_{actor_label}"],
            r[f"logit_right_{actor_label}"],
        ])
        lc = np.array([
            r[f"logit_left_{cross_label}"],
            r[f"logit_stay_{cross_label}"],
            r[f"logit_right_{cross_label}"],
        ])
        diff = la - lc
        linf_per_row.append(float(np.max(np.abs(diff))))
        l1_per_row.append(float(np.sum(np.abs(diff))))
        margins_actor.append(float(r[f"margin_{actor_label}"]))
        margins_cross.append(float(r[f"margin_{cross_label}"]))
    for s in sorted(seeds_seen):
        first_diff = next(
            (int(r["step"]) for r in pass1_rows
             if int(r["eval_seed"]) == s and not r["argmax_match"]),
            None,
        )
        per_seed_first_diff[int(s)] = first_diff
    return {
        "n_rows": int(n),
        "argmax_match_count": int(argmax_match),
        "argmax_match_fraction": float(argmax_match) / n if n else 0.0,
        "logit_linf_mean": float(np.mean(linf_per_row)),
        "logit_linf_max": float(np.max(linf_per_row)),
        "logit_l1_mean": float(np.mean(l1_per_row)),
        "logit_l1_max": float(np.max(l1_per_row)),
        f"margin_{actor_label}_mean": float(np.mean(margins_actor)),
        f"margin_{actor_label}_min": float(np.min(margins_actor)),
        f"margin_{actor_label}_max": float(np.max(margins_actor)),
        f"margin_{cross_label}_mean": float(np.mean(margins_cross)),
        f"margin_{cross_label}_min": float(np.min(margins_cross)),
        f"margin_{cross_label}_max": float(np.max(margins_cross)),
        "per_seed_first_argmax_diff_step": per_seed_first_diff,
    }


def classify_k4_1(
    logit_agg: dict[str, Any],
    trajectory_cmp: dict[str, Any],
    actor_label: str,
    cross_label: str,
) -> dict[str, Any]:
    """Map K4.1 evidence onto K4.1-A through K4.1-E."""
    reproducibility_ok = bool(
        trajectory_cmp["length_matches"] == trajectory_cmp["n_seeds"]
        and trajectory_cmp["termination_matches"] == trajectory_cmp["n_seeds"]
        and trajectory_cmp["full_obs_matches"] == trajectory_cmp["n_seeds"]
        and trajectory_cmp["full_action_matches"] == trajectory_cmp["n_seeds"]
    )
    argmax_match_frac = logit_agg.get("argmax_match_fraction", 0.0)
    cross_argmax_full = bool(argmax_match_frac >= ARGMAX_MATCH_TIGHT)
    logits_material = bool(
        logit_agg.get("logit_linf_max", 0.0) >= INTER_MODEL_LOGIT_LINF_MATERIAL
    )

    if not reproducibility_ok:
        verdict = "K4.1-E"
        rationale = (
            f"reproducibility failed: length_matches="
            f"{trajectory_cmp['length_matches']}/{trajectory_cmp['n_seeds']} "
            f"termination_matches={trajectory_cmp['termination_matches']} "
            f"full_obs_matches={trajectory_cmp['full_obs_matches']} "
            f"full_action_matches={trajectory_cmp['full_action_matches']}; "
            "Pass 1 and Pass 2 do not reproduce a bit-identical eval, "
            "so the K3.5c claim cannot be probed by this capture path."
        )
    elif cross_argmax_full and not logits_material:
        verdict = "K4.1-A"
        rationale = (
            f"every eval obs maps to the same argmax on both checkpoints "
            f"(match fraction={argmax_match_frac:.6f}) and raw logits do "
            f"not differ materially (linf_max="
            f"{logit_agg.get('logit_linf_max', 0.0):.6g} < "
            f"{INTER_MODEL_LOGIT_LINF_MATERIAL}); trajectories coincide "
            "step by step on the eval distribution."
        )
    elif cross_argmax_full and logits_material:
        verdict = "K4.1-B"
        rationale = (
            f"every eval obs maps to the same argmax on both checkpoints "
            f"(match fraction={argmax_match_frac:.6f}), but raw logits "
            f"drift materially (linf_max="
            f"{logit_agg.get('logit_linf_max', 0.0):.6g} >= "
            f"{INTER_MODEL_LOGIT_LINF_MATERIAL}); decision boundaries are "
            "intact across the eval distribution; intervention to change "
            "eval requires crossing a boundary."
        )
    else:
        # cross_argmax_full is false: cross-probe argmax differs on some
        # eval obs. Distinguish K4.1-C from K4.1-D using Pass 2 actor
        # behavior. If Pass 2 actor actions match Pass 1 actor actions
        # exactly, then K3.5c 2048 in its own rollout DID select the
        # same action as K3.5c 10000 at every step, even though its
        # forward pass on the captured obs from Pass 1 says otherwise
        # at the diverging step. That implies an alignment defect:
        # K4.1-D.
        full_action_match = bool(
            trajectory_cmp["full_action_matches"] == trajectory_cmp["n_seeds"]
        )
        if full_action_match:
            verdict = "K4.1-D"
            rationale = (
                f"Pass 2 actor (K3.5c 2048) action trace matches Pass 1 "
                f"actor (K3.5c 10000) at every step on every seed, but "
                f"cross-probe argmax differs (match fraction="
                f"{argmax_match_frac:.6f}); eval-capture obs alignment "
                "is suspect, or the captured obs do not correspond to "
                "the policy's forward input at decision time."
            )
        else:
            verdict = "K4.1-C"
            rationale = (
                f"cross-probe argmax differs (match fraction="
                f"{argmax_match_frac:.6f}) AND Pass 2 actor actions "
                f"differ from Pass 1 actor actions at some step, yet "
                f"per-seed lengths and terminations match. Either the "
                "K3.5c bit-identical claim is panel-narrow only, or "
                "trajectories coincide through env-level coincidence."
            )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "flags": {
            "reproducibility_ok": reproducibility_ok,
            "cross_argmax_full_match": cross_argmax_full,
            "logits_material_drift": logits_material,
        },
        "thresholds": {
            "INTER_MODEL_LOGIT_LINF_MATERIAL": INTER_MODEL_LOGIT_LINF_MATERIAL,
            "ARGMAX_MATCH_TIGHT": ARGMAX_MATCH_TIGHT,
        },
    }

def write_outputs(
    out_dir: Path,
    label: str,
    config_path: str,
    seeds: list[int],
    max_steps: int,
    actor_label: str,
    cross_label: str,
    model_paths: dict[str, str],
    model_sha256s: dict[str, str],
    pass1_summaries: list[dict[str, Any]],
    pass2_summaries: list[dict[str, Any]],
    pass1_rows: list[dict[str, Any]],
    trajectory_cmp: dict[str, Any],
    logit_agg: dict[str, Any],
    classification: dict[str, Any],
) -> tuple[Path, Path]:
    """Write JSON + CSV artifacts."""
    json_path = out_dir / f"{label}.json"
    csv_path = out_dir / f"{label}_rows.csv"

    # JSON: full payload sans per-step rows (CSV is the row store).
    # Pass summaries strip the high-volume traces but keep length / outcome
    # plus a count of trace entries for reproducibility cross-check.
    def slim(summary: dict[str, Any]) -> dict[str, Any]:
        x = dict(summary)
        for k in ("obs_sha_trace", "action_trace", "cross_argmax_trace"):
            v = x.pop(k, None)
            if v is not None:
                x[f"{k}_len"] = int(len(v))
        return x

    payload = {
        "_header": True,
        "tool": "tools/k4_1_eval_obs_logit_probe.py",
        "phase": "H5-K-K4.1",
        "config_path": str(config_path),
        "seeds": list(seeds),
        "max_steps": int(max_steps),
        "actor_label": actor_label,
        "cross_label": cross_label,
        "model_paths": model_paths,
        "model_sha256": model_sha256s,
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pass1_actor_summaries": [slim(s) for s in pass1_summaries],
        "pass2_actor_summaries": [slim(s) for s in pass2_summaries],
        "trajectory_comparison": trajectory_cmp,
        "logit_aggregate": logit_agg,
        "classification": classification,
    }
    with json_path.open("w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2)

    # CSV: per-step rows from Pass 1 (the rich capture pass).
    if pass1_rows:
        fieldnames = list(pass1_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in pass1_rows:
                writer.writerow(r)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("# no rows produced\n")
    return json_path, csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="k4_1_eval_obs_logit_probe",
        description=(
            "K4.1 eval-observation panel-logit mechanism diagnostic. "
            "Probes K3.5c 2048 and K3.5c 10000 on observations captured "
            "from actual K3.5c eval rollouts on seeds 1000-1009."
        ),
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML env+algo config (use the K3.5c entropy config).",
    )
    p.add_argument(
        "--seeds", default="1000-1009",
        help="Eval seed range, default '1000-1009'.",
    )
    p.add_argument(
        "--max-steps", type=int, default=1800,
        help="Max steps per episode (default 1800, matches K3.5c eval).",
    )
    p.add_argument(
        "--out-dir", default="runs/phase_k",
        help="Output directory under runs/.",
    )
    p.add_argument(
        "--label", default="k4_1_eval_obs_logit_probe",
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


def _parse_seed_range(spec: str) -> list[int]:
    out: list[int] = []
    for raw in spec.split(","):
        tok = raw.strip()
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

def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seed_range(args.seeds)
    if not seeds:
        raise ValueError("no eval seeds parsed")

    # Load both checkpoints once on cpu; reused across both passes.
    p2048 = Path(args.k3_5c_2048)
    p10000 = Path(args.k3_5c_10000)
    if not p2048.exists():
        raise FileNotFoundError(f"K3.5c 2048 checkpoint not found: {p2048}")
    if not p10000.exists():
        raise FileNotFoundError(f"K3.5c 10000 checkpoint not found: {p10000}")
    print(f"[k4_1] loading k3_5c_2048 from {p2048}", flush=True)
    model_2048 = PPO.load(str(p2048), device="cpu")
    print(f"[k4_1] loading k3_5c_10000 from {p10000}", flush=True)
    model_10000 = PPO.load(str(p10000), device="cpu")
    sha_2048 = _sha256_of_file(p2048)
    sha_10000 = _sha256_of_file(p10000)
    if sha_2048 == sha_10000:
        raise RuntimeError(
            "K3.5c 2048 and K3.5c 10000 checkpoints share identical "
            "SHA-256; model-loading bug suspected."
        )
    print(f"[k4_1] k3_5c_2048 sha256={sha_2048}", flush=True)
    print(f"[k4_1] k3_5c_10000 sha256={sha_10000}", flush=True)

    # Pass 1: actor = K3.5c 10000 (the reference / later checkpoint),
    # cross-probe = K3.5c 2048. Rich per-step capture.
    pass1_label = "k4_1_pass1_actor_10000"
    pass1_env = build_eval_env(cfg, seed=int(seeds[0]), run_dir=out_dir / f"godot_{pass1_label}")
    pass1_summaries: list[dict[str, Any]] = []
    pass1_rows: list[dict[str, Any]] = []
    print(f"[k4_1] Pass 1 actor=k3_5c_10000 cross=k3_5c_2048 seeds={seeds}", flush=True)
    try:
        for i, s in enumerate(seeds):
            print(f"[k4_1] Pass 1 seed={s} starting...", flush=True)
            summary, rows = rollout_with_full_capture(
                env=pass1_env,
                actor_model=model_10000,
                cross_model=model_2048,
                actor_label="k3_5c_10000",
                cross_label="k3_5c_2048",
                eval_seed=int(s),
                episode_id=int(i),
                max_steps=int(args.max_steps),
            )
            pass1_summaries.append(summary)
            pass1_rows.extend(rows)
            print(
                f"[k4_1] Pass 1 seed={s} len={summary['episode_length']} "
                f"coll={summary['collision']} to={summary['timeout']} "
                f"cross_diffs={summary['cross_argmax_diffs']} "
                f"first_diff={summary['first_diff_step']} "
                f"elapsed={summary['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        try:
            pass1_env.close()
        except Exception:
            pass

    # Pass 2: actor = K3.5c 2048, minimal capture for trajectory parity
    # cross-check against Pass 1.
    pass2_label = "k4_1_pass2_actor_2048"
    pass2_env = build_eval_env(cfg, seed=int(seeds[0]), run_dir=out_dir / f"godot_{pass2_label}")
    pass2_summaries: list[dict[str, Any]] = []
    print(f"[k4_1] Pass 2 actor=k3_5c_2048 seeds={seeds}", flush=True)
    try:
        for i, s in enumerate(seeds):
            print(f"[k4_1] Pass 2 seed={s} starting...", flush=True)
            summary = rollout_minimal(
                env=pass2_env,
                actor_model=model_2048,
                eval_seed=int(s),
                episode_id=int(i),
                max_steps=int(args.max_steps),
                actor_label="k3_5c_2048",
            )
            pass2_summaries.append(summary)
            print(
                f"[k4_1] Pass 2 seed={s} len={summary['episode_length']} "
                f"coll={summary['collision']} to={summary['timeout']} "
                f"elapsed={summary['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        try:
            pass2_env.close()
        except Exception:
            pass

    print("[k4_1] comparing trajectories Pass 1 vs Pass 2...", flush=True)
    trajectory_cmp = compare_trajectories(pass1_summaries, pass2_summaries)
    print("[k4_1] aggregating logit stats on Pass 1 captures...", flush=True)
    logit_agg = aggregate_logit_stats(
        pass1_rows, actor_label="k3_5c_10000", cross_label="k3_5c_2048",
    )
    classification = classify_k4_1(
        logit_agg, trajectory_cmp,
        actor_label="k3_5c_10000", cross_label="k3_5c_2048",
    )

    json_path, csv_path = write_outputs(
        out_dir=out_dir,
        label=args.label,
        config_path=args.config,
        seeds=seeds,
        max_steps=int(args.max_steps),
        actor_label="k3_5c_10000",
        cross_label="k3_5c_2048",
        model_paths={"k3_5c_2048": str(p2048), "k3_5c_10000": str(p10000)},
        model_sha256s={"k3_5c_2048": sha_2048, "k3_5c_10000": sha_10000},
        pass1_summaries=pass1_summaries,
        pass2_summaries=pass2_summaries,
        pass1_rows=pass1_rows,
        trajectory_cmp=trajectory_cmp,
        logit_agg=logit_agg,
        classification=classification,
    )
    print(
        f"[k4_1] DONE verdict={classification['verdict']}",
        flush=True,
    )
    print(f"[k4_1] {classification['rationale']}", flush=True)
    print(f"[k4_1] wrote {json_path}", flush=True)
    print(f"[k4_1] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
