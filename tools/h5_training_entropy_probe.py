"""Phase K K0 training-time entropy-collapse probe.

Runs a short SB3 PPO training session under the H5 pixel entropy
config, instrumenting PPO.train() internals to record per-update
rollout statistics, optimization losses, gradient norms, action_net
weight/bias deltas, and pre/post-update entropy and raw-logit margin
on the same rollout observations.

Scope per GPT Phase K K0 plan:
- Use configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml
- Override seed to 2, total_timesteps to 2048 (pilot gate, not whole diagnostic)
- No env change. No reward change. No architecture change.
- Instrument PPO train internals (subclass PPO, override train() with
  hooks). Callbacks alone cannot expose minibatch gradient/loss internals.
- Write runs/phase_k/entropy_probe_seed2.{ndjson, summary.json}.

Collapse thresholds for this probe (per GPT plan):
- mean rollout entropy < 0.20
- top-action fraction >= 0.95
- mean raw top1-top2 margin >= 4.0

Auto-classification:
- K-A: entropy collapses in first 1-3 PPO updates
- K-B: value/advantage signal degenerates (advantage std collapse AND
  near-zero explained variance) before entropy collapse
- K-C: entropy healthy through 2048; recommend rerun to 10000 next session
- K-D: entropy healthy but argmax/wedge behavior forms anyway;
       recommend K1 architecture probe next

Treats 2048 as a pilot gate. Does not auto-run 10000 timesteps. If
K-C, writes evidence and handoff recommending 10000 timesteps as the
next slice.

No model.zip is written by this tool. K0 is diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_TOOLS = _REPO_ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.utils import explained_variance  # noqa: E402

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.factories import make_env  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402


ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}

# Collapse thresholds per GPT Phase K K0 plan.
COLLAPSE_ENTROPY_LT = 0.20
COLLAPSE_TOP_ACTION_FRAC_GE = 0.95
COLLAPSE_MARGIN_GE = 4.0


def snapshot_action_net(policy: Any) -> dict[str, Any]:
    """Capture action_net Linear layer weight row norms and biases.

    Returns row-norm and bias arrays indexed by action id 0..2 plus a
    blake2b-128 digest over the action_net state_dict for cheap equality
    checks across updates.
    """
    import hashlib

    an = policy.action_net
    W = an.weight.detach().cpu().contiguous().to(th.float64).numpy()
    b = an.bias.detach().cpu().contiguous().to(th.float64).numpy()
    row_norms = np.linalg.norm(W, axis=1)
    h = hashlib.blake2b(digest_size=16)
    h.update(W.astype(np.float32).tobytes())
    h.update(b.astype(np.float32).tobytes())
    return {
        "row_norms": {ACTION_NAMES[i]: float(row_norms[i]) for i in range(W.shape[0])},
        "biases": {ACTION_NAMES[i]: float(b[i]) for i in range(b.shape[0])},
        "blake2b16": h.hexdigest(),
    }


def snapshot_rollout_policy_state(
    policy: Any, rollout_obs_tensor: th.Tensor
) -> dict[str, Any]:
    """Compute pre/post-update entropy and raw-logit margin on rollout obs.

    Uses the same actor-path extraction as tools/h5_activation_compare.py
    (extract_features -> mlp_extractor -> action_net -> softmax) so the
    values are commensurable with Phase I summaries. Operates under
    torch.no_grad() and does not affect the optimizer state.
    """
    with th.no_grad():
        features = policy.extract_features(rollout_obs_tensor)
        latent_pi, _latent_vf = policy.mlp_extractor(features)
        raw_logits = policy.action_net(latent_pi)
        probs = th.softmax(raw_logits, dim=-1)
        # Per-action probability fractions across the rollout (mean of probs).
        mean_probs = probs.mean(dim=0).detach().cpu().numpy().astype(np.float64)
        # Top-1 minus top-2 raw logit margin per step.
        top2 = th.topk(raw_logits, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy().astype(np.float64)
        # Entropy per step over the categorical distribution.
        eps = 1e-12
        log_probs = th.log(probs.clamp_min(eps))
        entropy_per_step = -(probs * log_probs).sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        # Per-step argmax over raw logits.
        argmax_per_step = raw_logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    n = int(margin.size)
    # Argmax fractions across the rollout obs.
    argmax_counts = {i: int(np.sum(argmax_per_step == i)) for i in range(3)}
    argmax_frac = {ACTION_NAMES[i]: float(argmax_counts[i] / n) if n else 0.0 for i in range(3)}
    return {
        "n": n,
        "entropy_mean": float(np.mean(entropy_per_step)) if n else 0.0,
        "entropy_min": float(np.min(entropy_per_step)) if n else 0.0,
        "entropy_max": float(np.max(entropy_per_step)) if n else 0.0,
        "margin_mean": float(np.mean(margin)) if n else 0.0,
        "margin_min": float(np.min(margin)) if n else 0.0,
        "margin_max": float(np.max(margin)) if n else 0.0,
        "mean_probs": {ACTION_NAMES[i]: float(mean_probs[i]) for i in range(3)},
        "argmax_fractions": argmax_frac,
        "top_argmax_action": ACTION_NAMES[int(max(argmax_counts.items(), key=lambda kv: kv[1])[0])],
        "top_argmax_fraction": float(max(argmax_frac.values())) if argmax_frac else 0.0,
    }


def compute_rollout_action_fractions(rollout_buffer: Any) -> dict[str, Any]:
    """Per-action fractions across the rollout's collected actions.

    The rollout buffer stores actions sampled by the policy during
    env interaction (not argmax). These are the actions the env
    actually saw and stepped on; their distribution tells us what
    the policy's sampling behavior looked like before this update.
    """
    actions = rollout_buffer.actions
    arr = np.asarray(actions).reshape(-1).astype(np.int64)
    n = int(arr.size)
    counts = {i: int(np.sum(arr == i)) for i in range(3)}
    fracs = {ACTION_NAMES[i]: float(counts[i] / n) if n else 0.0 for i in range(3)}
    top_action_id = int(max(counts.items(), key=lambda kv: kv[1])[0])
    return {
        "counts": {ACTION_NAMES[i]: int(counts[i]) for i in range(3)},
        "fractions": fracs,
        "top_action": ACTION_NAMES[top_action_id],
        "top_action_fraction": float(max(fracs.values())) if fracs else 0.0,
        "n_actions": n,
    }


def compute_rollout_episode_stats(rollout_buffer: Any) -> dict[str, Any]:
    """Best-effort episode-boundary stats from rollout buffer episode_starts.

    A True at index t means env was reset at step t (new episode). So
    the number of terminals during this rollout is roughly the number
    of True entries past the first, since the first True marks the
    starting condition of the buffer. Cannot distinguish collision vs
    timeout from buffer alone; that requires infos via callback.
    """
    es = np.asarray(rollout_buffer.episode_starts).reshape(-1).astype(np.bool_)
    n = int(es.size)
    n_resets = int(np.sum(es))
    return {
        "rollout_length": n,
        "episode_resets": n_resets,
        "first_step_was_reset": bool(es[0]) if n else False,
    }


def compute_advantage_return_value_stats(rollout_buffer: Any) -> dict[str, Any]:
    """Aggregate advantage, return, value, and value-error statistics."""
    adv = np.asarray(rollout_buffer.advantages).reshape(-1).astype(np.float64)
    ret = np.asarray(rollout_buffer.returns).reshape(-1).astype(np.float64)
    val = np.asarray(rollout_buffer.values).reshape(-1).astype(np.float64)
    err = (ret - val).astype(np.float64)
    n = int(adv.size)
    pos_frac = float(np.mean(adv > 0.0)) if n else 0.0
    neg_frac = float(np.mean(adv < 0.0)) if n else 0.0
    return {
        "advantages": {
            "mean": float(np.mean(adv)) if n else 0.0,
            "std": float(np.std(adv)) if n else 0.0,
            "min": float(np.min(adv)) if n else 0.0,
            "max": float(np.max(adv)) if n else 0.0,
            "positive_fraction": pos_frac,
            "negative_fraction": neg_frac,
        },
        "returns": {
            "mean": float(np.mean(ret)) if n else 0.0,
            "std": float(np.std(ret)) if n else 0.0,
            "min": float(np.min(ret)) if n else 0.0,
            "max": float(np.max(ret)) if n else 0.0,
        },
        "values": {
            "mean": float(np.mean(val)) if n else 0.0,
            "std": float(np.std(val)) if n else 0.0,
            "min": float(np.min(val)) if n else 0.0,
            "max": float(np.max(val)) if n else 0.0,
        },
        "value_error_vs_returns": {
            "mean": float(np.mean(err)) if n else 0.0,
            "std": float(np.std(err)) if n else 0.0,
            "abs_mean": float(np.mean(np.abs(err))) if n else 0.0,
        },
        "explained_variance": float(explained_variance(val, ret)) if n else 0.0,
    }


def compute_module_grad_norm(module: Any) -> float:
    """L2 norm of the concatenated gradient vector over a module's params."""
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        total_sq += float(p.grad.detach().pow(2).sum().item())
    return float(total_sq ** 0.5)


def compute_total_grad_norm(policy: Any) -> float:
    """L2 norm of the full policy parameter gradient vector."""
    total_sq = 0.0
    for p in policy.parameters():
        if p.grad is None:
            continue
        total_sq += float(p.grad.detach().pow(2).sum().item())
    return float(total_sq ** 0.5)


class InstrumentedPPO(PPO):
    """SB3 PPO with full training-internals instrumentation.

    Overrides train() to mirror SB3 2.8.0 PPO.train() logic
    (stable_baselines3/ppo/ppo.py) while capturing per-minibatch and
    per-update statistics. The mirroring is deliberate: a callback
    fires only at rollout boundaries, so it cannot capture grad norms
    between loss.backward() and clip_grad_norm_().

    After each train() call, appends one dict to self.probe_records
    summarizing that PPO update.
    """

    def __init__(self, *args: Any, probe_records: list | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_records: list = probe_records if probe_records is not None else []
        self._probe_update_idx: int = 0

    def train(self) -> None:
        # --- Pre-update snapshots ---------------------------------------------------
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        rollout_buffer = self.rollout_buffer
        rb_obs = th.as_tensor(np.asarray(rollout_buffer.observations)).to(self.device)
        # SB3 stores observations as (n_steps, n_envs, *obs_shape); reshape to
        # (n_steps*n_envs, *obs_shape) for batched policy evaluation.
        rb_obs_flat = rb_obs.reshape((-1,) + rb_obs.shape[2:])

        pre_action_net = snapshot_action_net(self.policy)
        pre_policy_state = snapshot_rollout_policy_state(self.policy, rb_obs_flat)
        rollout_action_stats = compute_rollout_action_fractions(rollout_buffer)
        rollout_episode_stats = compute_rollout_episode_stats(rollout_buffer)
        adv_ret_val_stats = compute_advantage_return_value_stats(rollout_buffer)

        # --- Mirror SB3 PPO.train() with per-minibatch instrumentation -------------
        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs_all: list[float] = []
        total_losses: list[float] = []

        # Pre-clip grad norm aggregations across all minibatches.
        total_grad_norms: list[float] = []
        feats_grad_norms: list[float] = []
        mlp_grad_norms: list[float] = []
        action_grad_norms: list[float] = []
        value_grad_norms: list[float] = []

        continue_training = True
        n_epochs_done = 0
        last_loss_value = float("nan")

        for epoch in range(self.n_epochs):
            approx_kl_divs_epoch: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.item()))
                clip_fraction = float(
                    th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                )
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(float(entropy_loss.item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                total_losses.append(float(loss.item()))
                last_loss_value = float(loss.item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = float(
                        th.mean((th.exp(log_ratio) - 1.0) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs_epoch.append(approx_kl_div)
                    approx_kl_divs_all.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    break

                # --- Optimization with grad-norm instrumentation ---
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Pre-clip total grad norm and per-module grad norms.
                total_grad_norms.append(compute_total_grad_norm(self.policy))
                feats_grad_norms.append(
                    compute_module_grad_norm(self.policy.features_extractor)
                )
                mlp_grad_norms.append(
                    compute_module_grad_norm(self.policy.mlp_extractor)
                )
                action_grad_norms.append(
                    compute_module_grad_norm(self.policy.action_net)
                )
                value_grad_norms.append(
                    compute_module_grad_norm(self.policy.value_net)
                )
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            n_epochs_done = epoch + 1
            if not continue_training:
                break

        # --- Post-update snapshots --------------------------------------------------
        post_action_net = snapshot_action_net(self.policy)
        post_policy_state = snapshot_rollout_policy_state(self.policy, rb_obs_flat)

        # SB3 logging parity (so model.learn() does not break if it
        # consults self.logger between calls).
        explained_var = float(
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
        )
        if self.logger is not None:
            self.logger.record(
                "train/entropy_loss", float(np.mean(entropy_losses))
            )
            self.logger.record(
                "train/policy_gradient_loss", float(np.mean(pg_losses))
            )
            self.logger.record("train/value_loss", float(np.mean(value_losses)))
            self.logger.record(
                "train/approx_kl",
                float(np.mean(approx_kl_divs_all)) if approx_kl_divs_all else 0.0,
            )
            self.logger.record(
                "train/clip_fraction", float(np.mean(clip_fractions))
            )
            self.logger.record("train/loss", last_loss_value)
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record(
                "train/n_updates", self._n_updates, exclude="tensorboard"
            )
            self.logger.record("train/clip_range", float(clip_range))
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", float(clip_range_vf))

        # --- Action-net weight/bias deltas -----------------------------------------
        action_net_delta = {
            "row_norm_delta": {
                a: float(post_action_net["row_norms"][a] - pre_action_net["row_norms"][a])
                for a in ACTION_NAMES.values()
            },
            "bias_delta": {
                a: float(post_action_net["biases"][a] - pre_action_net["biases"][a])
                for a in ACTION_NAMES.values()
            },
            "pre_blake2b16": pre_action_net["blake2b16"],
            "post_blake2b16": post_action_net["blake2b16"],
            "weights_changed": (
                pre_action_net["blake2b16"] != post_action_net["blake2b16"]
            ),
        }

        # --- Build per-update record -----------------------------------------------
        update_idx = self._probe_update_idx + 1
        self._probe_update_idx = update_idx
        record = {
            "update_idx": int(update_idx),
            "num_timesteps": int(self.num_timesteps),
            "n_epochs_done": int(n_epochs_done),
            "n_minibatches": int(len(pg_losses)),
            "clip_range": float(clip_range),
            "rollout_action_stats": rollout_action_stats,
            "rollout_episode_stats": rollout_episode_stats,
            "adv_ret_val_stats": adv_ret_val_stats,
            "pre_update": {
                "policy_state": pre_policy_state,
                "action_net": pre_action_net,
            },
            "post_update": {
                "policy_state": post_policy_state,
                "action_net": post_action_net,
            },
            "action_net_delta": action_net_delta,
            "delta_entropy_mean": float(
                post_policy_state["entropy_mean"] - pre_policy_state["entropy_mean"]
            ),
            "delta_margin_mean": float(
                post_policy_state["margin_mean"] - pre_policy_state["margin_mean"]
            ),
            "delta_top_argmax_fraction": float(
                post_policy_state["top_argmax_fraction"]
                - pre_policy_state["top_argmax_fraction"]
            ),
            "losses": {
                "policy_gradient_loss_mean": float(np.mean(pg_losses)),
                "entropy_loss_mean": float(np.mean(entropy_losses)),
                "value_loss_mean": float(np.mean(value_losses)),
                "total_loss_last": float(last_loss_value),
                "approx_kl_mean": (
                    float(np.mean(approx_kl_divs_all)) if approx_kl_divs_all else 0.0
                ),
                "approx_kl_max": (
                    float(np.max(approx_kl_divs_all)) if approx_kl_divs_all else 0.0
                ),
                "clip_fraction_mean": float(np.mean(clip_fractions)),
            },
            "grad_norms_preclip": {
                "total_mean": float(np.mean(total_grad_norms)),
                "total_max": float(np.max(total_grad_norms)),
                "features_extractor_mean": float(np.mean(feats_grad_norms)),
                "mlp_extractor_mean": float(np.mean(mlp_grad_norms)),
                "action_net_mean": float(np.mean(action_grad_norms)),
                "value_net_mean": float(np.mean(value_grad_norms)),
            },
            "max_grad_norm_clip": float(self.max_grad_norm),
            "explained_variance": explained_var,
            "_record_ts_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        self.probe_records.append(record)

        # Live progress print for the driver log.
        print(
            f"  upd {update_idx}: ts={self.num_timesteps} "
            f"H_pre={pre_policy_state['entropy_mean']:.4f} "
            f"H_post={post_policy_state['entropy_mean']:.4f} "
            f"margin_pre={pre_policy_state['margin_mean']:.4f} "
            f"margin_post={post_policy_state['margin_mean']:.4f} "
            f"top_act={rollout_action_stats['top_action']}={rollout_action_stats['top_action_fraction']:.3f} "
            f"adv_std={adv_ret_val_stats['advantages']['std']:.4f} "
            f"ev={explained_var:.4f} "
            f"|g|={float(np.mean(total_grad_norms)):.4f}",
            flush=True,
        )


def _is_collapsed(record: dict[str, Any]) -> dict[str, bool]:
    """Return per-criterion collapse flags for one update record.

    Uses post-update stats since those reflect the policy after this
    update's gradient step. For top-action-fraction, uses the rollout
    sampled action fractions because that is the operational quantity
    the env saw and what the env-policy coupling cares about.
    """
    post = record["post_update"]["policy_state"]
    rs = record["rollout_action_stats"]
    return {
        "entropy": bool(post["entropy_mean"] < COLLAPSE_ENTROPY_LT),
        "top_action_fraction": bool(
            rs["top_action_fraction"] >= COLLAPSE_TOP_ACTION_FRAC_GE
        ),
        "margin": bool(post["margin_mean"] >= COLLAPSE_MARGIN_GE),
    }


def classify_collapse(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Map per-update probe records to K-A / K-B / K-C / K-D.

    Rules:
    - K-A: any of {entropy, top_action_fraction, margin} crosses its
      threshold by update index 3 (i.e. within first 1-3 updates).
    - K-B: advantage std collapses to near zero (< 0.05) AND
      explained_variance becomes negative or near zero (< 0.10) before
      any entropy threshold crosses. Treated as value/advantage signal
      degeneration preceding entropy collapse. Advantage collapse is
      the load-bearing signal; low EV alone is normal early in PPO
      and is supporting evidence, not an independent tripwire.
    - K-C: no threshold crosses through the entire probe.
    - K-D: entropy stays above 0.20 throughout AND raw margin stays
      below 4.0 throughout, but rollout top-action-fraction >= 0.95
      forms anyway (wedge behavior without distributional collapse).
    Priority: K-B if value/advantage degeneration leads any entropy
    flag; else K-A if early collapse; else K-D if wedge-only; else K-C.
    """
    if not records:
        return {"verdict": "K-C", "rationale": "no PPO updates ran"}

    # Per-update collapse flags.
    flags = [_is_collapsed(r) for r in records]

    # Value/advantage degeneration detector for K-B.
    # AND semantics: advantage collapse is the load-bearing signal; low
    # explained_variance alone is normal early in PPO (near zero in the
    # first ~2k timesteps regardless of degeneration) and is supporting
    # evidence, not an independent tripwire.
    def _value_adv_degenerate(r: dict[str, Any]) -> bool:
        adv_std = r["adv_ret_val_stats"]["advantages"]["std"]
        ev = r["explained_variance"]
        return bool(adv_std < 0.05 and ev < 0.10)

    first_entropy_collapse = None
    first_action_collapse = None
    first_margin_collapse = None
    first_value_adv = None
    for i, r in enumerate(records):
        if first_entropy_collapse is None and flags[i]["entropy"]:
            first_entropy_collapse = r["update_idx"]
        if first_action_collapse is None and flags[i]["top_action_fraction"]:
            first_action_collapse = r["update_idx"]
        if first_margin_collapse is None and flags[i]["margin"]:
            first_margin_collapse = r["update_idx"]
        if first_value_adv is None and _value_adv_degenerate(r):
            first_value_adv = r["update_idx"]

    earliest_collapse_idxs = [
        x for x in (first_entropy_collapse, first_action_collapse, first_margin_collapse)
        if x is not None
    ]
    any_collapse = bool(earliest_collapse_idxs)

    # K-B: value/advantage signal degeneration precedes any entropy collapse.
    if (
        first_value_adv is not None
        and first_entropy_collapse is not None
        and first_value_adv < first_entropy_collapse
    ):
        return {
            "verdict": "K-B",
            "rationale": (
                f"value/advantage degeneration at update {first_value_adv} "
                f"precedes entropy collapse at update {first_entropy_collapse}"
            ),
            "first_value_adv_degen": int(first_value_adv),
            "first_entropy_collapse": int(first_entropy_collapse),
            "first_action_collapse": (
                int(first_action_collapse)
                if first_action_collapse is not None else None
            ),
            "first_margin_collapse": (
                int(first_margin_collapse)
                if first_margin_collapse is not None else None
            ),
        }

    if any_collapse:
        earliest = min(earliest_collapse_idxs)
        # K-A: early collapse within updates 1-3.
        if earliest <= 3:
            return {
                "verdict": "K-A",
                "rationale": (
                    f"collapse threshold crossed at update {earliest} "
                    f"(within first 3 PPO updates); entropy_first="
                    f"{first_entropy_collapse} action_first="
                    f"{first_action_collapse} margin_first="
                    f"{first_margin_collapse}"
                ),
                "first_entropy_collapse": (
                    int(first_entropy_collapse)
                    if first_entropy_collapse is not None else None
                ),
                "first_action_collapse": (
                    int(first_action_collapse)
                    if first_action_collapse is not None else None
                ),
                "first_margin_collapse": (
                    int(first_margin_collapse)
                    if first_margin_collapse is not None else None
                ),
            }
        # K-D vs late K-A: if only top_action_fraction crossed (entropy and
        # margin healthy throughout), call it K-D; otherwise late K-A.
        if (
            first_entropy_collapse is None
            and first_margin_collapse is None
            and first_action_collapse is not None
        ):
            return {
                "verdict": "K-D",
                "rationale": (
                    f"rollout top-action fraction >= 0.95 at update "
                    f"{first_action_collapse} but entropy stayed >= 0.20 "
                    "and raw margin stayed < 4.0 throughout (wedge behavior "
                    "without distributional collapse); architecture probe "
                    "next"
                ),
                "first_action_collapse": int(first_action_collapse),
            }
        # Late entropy/margin collapse: still K-A semantically (entropy
        # collapses inside the probe budget) but at later updates 4..N.
        return {
            "verdict": "K-A",
            "rationale": (
                f"collapse threshold crossed at update {earliest} (later "
                f"than first 3 PPO updates but still within probe budget); "
                f"entropy_first={first_entropy_collapse} action_first="
                f"{first_action_collapse} margin_first={first_margin_collapse}"
            ),
            "first_entropy_collapse": (
                int(first_entropy_collapse)
                if first_entropy_collapse is not None else None
            ),
            "first_action_collapse": (
                int(first_action_collapse)
                if first_action_collapse is not None else None
            ),
            "first_margin_collapse": (
                int(first_margin_collapse)
                if first_margin_collapse is not None else None
            ),
        }

    # K-C: nothing collapsed through the entire probe.
    return {
        "verdict": "K-C",
        "rationale": (
            f"no collapse threshold crossed across {len(records)} PPO updates; "
            "entropy stayed >= 0.20, rollout top-action fraction stayed < 0.95, "
            "raw margin stayed < 4.0. Recommend rerun same probe to 10000 "
            "timesteps next session before drawing structural conclusions."
        ),
    }

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="h5_training_entropy_probe",
        description=(
            "Phase K K0 training-time entropy-collapse probe. Instruments "
            "SB3 PPO train() to record per-update rollout stats, "
            "optimization losses, gradient norms, action_net deltas, and "
            "pre/post-update entropy and raw-logit margin on rollout obs."
        ),
    )
    p.add_argument("--config", required=True, help="Path to YAML env+algo config.")
    p.add_argument(
        "--seed", type=int, default=2,
        help="Train seed (overrides run.seed). K0 default 2.",
    )
    p.add_argument(
        "--total-timesteps", type=int, default=2048,
        help=(
            "Override train.total_timesteps. K0 default 2048 (pilot gate). "
            "Per GPT plan: do not auto-run 10000 in this prompt."
        ),
    )
    p.add_argument(
        "--out-dir", default="runs/phase_k",
        help="Output directory under runs/.",
    )
    p.add_argument(
        "--label", default="entropy_probe_seed2",
        help="Output filename stem.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / f"{args.label}.ndjson"
    summary_path = out_dir / f"{args.label}.summary.json"
    run_dir = out_dir / f"godot_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_id = cfg["env"]["id"]
    godot_extra = resolve_godot_kwargs(cfg)
    train_seed = int(args.seed)
    total_timesteps = int(args.total_timesteps)

    # Build the train VecEnv via the same factory the production trainer uses.
    if godot_extra:
        env = make_env(
            env_id, n_envs=int(cfg["env"]["n_envs"]),
            seed=train_seed, mode="train",
            run_dir=str(run_dir), **godot_extra,
        )
    else:
        env = make_env(
            env_id, n_envs=int(cfg["env"]["n_envs"]),
            seed=train_seed, mode="train",
        )

    algo_cfg = cfg["algo"]
    hyperparams = dict(algo_cfg.get("hyperparams") or {})
    policy = algo_cfg["policy"]
    device = algo_cfg.get("device", "cpu")

    probe_records: list[dict[str, Any]] = []
    model = InstrumentedPPO(
        policy=policy,
        env=env,
        seed=train_seed,
        device=device,
        probe_records=probe_records,
        **hyperparams,
    )

    print(
        f"[h5_training_entropy_probe] config={args.config} seed={train_seed} "
        f"total_timesteps={total_timesteps} n_steps={hyperparams.get('n_steps')} "
        f"batch_size={hyperparams.get('batch_size')} n_epochs={hyperparams.get('n_epochs')} "
        f"ent_coef={hyperparams.get('ent_coef')}",
        flush=True,
    )
    print(
        f"[h5_training_entropy_probe] expecting "
        f"{total_timesteps // int(hyperparams.get('n_steps') or 1)} PPO updates",
        flush=True,
    )

    started_at = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
    finally:
        try:
            env.close()
        except Exception:
            pass
    elapsed = time.time() - started_at

    # --- Write NDJSON ---------------------------------------------------------------
    header = {
        "_header": True,
        "tool": "tools/h5_training_entropy_probe.py",
        "phase": "H5-K-K0",
        "config_path": str(args.config),
        "train_seed": int(train_seed),
        "total_timesteps_requested": int(total_timesteps),
        "n_updates_recorded": int(len(probe_records)),
        "elapsed_seconds": float(elapsed),
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ent_coef": float(hyperparams.get("ent_coef", 0.0)),
        "n_steps": int(hyperparams.get("n_steps", 0)),
        "batch_size": int(hyperparams.get("batch_size", 0)),
        "n_epochs": int(hyperparams.get("n_epochs", 0)),
        "learning_rate": float(hyperparams.get("learning_rate", 0.0)),
        "gamma": float(hyperparams.get("gamma", 0.99)),
        "gae_lambda": float(hyperparams.get("gae_lambda", 0.95)),
        "clip_range": (
            float(hyperparams["clip_range"])
            if "clip_range" in hyperparams else None
        ),
        "vf_coef": float(getattr(model, "vf_coef", 0.5)),
        "max_grad_norm": float(getattr(model, "max_grad_norm", 0.5)),
        "collapse_thresholds": {
            "entropy_lt": float(COLLAPSE_ENTROPY_LT),
            "top_action_fraction_ge": float(COLLAPSE_TOP_ACTION_FRAC_GE),
            "margin_ge": float(COLLAPSE_MARGIN_GE),
        },
    }
    with ndjson_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for rec in probe_records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # --- Compute verdict + summary -------------------------------------------------
    verdict = classify_collapse(probe_records)

    # Per-update one-line digest for the summary file.
    digest: list[dict[str, Any]] = []
    for rec in probe_records:
        flags = _is_collapsed(rec)
        digest.append({
            "update_idx": int(rec["update_idx"]),
            "num_timesteps": int(rec["num_timesteps"]),
            "rollout_top_action": rec["rollout_action_stats"]["top_action"],
            "rollout_top_action_fraction": float(
                rec["rollout_action_stats"]["top_action_fraction"]
            ),
            "rollout_left_fraction": float(
                rec["rollout_action_stats"]["fractions"]["left"]
            ),
            "rollout_stay_fraction": float(
                rec["rollout_action_stats"]["fractions"]["stay"]
            ),
            "rollout_right_fraction": float(
                rec["rollout_action_stats"]["fractions"]["right"]
            ),
            "det_argmax_pre": rec["pre_update"]["policy_state"]["top_argmax_action"],
            "det_argmax_fraction_pre": float(
                rec["pre_update"]["policy_state"]["top_argmax_fraction"]
            ),
            "det_argmax_post": rec["post_update"]["policy_state"]["top_argmax_action"],
            "det_argmax_fraction_post": float(
                rec["post_update"]["policy_state"]["top_argmax_fraction"]
            ),
            "policy_prob_left_post": float(
                rec["post_update"]["policy_state"]["mean_probs"]["left"]
            ),
            "policy_prob_stay_post": float(
                rec["post_update"]["policy_state"]["mean_probs"]["stay"]
            ),
            "policy_prob_right_post": float(
                rec["post_update"]["policy_state"]["mean_probs"]["right"]
            ),
            "entropy_pre": float(rec["pre_update"]["policy_state"]["entropy_mean"]),
            "entropy_post": float(rec["post_update"]["policy_state"]["entropy_mean"]),
            "margin_pre": float(rec["pre_update"]["policy_state"]["margin_mean"]),
            "margin_post": float(rec["post_update"]["policy_state"]["margin_mean"]),
            "delta_entropy": float(rec["delta_entropy_mean"]),
            "delta_margin": float(rec["delta_margin_mean"]),
            "advantage_std": float(rec["adv_ret_val_stats"]["advantages"]["std"]),
            "explained_variance": float(rec["explained_variance"]),
            "policy_gradient_loss_mean": float(
                rec["losses"]["policy_gradient_loss_mean"]
            ),
            "value_loss_mean": float(rec["losses"]["value_loss_mean"]),
            "entropy_loss_mean": float(rec["losses"]["entropy_loss_mean"]),
            "approx_kl_mean": float(rec["losses"]["approx_kl_mean"]),
            "clip_fraction_mean": float(rec["losses"]["clip_fraction_mean"]),
            "grad_norm_total_mean": float(
                rec["grad_norms_preclip"]["total_mean"]
            ),
            "grad_norm_action_net_mean": float(
                rec["grad_norms_preclip"]["action_net_mean"]
            ),
            "weights_changed": bool(rec["action_net_delta"]["weights_changed"]),
            "collapse_flags": flags,
        })

    summary = {
        "header": header,
        "verdict": verdict,
        "per_update_digest": digest,
        "n_updates": int(len(probe_records)),
        "final_action_net": (
            probe_records[-1]["post_update"]["action_net"] if probe_records else None
        ),
        "final_policy_state": (
            probe_records[-1]["post_update"]["policy_state"]
            if probe_records else None
        ),
        "initial_action_net": (
            probe_records[0]["pre_update"]["action_net"] if probe_records else None
        ),
        "initial_policy_state": (
            probe_records[0]["pre_update"]["policy_state"] if probe_records else None
        ),
    }
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"[h5_training_entropy_probe] DONE n_updates={len(probe_records)} "
        f"elapsed={elapsed:.1f}s verdict={verdict['verdict']}",
        flush=True,
    )
    print(
        f"[h5_training_entropy_probe] rationale: {verdict['rationale']}",
        flush=True,
    )
    print(
        f"[h5_training_entropy_probe] wrote {ndjson_path} and {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
