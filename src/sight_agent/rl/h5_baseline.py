"""H5 baseline / non-saturation evaluation harness.

Implements the four-policy evaluation contract defined in
``docs/sight-h5-plan.md`` sections 3-5:

    stay_only       deterministic action 1 every step (do-nothing floor)
    seeded_random   uniform Discrete(3) under a policy-side NumPy RNG
                    seeded as ``eval_seed + 1_000_000`` so the policy
                    action stream is independent of the env reset RNG
    untrained_cnn   freshly-constructed SB3 PPO CnnPolicy with zero
                    training steps (H4 boundary signal)
    trained_cnn     SB3 PPO loaded from a completed train run's
                    ``model.zip`` (path-validated only when the caller
                    has no live trained run on disk)

Each policy is evaluated against the SAME microgame profile, seed set,
``max_steps``, observation mode, and eval posture. Per-policy summaries
land under ``<run_dir>/evaluation/<policy>/summary.json`` with per-seed
rows and aggregates; the run-level ``<run_dir>/evaluation/index.json``
records the seed list, the canonical non-saturation thresholds, and the
overall saturation decision.

The non-saturation gate is implemented here per the canonical thresholds
pinned in ``docs/sight-h5-plan.md`` section 5:

    saturated negative control if
        timeout_rate >= 0.50 OR
        mean_episode_length >= 0.80 * max_steps
    profile FAILs if any of (stay_only, seeded_random, untrained_cnn)
    is saturated

Per-seed rows record reward, episode_length, collision, timeout, and
elapsed_seconds. ``obs.data`` is never copied into the summaries; the
audit shape mirrors H4 (artifacts contain metadata, not raw frames).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np


# Canonical non-saturation thresholds per docs/sight-h5-plan.md section 5.
# These are NOT runtime knobs; changing them requires a docs-level
# amendment to that section. The harness records the exact values used
# in every policy summary and in the run-level index so an audit can
# detect drift.
NON_SATURATION_TIMEOUT_RATE_THRESHOLD: float = 0.50
NON_SATURATION_LENGTH_RATIO_THRESHOLD: float = 0.80

# The three negative-control policies whose saturation determines whether
# the configured Signal Dodge profile passes the H5 non-saturation gate.
NEGATIVE_CONTROL_POLICIES: tuple[str, ...] = (
    "stay_only",
    "seeded_random",
    "untrained_cnn",
)

# Policy-side RNG seed derivation. The eval seed seeds the env reset
# RNG; the policy uses an offset so its action stream is independent of
# env stochasticity. See docs/sight-h5-plan.md section 3 (seeded_random).
POLICY_SEED_OFFSET: int = 1_000_000


def derive_policy_seed(eval_seed: int) -> int:
    """Deterministic policy-side RNG seed from a given eval seed.

    Pure function so tests can assert independence from env-reset seeding
    without instantiating an env.
    """
    return int(eval_seed) + POLICY_SEED_OFFSET

class _Policy(Protocol):
    """Minimal interface H5 policies expose to the rollout loop.

    Implementations are deliberately lightweight; the rollout layer is
    agnostic to whether the underlying decision is deterministic, random,
    or a neural network. ``branch`` is recorded in the per-policy summary
    as ``branch_metadata`` so audits can confirm which code path produced
    a given evaluation row.
    """

    name: str
    branch: str

    def reset_for_seed(self, eval_seed: int) -> None: ...

    def predict(self, obs: Any, deterministic: bool = True) -> int: ...


@dataclass
class StayOnlyPolicy:
    """Deterministic ``action 1`` (stay) every step."""

    name: str = "stay_only"
    branch: str = "deterministic_stay"

    def reset_for_seed(self, eval_seed: int) -> None:  # noqa: ARG002
        # No internal state; reset is a no-op. Signature kept for the
        # Policy protocol.
        return None

    def predict(self, obs: Any, deterministic: bool = True) -> int:  # noqa: ARG002
        return 1

class SeededRandomPolicy:
    """Uniform Discrete(3) policy seeded per eval seed.

    The policy maintains its own ``numpy.random.Generator`` constructed
    via ``numpy.random.default_rng(derive_policy_seed(eval_seed))`` on
    every call to :meth:`reset_for_seed`. This RNG is policy-side: it
    is NEVER drawn from the env's reset RNG, which means the policy's
    action stream is byte-identical across environments that share the
    same eval seed but differ in env-internal randomness (e.g. pre-mode-
    lock physics-tick variance from H3).
    """

    name: str = "seeded_random"
    branch: str = "policy_side_numpy_rng"

    def __init__(self, n_actions: int = 3) -> None:
        if not isinstance(n_actions, int) or n_actions < 2:
            raise ValueError(
                f"n_actions must be int >= 2, got {n_actions!r}"
            )
        self._n_actions = int(n_actions)
        self._rng: np.random.Generator | None = None
        self._policy_seed: int | None = None

    @property
    def policy_seed(self) -> int | None:
        return self._policy_seed

    def reset_for_seed(self, eval_seed: int) -> None:
        self._policy_seed = derive_policy_seed(int(eval_seed))
        self._rng = np.random.default_rng(self._policy_seed)

    def predict(self, obs: Any, deterministic: bool = True) -> int:  # noqa: ARG002
        if self._rng is None:
            raise RuntimeError(
                "SeededRandomPolicy.predict called before reset_for_seed"
            )
        # ``deterministic`` is honored only in the sense that the RNG
        # stream is deterministic given the policy seed; the action is
        # never an argmax. Mirrors SB3 stochastic-policy semantics.
        return int(self._rng.integers(0, self._n_actions))

class _SB3PolicyAdapter:
    """Internal adapter shared by untrained and trained CnnPolicy.

    Wraps an SB3 ``BaseAlgorithm`` (PPO instance) so the rollout layer
    sees the same ``reset_for_seed`` / ``predict`` surface as the
    handcrafted policies. ``model.predict`` already accepts the
    VecEnv-shaped observation and returns ``(action_array, state)``;
    the adapter extracts the scalar action for the n_envs=1 case used
    by the Godot env in H5.
    """

    def __init__(
        self,
        *,
        name: str,
        branch: str,
        model: Any,
    ) -> None:
        self.name = name
        self.branch = branch
        self._model = model

    def reset_for_seed(self, eval_seed: int) -> None:
        # Reseed numpy and (defensively) torch so any policy-internal
        # stochastic sampling is reproducible per eval seed. SB3
        # CnnPolicy.predict(deterministic=True) does not sample, but
        # we keep this for parity with stochastic-eval future variants.
        import random

        import torch

        random.seed(int(eval_seed))
        np.random.seed(int(eval_seed))
        torch.manual_seed(int(eval_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(eval_seed))

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        action_arr, _state = self._model.predict(obs, deterministic=bool(deterministic))
        action_arr = np.asarray(action_arr).reshape(-1)
        return int(action_arr[0])

def build_untrained_cnn_policy(env: Any, *, seed: int = 0) -> _SB3PolicyAdapter:
    """Construct an SB3 PPO CnnPolicy with zero training steps.

    ``env`` is a SB3 VecEnv whose observation space defines the network
    shape. The returned adapter wraps the freshly-constructed model;
    callers must NOT invoke ``model.learn`` on it before evaluation.
    """
    from stable_baselines3 import PPO

    model = PPO(
        policy="CnnPolicy",
        env=env,
        seed=int(seed),
        device="cpu",
        verbose=0,
    )
    return _SB3PolicyAdapter(
        name="untrained_cnn",
        branch="ppo_cnnpolicy_zero_steps",
        model=model,
    )


def resolve_trained_model_path(train_run_dir: Path) -> Path:
    """Return the path to ``model.zip`` for a completed train run.

    Reuses :func:`sight_agent.rl.evaluate._resolve_model_path` so the
    trained-policy branch shares the same artifact-resolution logic as
    the H2 out-of-band evaluator. This is the path validated by the
    Step 1 unit tests when no trained model exists end-to-end yet.
    """
    from .evaluate import _load_train_summary, _resolve_model_path

    train_run_dir = Path(train_run_dir)
    summary = _load_train_summary(train_run_dir)
    return _resolve_model_path(train_run_dir, summary)


def build_trained_cnn_policy(
    train_run_dir: Path,
    *,
    env: Any | None = None,
) -> _SB3PolicyAdapter:
    """Load an SB3 PPO model from a completed train run.

    The model is loaded with ``device='cpu'`` for reproducible CPU eval
    per ``docs/sight-h5-plan.md`` section 7 (no GPU dependency for
    acceptance). When ``env`` is supplied it is attached to the loaded
    model so ``predict`` can validate observation shapes.
    """
    from stable_baselines3 import PPO

    model_path = resolve_trained_model_path(train_run_dir)
    model = PPO.load(str(model_path), env=env, device="cpu")
    return _SB3PolicyAdapter(
        name="trained_cnn",
        branch="ppo_cnnpolicy_loaded_from_disk",
        model=model,
    )

@dataclass(frozen=True)
class EpisodeResult:
    """Per-seed rollout result. ``obs.data`` is deliberately excluded."""

    seed: int
    policy_seed: int | None
    reward: float
    episode_length: int
    collision: bool
    timeout: bool
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "policy_seed": (None if self.policy_seed is None else int(self.policy_seed)),
            "reward": float(self.reward),
            "episode_length": int(self.episode_length),
            "collision": bool(self.collision),
            "timeout": bool(self.timeout),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def rollout_one_episode(
    policy: _Policy,
    env: Any,
    max_steps: int,
    deterministic: bool = True,
) -> EpisodeResult:
    """Run one episode of ``policy`` on ``env`` and return per-seed metrics.

    Caller is responsible for calling ``env.seed(seed)`` and
    ``policy.reset_for_seed(seed)`` BEFORE this function. The function
    itself only consumes the next ``env.reset()`` and runs until done.

    Collision vs timeout discrimination uses the SB3 ``DummyVecEnv``
    convention: ``info["TimeLimit.truncated"] = truncated and not
    terminated``. Combined with the Signal Dodge contract (truncation
    only at ``max_steps``; termination only on collision), this gives a
    clean classification at the final step.
    """
    obs = env.reset()
    t0 = time.time()
    ep_reward = 0.0
    ep_len = 0
    final_info: dict[str, Any] = {}
    done_flag = False
    while ep_len < max_steps:
        action = policy.predict(obs, deterministic=deterministic)
        action_arr = np.asarray([int(action)])
        obs, reward, dones, infos = env.step(action_arr)
        ep_reward += float(np.asarray(reward).sum())
        ep_len += 1
        done_flag = bool(np.asarray(dones).any())
        if done_flag:
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                final_info = dict(infos[0])
            break
    elapsed = time.time() - t0

    if not done_flag:
        # Loop terminated via the while condition without env-reported
        # done. Defensive: the Godot env always reports truncated=True at
        # max_steps, but a fake or buggy env might not. Treat as timeout.
        collision = False
        timeout = True
    else:
        truncated_flag = bool(final_info.get("TimeLimit.truncated", False))
        if truncated_flag:
            collision = False
            timeout = True
        else:
            collision = True
            timeout = False
    policy_seed = getattr(policy, "policy_seed", None)
    return EpisodeResult(
        seed=-1,
        policy_seed=policy_seed,
        reward=float(ep_reward),
        episode_length=int(ep_len),
        collision=bool(collision),
        timeout=bool(timeout),
        elapsed_seconds=float(elapsed),
    )


def _aggregate(values: list[float]) -> dict[str, float]:
    """mean / median / min / max / std for a list of floats.

    Empty input returns zeros so the summary shape stays stable when an
    eval fails to produce any episodes (status: error).
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    vs = [float(v) for v in values]
    return {
        "mean": float(statistics.fmean(vs)),
        "median": float(statistics.median(vs)),
        "min": float(min(vs)),
        "max": float(max(vs)),
        "std": float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    }


def build_policy_summary(
    *,
    policy_name: str,
    branch: str,
    env_id: str,
    observation_mode: str,
    max_steps: int,
    seeds: list[int],
    rows: list[EpisodeResult],
    git_commit: str | None,
    artifact_paths: dict[str, str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the per-policy summary dict.

    The result is JSON-ready and includes per-seed rows plus aggregates,
    the configured non-saturation thresholds, and a per-policy saturation
    decision computed in isolation (the overall run-level decision is
    computed in :func:`evaluate_non_saturation`).
    """
    rewards = [r.reward for r in rows]
    lengths = [r.episode_length for r in rows]
    elapsed = [r.elapsed_seconds for r in rows]
    collision_rate = (
        sum(1 for r in rows if r.collision) / len(rows) if rows else 0.0
    )
    timeout_rate = (
        sum(1 for r in rows if r.timeout) / len(rows) if rows else 0.0
    )
    thr = thresholds or canonical_non_saturation_thresholds()
    mean_length = _aggregate(lengths)["mean"]
    saturated = (
        timeout_rate >= float(thr["timeout_rate_threshold"])
    ) or (
        mean_length >= float(thr["length_ratio_threshold"]) * float(max_steps)
    )
    return {
        "policy_name": str(policy_name),
        "branch_metadata": str(branch),
        "env_id": str(env_id),
        "observation_mode": str(observation_mode),
        "max_steps": int(max_steps),
        "seeds": [int(s) for s in seeds],
        "per_seed": [r.as_dict() for r in rows],
        "aggregate_reward": _aggregate(rewards),
        "aggregate_episode_length": _aggregate(lengths),
        "aggregate_elapsed_seconds": _aggregate(elapsed),
        "collision_rate": float(collision_rate),
        "timeout_rate": float(timeout_rate),
        "non_saturation_thresholds": dict(thr),
        "saturation_decision": {
            "is_negative_control": policy_name in NEGATIVE_CONTROL_POLICIES,
            "saturated": bool(saturated),
            "mean_episode_length": float(mean_length),
            "length_ratio": (
                float(mean_length) / float(max_steps) if max_steps > 0 else 0.0
            ),
        },
        "artifact_paths": dict(artifact_paths),
        "git_commit": (None if git_commit is None else str(git_commit)),
    }


def canonical_non_saturation_thresholds() -> dict[str, float]:
    """Return the canonical thresholds pinned in the H5 plan.

    Returns a fresh dict so callers can mutate freely without affecting
    the module-level constants.
    """
    return {
        "timeout_rate_threshold": float(NON_SATURATION_TIMEOUT_RATE_THRESHOLD),
        "length_ratio_threshold": float(NON_SATURATION_LENGTH_RATIO_THRESHOLD),
    }


def evaluate_non_saturation(
    summaries: dict[str, dict[str, Any]],
    max_steps: int,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the H5 non-saturation gate across the four policy summaries.

    ``summaries`` is keyed by policy name. Only the three negative-control
    policies are checked; the trained policy's saturation is reported but
    does NOT affect the gate. The gate FAILs (``passed=False``) if any
    negative control is saturated by the canonical rule.
    """
    thr = thresholds or canonical_non_saturation_thresholds()
    timeout_rate_thr = float(thr["timeout_rate_threshold"])
    length_ratio_thr = float(thr["length_ratio_threshold"])
    max_steps_f = float(max_steps)

    per_policy: dict[str, dict[str, Any]] = {}
    saturated_negative_controls: list[str] = []
    for name, summ in summaries.items():
        timeout_rate = float(summ.get("timeout_rate", 0.0))
        mean_length = float(
            summ.get("aggregate_episode_length", {}).get("mean", 0.0)
        )
        saturated = (
            timeout_rate >= timeout_rate_thr
            or mean_length >= length_ratio_thr * max_steps_f
        )
        per_policy[name] = {
            "timeout_rate": float(timeout_rate),
            "mean_episode_length": float(mean_length),
            "length_ratio": (
                float(mean_length) / max_steps_f if max_steps_f > 0 else 0.0
            ),
            "saturated": bool(saturated),
            "is_negative_control": name in NEGATIVE_CONTROL_POLICIES,
        }
        if saturated and name in NEGATIVE_CONTROL_POLICIES:
            saturated_negative_controls.append(name)

    return {
        "thresholds": dict(thr),
        "max_steps": int(max_steps),
        "per_policy": per_policy,
        "saturated_negative_controls": sorted(saturated_negative_controls),
        "passed": len(saturated_negative_controls) == 0,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_policy_seeded(
    *,
    policy: _Policy,
    env_factory: Callable[[], Any],
    seeds: Iterable[int],
    max_steps: int,
    env_id: str,
    observation_mode: str,
    out_dir: Path,
    git_commit: str | None = None,
    deterministic: bool = True,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate one policy across ``seeds`` and write its summary.

    Builds the env via ``env_factory`` (one env per policy is the simplest
    contract; the Godot env's TCP port allocator picks a fresh port on
    every construction, so reuse across policies would still need a fresh
    factory call). For each seed: re-seeds the env, re-seeds the policy,
    rolls out one episode, captures the per-seed row.

    Writes ``<out_dir>/<policy>/summary.json`` and returns the same
    summary dict in memory.
    """
    seeds_list = [int(s) for s in seeds]
    policy_dir = Path(out_dir) / policy.name
    policy_dir.mkdir(parents=True, exist_ok=True)
    summary_path = policy_dir / "summary.json"
    episodes_path = policy_dir / "episodes.ndjson"

    env = env_factory()
    rows: list[EpisodeResult] = []
    try:
        for seed in seeds_list:
            try:
                env.seed(int(seed))
            except (AttributeError, TypeError):
                # Some VecEnv variants do not expose ``seed``; rely on the
                # env factory's own seeding posture in that case.
                pass
            policy.reset_for_seed(int(seed))
            row = rollout_one_episode(
                policy=policy,
                env=env,
                max_steps=int(max_steps),
                deterministic=deterministic,
            )
            row = EpisodeResult(
                seed=int(seed),
                policy_seed=row.policy_seed,
                reward=row.reward,
                episode_length=row.episode_length,
                collision=row.collision,
                timeout=row.timeout,
                elapsed_seconds=row.elapsed_seconds,
            )
            rows.append(row)
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = build_policy_summary(
        policy_name=policy.name,
        branch=policy.branch,
        env_id=env_id,
        observation_mode=observation_mode,
        max_steps=int(max_steps),
        seeds=seeds_list,
        rows=rows,
        git_commit=git_commit,
        artifact_paths={
            "summary": str(summary_path),
            "episodes": str(episodes_path),
        },
        thresholds=thresholds,
    )
    _write_json(summary_path, summary)
    # Per-seed NDJSON: one JSON object per row. obs.data is excluded by
    # construction (EpisodeResult does not carry it).
    with episodes_path.open("w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write(json.dumps(r.as_dict(), separators=(",", ":")) + "\n")
    return summary


def run_h5_baseline(
    *,
    run_dir: Path,
    env_id: str,
    observation_mode: str,
    max_steps: int,
    seeds: Iterable[int],
    env_factory_for_policy: Callable[[str], Any],
    policies: dict[str, _Policy],
    git_commit: str | None = None,
    deterministic: bool = True,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the H5 baseline harness end-to-end.

    Writes one summary per policy under
    ``<run_dir>/evaluation/<policy>/summary.json`` and the run-level
    saturation decision under ``<run_dir>/evaluation/index.json``.

    ``env_factory_for_policy(policy_name)`` returns a freshly-built
    VecEnv for the given policy. The harness does NOT cache envs across
    policies; the caller decides whether and how to share construction.
    For Godot, a fresh env per policy is the documented contract because
    the TCP port allocator assigns a new port per construction.

    Returns the run-level index dict (also written to disk).
    """
    run_dir = Path(run_dir)
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    thr = thresholds or canonical_non_saturation_thresholds()
    seeds_list = [int(s) for s in seeds]

    summaries: dict[str, dict[str, Any]] = {}
    for name, policy in policies.items():
        env_factory = lambda n=name: env_factory_for_policy(n)
        summary = evaluate_policy_seeded(
            policy=policy,
            env_factory=env_factory,
            seeds=seeds_list,
            max_steps=int(max_steps),
            env_id=env_id,
            observation_mode=observation_mode,
            out_dir=eval_dir,
            git_commit=git_commit,
            deterministic=deterministic,
            thresholds=thr,
        )
        summaries[name] = summary

    decision = evaluate_non_saturation(
        summaries={k: v for k, v in summaries.items()},
        max_steps=int(max_steps),
        thresholds=thr,
    )
    index = {
        "schema_version": 1,
        "kind": "h5_evaluation_index",
        "env_id": str(env_id),
        "observation_mode": str(observation_mode),
        "max_steps": int(max_steps),
        "seeds": seeds_list,
        "policies": sorted(policies.keys()),
        "non_saturation_thresholds": dict(thr),
        "saturation_decision": decision,
        "git_commit": (None if git_commit is None else str(git_commit)),
        "policy_summaries": {
            name: str(Path(eval_dir) / name / "summary.json")
            for name in policies
        },
    }
    _write_json(eval_dir / "index.json", index)
    return index
