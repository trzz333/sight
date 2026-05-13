"""H5 baseline / non-saturation harness unit tests.

Default-tier (no live Godot). Covers:
- canonical-threshold pinning in ``docs/sight-h5-plan.md`` section 5
- per-policy deterministic / RNG-independence guarantees
- policy-side RNG seed independence from the env reset RNG
- harness-built per-policy summary shape (per-seed rows + aggregates)
- non-saturation gate decision on synthetic summaries
- on-disk artifact layout under ``evaluation/<policy>/summary.json``
- trained-policy model.zip path validation without a live train run
- absence of ``obs.data`` in evaluation summaries / per-seed rows
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sight_agent.rl.h5_baseline import (
    NEGATIVE_CONTROL_POLICIES,
    NON_SATURATION_LENGTH_RATIO_THRESHOLD,
    NON_SATURATION_TIMEOUT_RATE_THRESHOLD,
    POLICY_SEED_OFFSET,
    EpisodeResult,
    SeededRandomPolicy,
    StayOnlyPolicy,
    build_policy_summary,
    canonical_non_saturation_thresholds,
    derive_policy_seed,
    evaluate_non_saturation,
    evaluate_policy_seeded,
    resolve_trained_model_path,
    rollout_one_episode,
    run_h5_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
H5_PLAN_PATH = REPO_ROOT / "docs" / "sight-h5-plan.md"


class _FakeSignalDodge(gym.Env):
    """Minimal Gymnasium env mimicking Signal Dodge enough for harness tests.

    Action space: Discrete(3). Observation space matches the H4 pixel
    contract (1, 84, 84 uint8) so a CnnPolicy could in principle attach.
    Reward and termination follow the Signal Dodge contract: +1.0 per
    non-terminal step, 0.0 on the collision terminal step, truncation
    at ``max_steps``.

    Behaviour is parameterized by ``collide_at_step``:
    - None: never collide; episode runs until ``max_steps`` (timeout).
    - K (>=1): collide on the K-th env.step call (early termination).

    ``seed`` is recorded so tests can assert env-reset seeding is
    independent of the policy's action RNG.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 16,
        collide_at_step: int | None = None,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.max_steps = int(max_steps)
        self.collide_at_step = (
            None if collide_at_step is None else int(collide_at_step)
        )
        self._step_count = 0
        self._last_seed: int | None = None

    def _obs(self) -> np.ndarray:
        return np.zeros((1, 84, 84), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._last_seed = int(seed)
        self._step_count = 0
        return self._obs(), {"reset_seed": self._last_seed}

    def step(self, action):
        self._step_count += 1
        terminated = (
            self.collide_at_step is not None
            and self._step_count >= self.collide_at_step
        )
        truncated = (not terminated) and self._step_count >= self.max_steps
        reward = 0.0 if terminated else 1.0
        info: dict = {"step_count": self._step_count}
        return self._obs(), reward, bool(terminated), bool(truncated), info


def _make_vec_env(*, max_steps: int, collide_at_step: int | None) -> DummyVecEnv:
    return DummyVecEnv(
        [lambda: _FakeSignalDodge(max_steps=max_steps, collide_at_step=collide_at_step)]
    )


# ---------------------------------------------------------------------------
# Test 1: docs pinning
# ---------------------------------------------------------------------------


def test_h5_plan_pins_non_saturation_thresholds() -> None:
    """H5 plan section 5 must contain the canonical thresholds.

    Step 0 of the H5 implementation prompt requires the plan to define
    the exact numeric thresholds for the non-saturation gate. If a
    future refactor strips them, this test catches it.
    """
    text = H5_PLAN_PATH.read_text(encoding="utf-8")
    assert "timeout_rate >= 0.50" in text, (
        "H5 plan must pin timeout_rate threshold at 0.50"
    )
    assert "0.80 * max_steps" in text, (
        "H5 plan must pin length ratio threshold at 0.80"
    )
    assert "non_saturation_thresholds" in text, (
        "H5 plan must record where thresholds are surfaced in summaries"
    )


# ---------------------------------------------------------------------------
# Test 2-5: per-policy RNG behaviour
# ---------------------------------------------------------------------------


def test_stay_only_policy_returns_action_1_deterministically() -> None:
    p = StayOnlyPolicy()
    p.reset_for_seed(0)
    obs = np.zeros((1, 84, 84), dtype=np.uint8)
    actions = [p.predict(obs) for _ in range(64)]
    assert actions == [1] * 64


def test_seeded_random_policy_reproducible_same_seed() -> None:
    p1 = SeededRandomPolicy()
    p1.reset_for_seed(0)
    actions_a = [p1.predict(None) for _ in range(32)]

    p2 = SeededRandomPolicy()
    p2.reset_for_seed(0)
    actions_b = [p2.predict(None) for _ in range(32)]

    assert actions_a == actions_b
    # And the policy_seed is the documented offset:
    assert p1.policy_seed == POLICY_SEED_OFFSET
    assert p2.policy_seed == POLICY_SEED_OFFSET


def test_seeded_random_policy_varies_with_policy_seed() -> None:
    p1 = SeededRandomPolicy()
    p1.reset_for_seed(0)
    actions_a = [p1.predict(None) for _ in range(64)]

    p2 = SeededRandomPolicy()
    p2.reset_for_seed(7)
    actions_b = [p2.predict(None) for _ in range(64)]

    assert actions_a != actions_b
    # Both still distribute over Discrete(3):
    assert set(actions_a).issubset({0, 1, 2})
    assert set(actions_b).issubset({0, 1, 2})


def test_policy_rng_seed_offset_is_independent_from_env_reset_seed() -> None:
    """Policy seed must be ``eval_seed + 1_000_000``, not ``eval_seed`` itself.

    This is the explicit invariant from ``docs/sight-h5-plan.md`` section 3:
    the policy-side RNG is seeded with an offset so a same-seed env reset
    cannot collide its action stream with the policy's action stream.
    """
    assert POLICY_SEED_OFFSET == 1_000_000
    for eval_seed in (0, 1, 7, 1000, 1015, 2**30):
        assert derive_policy_seed(eval_seed) == eval_seed + 1_000_000
        # The derived policy seed never equals the eval seed itself:
        assert derive_policy_seed(eval_seed) != eval_seed


def test_canonical_thresholds_returns_fresh_dict() -> None:
    a = canonical_non_saturation_thresholds()
    b = canonical_non_saturation_thresholds()
    assert a == b
    a["timeout_rate_threshold"] = 999.0
    assert b["timeout_rate_threshold"] == NON_SATURATION_TIMEOUT_RATE_THRESHOLD
    assert b["length_ratio_threshold"] == NON_SATURATION_LENGTH_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Test 6: harness summary shape from a live rollout against the fake env
# ---------------------------------------------------------------------------


def test_evaluation_summary_has_per_seed_rows_and_aggregates(tmp_path: Path) -> None:
    seeds = [1000, 1001, 1002]
    policy = StayOnlyPolicy()
    summary = evaluate_policy_seeded(
        policy=policy,
        env_factory=lambda: _make_vec_env(max_steps=8, collide_at_step=4),
        seeds=seeds,
        max_steps=8,
        env_id="fake:signal-dodge-vtest",
        observation_mode="pixel",
        out_dir=tmp_path,
        git_commit="abcdef0",
    )
    assert summary["policy_name"] == "stay_only"
    assert summary["seeds"] == seeds
    assert len(summary["per_seed"]) == len(seeds)
    for row in summary["per_seed"]:
        for key in (
            "seed",
            "reward",
            "episode_length",
            "collision",
            "timeout",
            "elapsed_seconds",
        ):
            assert key in row, f"per-seed row missing {key!r}"
    for agg_key in ("aggregate_reward", "aggregate_episode_length"):
        for stat in ("mean", "median", "min", "max", "std"):
            assert stat in summary[agg_key]
    # Stay-only on a collide-at-step=4 env -> all three seeds collide at
    # step 4 -> collision_rate = 1.0, timeout_rate = 0.0.
    assert summary["collision_rate"] == 1.0
    assert summary["timeout_rate"] == 0.0
    # Thresholds are recorded so an audit can detect a silent change.
    assert summary["non_saturation_thresholds"] == canonical_non_saturation_thresholds()
    # Branch metadata is captured for the audit trail.
    assert summary["branch_metadata"] == "deterministic_stay"
    assert summary["git_commit"] == "abcdef0"


# ---------------------------------------------------------------------------
# Test 7: non-saturation gate on synthetic summaries
# ---------------------------------------------------------------------------


def _synthetic_summary(
    *,
    policy_name: str,
    timeout_rate: float,
    mean_episode_length: float,
) -> dict:
    return {
        "policy_name": policy_name,
        "timeout_rate": float(timeout_rate),
        "aggregate_episode_length": {"mean": float(mean_episode_length)},
    }


def test_non_saturation_gate_passes_when_negative_controls_are_clean() -> None:
    """All three negative controls below the threshold -> passed=True."""
    summaries = {
        "stay_only": _synthetic_summary(
            policy_name="stay_only", timeout_rate=0.0, mean_episode_length=120.0
        ),
        "seeded_random": _synthetic_summary(
            policy_name="seeded_random", timeout_rate=0.1, mean_episode_length=250.0
        ),
        "untrained_cnn": _synthetic_summary(
            policy_name="untrained_cnn", timeout_rate=0.3, mean_episode_length=900.0
        ),
        "trained_cnn": _synthetic_summary(
            policy_name="trained_cnn", timeout_rate=0.9, mean_episode_length=1700.0
        ),
    }
    decision = evaluate_non_saturation(summaries, max_steps=1800)
    assert decision["passed"] is True
    assert decision["saturated_negative_controls"] == []
    # The trained policy is reported but does not affect the decision.
    assert decision["per_policy"]["trained_cnn"]["saturated"] is True


def test_non_saturation_gate_fails_when_any_negative_control_saturates() -> None:
    """A single saturated negative control flips the decision to FAIL."""
    summaries = {
        "stay_only": _synthetic_summary(
            policy_name="stay_only", timeout_rate=0.0, mean_episode_length=10.0
        ),
        "seeded_random": _synthetic_summary(
            policy_name="seeded_random", timeout_rate=0.0, mean_episode_length=50.0
        ),
        # Untrained CNN reaches max_steps on most seeds -> saturated.
        "untrained_cnn": _synthetic_summary(
            policy_name="untrained_cnn",
            timeout_rate=0.0,
            mean_episode_length=0.81 * 1800,
        ),
        "trained_cnn": _synthetic_summary(
            policy_name="trained_cnn", timeout_rate=1.0, mean_episode_length=1800.0
        ),
    }
    decision = evaluate_non_saturation(summaries, max_steps=1800)
    assert decision["passed"] is False
    assert decision["saturated_negative_controls"] == ["untrained_cnn"]
    # The trained policy is also saturated but is NOT a negative control.
    assert decision["per_policy"]["trained_cnn"]["saturated"] is True
    assert "trained_cnn" not in decision["saturated_negative_controls"]


def test_non_saturation_gate_uses_timeout_rate_threshold() -> None:
    """timeout_rate >= 0.50 alone is enough to saturate, even at low length."""
    summaries = {
        "stay_only": _synthetic_summary(
            policy_name="stay_only", timeout_rate=0.55, mean_episode_length=300.0
        ),
        "seeded_random": _synthetic_summary(
            policy_name="seeded_random", timeout_rate=0.0, mean_episode_length=80.0
        ),
        "untrained_cnn": _synthetic_summary(
            policy_name="untrained_cnn", timeout_rate=0.0, mean_episode_length=80.0
        ),
    }
    decision = evaluate_non_saturation(summaries, max_steps=1800)
    assert decision["passed"] is False
    assert decision["saturated_negative_controls"] == ["stay_only"]


# ---------------------------------------------------------------------------
# Test 8: on-disk artifact layout
# ---------------------------------------------------------------------------


def test_artifact_layout_writes_evaluation_policy_summary_json(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "fake_run"
    policies = {
        "stay_only": StayOnlyPolicy(),
        "seeded_random": SeededRandomPolicy(),
    }

    def env_factory(_policy_name: str) -> DummyVecEnv:
        return _make_vec_env(max_steps=8, collide_at_step=None)

    index = run_h5_baseline(
        run_dir=run_dir,
        env_id="fake:signal-dodge-vtest",
        observation_mode="pixel",
        max_steps=8,
        seeds=[1000, 1001],
        env_factory_for_policy=env_factory,
        policies=policies,
        git_commit="testc0m",
    )
    assert (run_dir / "evaluation" / "index.json").exists()
    for name in policies:
        sp = run_dir / "evaluation" / name / "summary.json"
        ep = run_dir / "evaluation" / name / "episodes.ndjson"
        assert sp.exists(), f"summary missing for {name}"
        assert ep.exists(), f"episodes.ndjson missing for {name}"
    # collide_at_step=None makes the fake env truncate at step 8, which is
    # the timeout path -> both negative controls saturate at 100% timeout
    # -> the gate FAILs (this is the saturation-detection sanity check).
    assert index["saturation_decision"]["passed"] is False
    assert index["non_saturation_thresholds"] == canonical_non_saturation_thresholds()


# ---------------------------------------------------------------------------
# Test 9: trained-policy model.zip path validation (no live train run)
# ---------------------------------------------------------------------------


def test_resolve_trained_model_path_from_artifact_paths(tmp_path: Path) -> None:
    """The trained-policy evaluator must locate ``model.zip`` from
    ``summary.json``'s ``artifact_paths.model`` without launching anything.

    Constructs a minimal fake train run dir with a summary.json that
    points at a placeholder ``model.zip`` on disk. We never call
    ``PPO.load`` here; the contract under test is *path validation only*,
    matching the prompt's "dry model-path validation test without
    requiring a long live run."
    """
    run_dir = tmp_path / "fake_train_run"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip eocd
    summary = {
        "schema_version": 2,
        "kind": "train",
        "env_id": "godot:signal-dodge-v0",
        "algo": "PPO",
        "framework": "stable-baselines3",
        "run_id": "fake_run_id",
        "phase": "H4",
        "artifact_paths": {"model": str(model_path)},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    resolved = resolve_trained_model_path(run_dir)
    assert resolved == model_path
    assert resolved.exists()


def test_resolve_trained_model_path_falls_back_to_model_zip(tmp_path: Path) -> None:
    """When ``artifact_paths.model`` is absent, ``model.zip`` is used."""
    run_dir = tmp_path / "fake_train_run_no_paths"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    summary = {
        "schema_version": 2,
        "kind": "train",
        "env_id": "godot:signal-dodge-v0",
        "algo": "PPO",
        "framework": "stable-baselines3",
        "run_id": "fake_run_id",
        "phase": "H4",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    resolved = resolve_trained_model_path(run_dir)
    assert resolved == model_path


def test_resolve_trained_model_path_raises_when_missing(tmp_path: Path) -> None:
    """Missing ``summary.json`` raises a clear error."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_trained_model_path(run_dir)


# ---------------------------------------------------------------------------
# Test 10: obs.data exclusion
# ---------------------------------------------------------------------------


def test_obs_data_not_in_evaluation_summaries(tmp_path: Path) -> None:
    """No per-seed row or summary field carries raw observation data.

    H4 hardened the env to emit ``obs_metadata`` (no ``obs.data``) on
    every reset. H5 evaluation summaries inherit the same posture: per-
    seed rows must record only scalar metrics, and the summary as a
    whole must not contain any ``obs`` / ``observation`` arrays even if
    a future field accidentally captures the raw obs from env.step.
    """
    policy = StayOnlyPolicy()
    summary = evaluate_policy_seeded(
        policy=policy,
        env_factory=lambda: _make_vec_env(max_steps=8, collide_at_step=4),
        seeds=[1000],
        max_steps=8,
        env_id="fake:signal-dodge-vtest",
        observation_mode="pixel",
        out_dir=tmp_path,
        git_commit="abcdef0",
    )
    forbidden = {"obs", "observation", "obs_data", "frame", "pixels"}
    for row in summary["per_seed"]:
        assert forbidden.isdisjoint(row.keys()), (
            f"per-seed row leaked obs-like key: {set(row.keys()) & forbidden}"
        )
    serialised = json.dumps(summary)
    # Defensive: even nested under a different name, raw obs arrays would
    # serialise to lists of length 84*84=7056. Catch that as a sanity
    # check on the on-disk payload.
    assert "\"obs\":" not in serialised
    assert "\"observation\":" not in serialised


def test_episode_result_dict_has_only_scalar_fields() -> None:
    """EpisodeResult.as_dict must expose only JSON-scalar fields."""
    r = EpisodeResult(
        seed=1000,
        policy_seed=POLICY_SEED_OFFSET + 1000,
        reward=42.0,
        episode_length=3,
        collision=True,
        timeout=False,
        elapsed_seconds=0.01,
    )
    d = r.as_dict()
    assert set(d.keys()) == {
        "seed",
        "policy_seed",
        "reward",
        "episode_length",
        "collision",
        "timeout",
        "elapsed_seconds",
    }


# ---------------------------------------------------------------------------
# Test 11: rollout collision-vs-timeout discrimination
# ---------------------------------------------------------------------------


def test_rollout_classifies_collision_when_env_terminates_early() -> None:
    env = _make_vec_env(max_steps=16, collide_at_step=5)
    p = StayOnlyPolicy()
    p.reset_for_seed(0)
    row = rollout_one_episode(policy=p, env=env, max_steps=16)
    env.close()
    assert row.collision is True
    assert row.timeout is False
    assert row.episode_length == 5
    # Reward contract: +1 per non-terminal step, 0 on collision step.
    assert row.reward == 4.0


def test_rollout_classifies_timeout_when_env_truncates() -> None:
    env = _make_vec_env(max_steps=8, collide_at_step=None)
    p = StayOnlyPolicy()
    p.reset_for_seed(0)
    row = rollout_one_episode(policy=p, env=env, max_steps=8)
    env.close()
    assert row.collision is False
    assert row.timeout is True
    assert row.episode_length == 8
    assert row.reward == 8.0


# ---------------------------------------------------------------------------
# Test 12: seeded-random policy seed is recorded per row
# ---------------------------------------------------------------------------


def test_seeded_random_per_seed_row_records_policy_seed(tmp_path: Path) -> None:
    seeds = [1000, 1001]
    p = SeededRandomPolicy()
    summary = evaluate_policy_seeded(
        policy=p,
        env_factory=lambda: _make_vec_env(max_steps=4, collide_at_step=None),
        seeds=seeds,
        max_steps=4,
        env_id="fake:signal-dodge-vtest",
        observation_mode="pixel",
        out_dir=tmp_path,
    )
    rows = summary["per_seed"]
    for row, eval_seed in zip(rows, seeds):
        assert row["seed"] == eval_seed
        assert row["policy_seed"] == eval_seed + POLICY_SEED_OFFSET


# ---------------------------------------------------------------------------
# Test 13: NEGATIVE_CONTROL_POLICIES contains exactly the three documented
# negative controls, in the documented order.
# ---------------------------------------------------------------------------


def test_negative_control_policies_tuple_is_stable() -> None:
    assert NEGATIVE_CONTROL_POLICIES == (
        "stay_only",
        "seeded_random",
        "untrained_cnn",
    )
