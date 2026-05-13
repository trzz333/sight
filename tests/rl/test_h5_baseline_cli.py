"""H5 baseline CLI unit tests.

Default-tier (no live Godot). Covers:
- seed parsing for both comma list and inclusive range forms (and mixes)
- mode->policies resolution defaults and explicit overrides
- negative-controls mode rejects trained_cnn with a clear error
- end-to-end CLI run with monkeypatched make_env / dummy untrained_cnn
  writes evaluation/index.json and per-policy summary.json
- handoff wording no longer implies trained_cnn participates in the
  pre-training non-saturation gate
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sight_agent.rl import h5_baseline_cli as cli_mod
from sight_agent.rl.h5_baseline_cli import (
    H5CLIError,
    VALID_POLICY_NAMES,
    build_arg_parser,
    parse_seeds,
    resolve_policies_for_mode,
    resolve_run_dir,
    run_cli,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_PATH = REPO_ROOT / "docs" / "sight-handoff.md"
H5_PLAN_PATH = REPO_ROOT / "docs" / "sight-h5-plan.md"


# ---------------------------------------------------------------------------
# Test 1: parse_seeds covers comma list, range, and mixed forms.
# ---------------------------------------------------------------------------


def test_parse_seeds_comma_list() -> None:
    assert parse_seeds("1000,1001") == [1000, 1001]


def test_parse_seeds_range_inclusive() -> None:
    assert parse_seeds("1000-1015") == list(range(1000, 1016))
    assert len(parse_seeds("1000-1015")) == 16


def test_parse_seeds_mixed_comma_and_range() -> None:
    assert parse_seeds("1000,1003-1005") == [1000, 1003, 1004, 1005]


def test_parse_seeds_single_value() -> None:
    assert parse_seeds("42") == [42]


def test_parse_seeds_preserves_order_and_duplicates() -> None:
    # Order is preserved as-given; duplicates are kept.
    assert parse_seeds("1001,1000,1001") == [1001, 1000, 1001]


def test_parse_seeds_rejects_empty_string() -> None:
    with pytest.raises(H5CLIError):
        parse_seeds("")


def test_parse_seeds_rejects_empty_token() -> None:
    with pytest.raises(H5CLIError):
        parse_seeds("1000,,1001")


def test_parse_seeds_rejects_malformed_range() -> None:
    for bad in ("1000-", "-1000", "1000-1001-1002"):
        with pytest.raises(H5CLIError):
            parse_seeds(bad)


def test_parse_seeds_rejects_non_integer() -> None:
    with pytest.raises(H5CLIError):
        parse_seeds("1000,abc")


def test_parse_seeds_rejects_hi_less_than_lo() -> None:
    with pytest.raises(H5CLIError):
        parse_seeds("1005-1000")


# ---------------------------------------------------------------------------
# Test 2-3: mode -> policies resolution.
# ---------------------------------------------------------------------------


def test_negative_controls_default_includes_exactly_three_policies() -> None:
    out = resolve_policies_for_mode("negative-controls", None)
    assert out == ["stay_only", "seeded_random", "untrained_cnn"]
    assert "trained_cnn" not in out


def test_full_mode_default_includes_trained_cnn() -> None:
    out = resolve_policies_for_mode("full", None)
    assert out == ["stay_only", "seeded_random", "untrained_cnn", "trained_cnn"]


def test_negative_controls_rejects_trained_cnn_explicit_request() -> None:
    with pytest.raises(H5CLIError) as excinfo:
        resolve_policies_for_mode(
            "negative-controls",
            ["stay_only", "trained_cnn"],
        )
    assert "trained_cnn" in str(excinfo.value)
    assert "negative-controls" in str(excinfo.value)


def test_negative_controls_accepts_subset_of_three() -> None:
    # A single negative control is permitted; the pre-training gate
    # decision still uses only the negative controls that are present.
    out = resolve_policies_for_mode("negative-controls", ["stay_only"])
    assert out == ["stay_only"]


def test_full_mode_accepts_trained_cnn() -> None:
    out = resolve_policies_for_mode(
        "full", ["stay_only", "trained_cnn"]
    )
    assert out == ["stay_only", "trained_cnn"]


def test_unknown_policy_name_rejected() -> None:
    with pytest.raises(H5CLIError):
        resolve_policies_for_mode("full", ["mystery_policy"])


def test_unknown_mode_rejected() -> None:
    with pytest.raises(H5CLIError):
        resolve_policies_for_mode("yolo", None)


def test_valid_policy_names_tuple_is_stable() -> None:
    assert VALID_POLICY_NAMES == (
        "stay_only",
        "seeded_random",
        "untrained_cnn",
        "trained_cnn",
    )


# ---------------------------------------------------------------------------
# Test 4: arg parser surface.
# ---------------------------------------------------------------------------


def test_arg_parser_minimal_required_args() -> None:
    parser = build_arg_parser()
    ns = parser.parse_args([
        "--config", "configs/rl/x.yaml",
        "--run-id", "my_run",
        "--seeds", "1000,1001",
    ])
    assert ns.config == "configs/rl/x.yaml"
    assert ns.run_id == "my_run"
    assert ns.seeds == "1000,1001"
    assert ns.mode == "negative-controls"
    assert ns.policies is None
    assert ns.train_run_dir is None
    assert ns.out_dir is None


def test_arg_parser_rejects_unknown_mode() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--config", "x.yaml",
            "--run-id", "r",
            "--seeds", "1000",
            "--mode", "yolo",
        ])


# ---------------------------------------------------------------------------
# Test 5: end-to-end CLI run writes evaluation/index.json + per-policy
# summaries, with make_env monkeypatched to a fake VecEnv (no Godot).
# ---------------------------------------------------------------------------


class _FakeSignalDodgeForCLI(gym.Env):
    """Pixel-shape fake env mirroring the H4 contract for CLI smoke tests."""

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 1800,
        collide_at_step: int | None = 5,
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

    def _obs(self) -> np.ndarray:
        return np.zeros((1, 84, 84), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):  # type: ignore[override]
        self._step_count += 1
        terminated = (
            self.collide_at_step is not None
            and self._step_count >= self.collide_at_step
        )
        truncated = (not terminated) and self._step_count >= self.max_steps
        reward = 0.0 if terminated else 1.0
        return self._obs(), reward, bool(terminated), bool(truncated), {}


def _fake_make_env(*_args, **_kwargs):
    return DummyVecEnv(
        [lambda: _FakeSignalDodgeForCLI(max_steps=1800, collide_at_step=5)]
    )


def _write_min_pixel_yaml(path: Path, out_dir: Path) -> None:
    """Write a minimal pixel YAML matching the H4 contract shape."""
    import yaml

    cfg = {
        "run": {
            "phase": "H5",
            "name": "test_h5_cli",
            "seed": 0,
            "out_dir": str(out_dir),
        },
        "env": {
            "id": "godot:signal-dodge-v0",
            "n_envs": 1,
            "godot_executable": None,
            "project_path": "games/signal-dodge",
            "max_steps": 1800,
            "headless": False,
            "observation_mode": "pixel",
            "pixel_width": 84,
            "pixel_height": 84,
            "pixel_channels": 1,
        },
        "algo": {
            "framework": "stable-baselines3",
            "name": "PPO",
            "policy": "CnnPolicy",
            "device": "cpu",
            "hyperparams": {
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 1,
            },
        },
        "train": {"total_timesteps": 128},
        "eval": {"eval_freq": 64, "n_eval_episodes": 1, "deterministic": True},
        "logging": {"format": "ndjson"},
        "checkpoint": {"enabled": True, "filename": "model.zip"},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_cli_writes_evaluation_index_and_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI smoke against a fake VecEnv: no Godot, no live model.

    monkeypatches:
      - h5_baseline_cli.make_env             -> fake VecEnv
      - h5_baseline_cli.build_dummy_vec_env_for_cfg -> fake VecEnv with
        matching spaces (so PPO CnnPolicy construction works without a
        real Godot env)

    Then runs the CLI's run_cli() function in negative-controls mode
    with two seeds and asserts the on-disk artifact layout matches the
    documented shape and that the run-level index records the canonical
    thresholds.
    """
    out_root = tmp_path / "runs"
    config_path = tmp_path / "min_h5.yaml"
    _write_min_pixel_yaml(config_path, out_root)

    monkeypatch.setattr(cli_mod, "make_env", _fake_make_env)
    monkeypatch.setattr(
        cli_mod, "build_dummy_vec_env_for_cfg",
        lambda _cfg: _fake_make_env(),
    )

    index = run_cli(
        config_path=str(config_path),
        run_id="h5_cli_test_run",
        seeds_spec="1000,1001",
        mode="negative-controls",
        requested_policies=None,
        train_run_dir=None,
        out_dir_override=None,
    )

    run_dir = out_root / "test_h5_cli" / "h5_cli_test_run"
    eval_dir = run_dir / "evaluation"
    assert (eval_dir / "index.json").exists(), "index.json missing on disk"
    for name in ("stay_only", "seeded_random", "untrained_cnn"):
        sp = eval_dir / name / "summary.json"
        ep = eval_dir / name / "episodes.ndjson"
        assert sp.exists(), f"summary missing for {name}"
        assert ep.exists(), f"episodes.ndjson missing for {name}"
    # No trained_cnn in negative-controls mode.
    assert not (eval_dir / "trained_cnn").exists()

    # Index shape: canonical thresholds, seed list, saturation decision.
    payload = json.loads((eval_dir / "index.json").read_text(encoding="utf-8"))
    assert payload["env_id"] == "godot:signal-dodge-v0"
    assert payload["observation_mode"] == "pixel"
    assert payload["max_steps"] == 1800
    assert payload["seeds"] == [1000, 1001]
    assert payload["policies"] == [
        "seeded_random", "stay_only", "untrained_cnn",
    ]
    assert payload["non_saturation_thresholds"] == {
        "timeout_rate_threshold": 0.50,
        "length_ratio_threshold": 0.80,
    }
    decision = payload["saturation_decision"]
    # Fake env collides at step 5: short episode -> no saturation.
    assert decision["passed"] is True
    assert decision["saturated_negative_controls"] == []

    # Return value matches on-disk payload.
    assert index["seeds"] == [1000, 1001]


def test_cli_run_cli_rejects_trained_cnn_in_negative_controls_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_cli surfaces H5CLIError when trained_cnn appears under
    --mode negative-controls. No env construction happens before the
    validation check fires.
    """
    out_root = tmp_path / "runs"
    config_path = tmp_path / "min_h5.yaml"
    _write_min_pixel_yaml(config_path, out_root)

    # Even though make_env is monkeypatched, the validation should fail
    # before any factory call. This also confirms the order of checks.
    monkeypatch.setattr(cli_mod, "make_env", _fake_make_env)
    monkeypatch.setattr(
        cli_mod, "build_dummy_vec_env_for_cfg",
        lambda _cfg: _fake_make_env(),
    )
    with pytest.raises(H5CLIError) as excinfo:
        run_cli(
            config_path=str(config_path),
            run_id="x",
            seeds_spec="1000",
            mode="negative-controls",
            requested_policies=["stay_only", "trained_cnn"],
            train_run_dir=None,
            out_dir_override=None,
        )
    assert "trained_cnn" in str(excinfo.value)


def test_resolve_run_dir_uses_yaml_out_dir(tmp_path: Path) -> None:
    cfg = {
        "run": {
            "name": "test_run_name",
            "out_dir": str(tmp_path / "runs"),
        }
    }
    out = resolve_run_dir(cfg, "abc123", None)
    assert out == tmp_path / "runs" / "test_run_name" / "abc123"


def test_resolve_run_dir_override_takes_precedence(tmp_path: Path) -> None:
    cfg = {
        "run": {
            "name": "test_run_name",
            "out_dir": "runs/wrong",
        }
    }
    out = resolve_run_dir(cfg, "abc123", str(tmp_path / "override"))
    assert out == tmp_path / "override" / "test_run_name" / "abc123"


def test_resolve_run_dir_requires_name() -> None:
    with pytest.raises(H5CLIError):
        resolve_run_dir({"run": {"out_dir": "runs"}}, "x", None)


def test_resolve_run_dir_requires_out_dir() -> None:
    with pytest.raises(H5CLIError):
        resolve_run_dir({"run": {"name": "x"}}, "y", None)


# ---------------------------------------------------------------------------
# Test 6: handoff and h5-plan wording precision.
# The pre-training non-saturation gate uses only three negative controls;
# trained_cnn is deferred until after a training slice produces model.zip.
# ---------------------------------------------------------------------------


def test_handoff_does_not_describe_pre_training_gate_as_four_policy() -> None:
    """Catches the imprecise 'four-policy baseline' wording on the
    pre-training non-saturation gate. The H5 acceptance suite eventually
    evaluates four policies; the pre-training gate is three-negative-
    control only.
    """
    text = HANDOFF_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    # The exact imprecise phrase from the prior handoff iteration is
    # banned. It must not return.
    assert "four-policy baseline" not in lower, (
        "handoff says 'four-policy baseline' for the pre-training non-"
        "saturation gate; the gate is three negative controls only "
        "(stay_only, seeded_random, untrained_cnn). trained_cnn does not "
        "exist until after a training slice produces model.zip."
    )
    assert "four-policy non-saturation" not in lower
    assert "non-saturation" in lower or "non_saturation" in lower


def test_h5_plan_distinguishes_pre_training_gate_from_acceptance_suite() -> None:
    """The H5 plan must still call out that the non-saturation gate
    operates on the three negative controls; the four-policy acceptance
    suite is a different, later contract.
    """
    text = H5_PLAN_PATH.read_text(encoding="utf-8")
    assert "three negative controls" in text.lower() or (
        "stay-only" in text.lower()
        and "seeded random" in text.lower()
        and "untrained" in text.lower()
    ), "H5 plan section 5 must name the three negative controls explicitly"
    # The plan IS allowed to talk about four-policy acceptance generally
    # (section 3), but the non-saturation gate must not be framed that way.
    plan_lower = text.lower()
    assert "non-saturation gate" in plan_lower
