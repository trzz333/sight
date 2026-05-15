"""H5 Phase F: frame-stack wrapper tests.

Covers the observation-contract change introduced by Phase F:

- ``env.frame_stack`` propagates from YAML through
  ``resolve_godot_kwargs`` into the factory.
- ``make_env`` wraps pixel-mode Godot VecEnvs with
  ``VecFrameStack(n_stack, channels_order="first")`` when frame_stack
  is >= 2, and leaves them untouched otherwise (frame_stack=None,
  frame_stack=1).
- ``make_env`` does NOT wrap state-mode envs even when frame_stack
  is set (defensive: frame stacking on a 1-d state Box would produce
  nonsense and is not the H5 plan).
- ``build_dummy_vec_env_for_cfg`` mirrors the factory: produces
  ``(C, H, W)`` without frame stacking and
  ``(C * n_stack, H, W)`` with frame stacking, so the untrained_cnn
  baseline policy is constructed against the same observation shape
  the trained policy will see at eval time.
- The new Phase F YAML
  ``configs/rl/signal_dodge_ppo_h5_pixel_frame_stack.yaml`` loads
  cleanly and exposes ``frame_stack=4`` through the resolver.

These tests do not launch Godot; the env constructor is lazy on
the subprocess side and ``VecFrameStack`` only reads
``observation_space`` at construction time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from sight_agent.rl.config import load_config
from sight_agent.rl.factories import make_env
from sight_agent.rl.godot_config import resolve_godot_kwargs
from sight_agent.rl.h5_baseline_cli import build_dummy_vec_env_for_cfg


_FAKE_GODOT_EXE = r"C:\fake\godot.exe"
_FAKE_PROJECT = r"C:\fake\project"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE_F_CONFIG = _REPO_ROOT / "configs" / "rl" / "signal_dodge_ppo_h5_pixel_frame_stack.yaml"
_PHASE_D_CONFIG = _REPO_ROOT / "configs" / "rl" / "signal_dodge_ppo_h5_pixel_entropy.yaml"


def _clear_godot_env(monkeypatch) -> None:
    monkeypatch.delenv("SIGHT_GODOT_EXE", raising=False)
    monkeypatch.delenv("SIGHT_GODOT_PROJECT", raising=False)


def _godot_pixel_kwargs(**overrides):
    base = {
        "godot_executable": _FAKE_GODOT_EXE,
        "project_path": _FAKE_PROJECT,
        "max_steps": 1800,
        "headless": False,
        "observation_mode": "pixel",
        "pixel_width": 84,
        "pixel_height": 84,
        "pixel_channels": 1,
    }
    base.update(overrides)
    return base


# --- make_env wrapping tests ---


def test_make_env_pixel_without_frame_stack_kwarg_keeps_single_frame_shape(monkeypatch) -> None:
    _clear_godot_env(monkeypatch)
    env = make_env(
        "godot:signal-dodge-v0", n_envs=1, seed=0, mode="train",
        **_godot_pixel_kwargs(),
    )
    try:
        assert isinstance(env, VecEnv)
        assert not isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (1, 84, 84)
    finally:
        env.close()


def test_make_env_pixel_with_frame_stack_one_is_a_noop(monkeypatch) -> None:
    _clear_godot_env(monkeypatch)
    env = make_env(
        "godot:signal-dodge-v0", n_envs=1, seed=0, mode="train",
        frame_stack=1, **_godot_pixel_kwargs(),
    )
    try:
        assert isinstance(env, VecEnv)
        assert not isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (1, 84, 84)
    finally:
        env.close()


def test_make_env_pixel_with_frame_stack_four_wraps_to_stacked_shape(monkeypatch) -> None:
    _clear_godot_env(monkeypatch)
    env = make_env(
        "godot:signal-dodge-v0", n_envs=1, seed=0, mode="train",
        frame_stack=4, **_godot_pixel_kwargs(),
    )
    try:
        assert isinstance(env, VecFrameStack)
        # Channel-first stacking expands the channel dimension to
        # n_stack * pixel_channels. Height and width are preserved.
        assert tuple(env.observation_space.shape) == (4, 84, 84)
    finally:
        env.close()


def test_make_env_pixel_eval_mode_also_wraps_with_frame_stack(monkeypatch) -> None:
    """Train/eval parity: the eval env must expose the same stacked shape."""
    _clear_godot_env(monkeypatch)
    env = make_env(
        "godot:signal-dodge-v0", n_envs=1, seed=0, mode="eval",
        frame_stack=4, **_godot_pixel_kwargs(),
    )
    try:
        assert isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (4, 84, 84)
    finally:
        env.close()


def test_make_env_state_mode_is_never_frame_stacked(monkeypatch) -> None:
    """State-mode envs must not be wrapped even when frame_stack is set.

    Frame stacking on a 1-d Box(10,) state vector would produce a
    nonsense (4,10) or similar shape and silently break the state
    policy. The factory must gate strictly on ``observation_mode ==
    "pixel"``.
    """
    _clear_godot_env(monkeypatch)
    state_kwargs = _godot_pixel_kwargs(observation_mode="state")
    # State mode does not consume pixel dims; pass them anyway to mimic a
    # YAML that simply set observation_mode=state alongside frame_stack
    # (defensive: the gate is on observation_mode, not the presence of
    # pixel dim keys).
    env = make_env(
        "godot:signal-dodge-v0", n_envs=1, seed=0, mode="train",
        frame_stack=4, **state_kwargs,
    )
    try:
        assert not isinstance(env, VecFrameStack)
        # H3 state contract: Box(-1, 1, (10,), float32)
        assert tuple(env.observation_space.shape) == (10,)
    finally:
        env.close()


# --- build_dummy_vec_env_for_cfg parity tests ---


def test_dummy_vec_env_without_frame_stack_keeps_single_frame_shape() -> None:
    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
        }
    }
    env = build_dummy_vec_env_for_cfg(cfg)
    try:
        assert not isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (1, 84, 84)
    finally:
        env.close()


def test_dummy_vec_env_with_frame_stack_four_matches_eval_shape() -> None:
    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
            "frame_stack": 4,
        }
    }
    env = build_dummy_vec_env_for_cfg(cfg)
    try:
        assert isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (4, 84, 84)
    finally:
        env.close()


def test_dummy_vec_env_with_frame_stack_one_is_a_noop() -> None:
    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
            "frame_stack": 1,
        }
    }
    env = build_dummy_vec_env_for_cfg(cfg)
    try:
        assert not isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (1, 84, 84)
    finally:
        env.close()


# --- resolver + config tests ---


def test_resolve_godot_kwargs_forwards_frame_stack_when_present() -> None:
    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "observation_mode": "pixel",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
            "frame_stack": 4,
        }
    }
    extra = resolve_godot_kwargs(cfg)
    assert extra.get("frame_stack") == 4
    # Other optional passthroughs remain working
    assert extra.get("observation_mode") == "pixel"
    assert extra.get("pixel_channels") == 1


def test_resolve_godot_kwargs_omits_frame_stack_when_absent() -> None:
    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "observation_mode": "pixel",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
        }
    }
    extra = resolve_godot_kwargs(cfg)
    assert "frame_stack" not in extra


def test_phase_f_yaml_loads_and_declares_frame_stack_four() -> None:
    assert _PHASE_F_CONFIG.exists(), f"missing {_PHASE_F_CONFIG}"
    cfg = load_config(str(_PHASE_F_CONFIG))
    assert cfg["env"]["observation_mode"] == "pixel"
    assert cfg["env"]["frame_stack"] == 4
    extra = resolve_godot_kwargs(cfg)
    assert extra["frame_stack"] == 4


def test_phase_d_entropy_yaml_remains_single_frame() -> None:
    """Phase D entropy config must not silently acquire frame_stack.

    Phase E was run against this exact config; if a future edit adds
    ``frame_stack`` here, the published Phase E aggregates would no
    longer be reproducible. This test pins the Phase D contract.
    """
    assert _PHASE_D_CONFIG.exists(), f"missing {_PHASE_D_CONFIG}"
    cfg = load_config(str(_PHASE_D_CONFIG))
    assert cfg["env"]["observation_mode"] == "pixel"
    assert "frame_stack" not in cfg["env"]
    extra = resolve_godot_kwargs(cfg)
    assert "frame_stack" not in extra


def test_make_env_pixel_frame_stack_via_phase_f_yaml(monkeypatch) -> None:
    """End-to-end: load the Phase F YAML and let make_env wrap the env.

    Walks the same code path train.py / h5_baseline_cli.py use:
    ``resolve_godot_kwargs`` -> splat into ``make_env``. Confirms the
    YAML -> wrapper bridge works without any Phase-F-only call sites
    in the trainer or eval CLI.
    """
    _clear_godot_env(monkeypatch)
    cfg = load_config(str(_PHASE_F_CONFIG))
    extra = resolve_godot_kwargs(cfg)
    # The YAML leaves godot_executable=null; inject a fake path so the
    # factory does not raise on the env-var fallback.
    extra["godot_executable"] = _FAKE_GODOT_EXE
    env = make_env(
        cfg["env"]["id"],
        n_envs=int(cfg["env"]["n_envs"]),
        seed=int(cfg["run"]["seed"]),
        mode="train",
        **extra,
    )
    try:
        assert isinstance(env, VecFrameStack)
        assert tuple(env.observation_space.shape) == (4, 84, 84)
    finally:
        env.close()

# --- run_start metadata parity tests ---


def test_godot_smoke_obs_metadata_pixel_no_frame_stack_reports_unstacked_shape() -> None:
    """H4 path: pixel mode without frame_stack reports (C, H, W) verbatim."""
    from sight_agent.rl.train import _godot_smoke_obs_metadata

    cfg = {
        "env": {
            "observation_mode": "pixel",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
        }
    }
    obs_shape, action_n = _godot_smoke_obs_metadata(cfg)
    assert obs_shape == (1, 84, 84)
    assert action_n == 3


def test_godot_smoke_obs_metadata_pixel_frame_stack_one_reports_unstacked_shape() -> None:
    from sight_agent.rl.train import _godot_smoke_obs_metadata

    cfg = {
        "env": {
            "observation_mode": "pixel",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
            "frame_stack": 1,
        }
    }
    obs_shape, action_n = _godot_smoke_obs_metadata(cfg)
    assert obs_shape == (1, 84, 84)
    assert action_n == 3


def test_godot_smoke_obs_metadata_pixel_frame_stack_four_reports_stacked_shape() -> None:
    """The run_start event must reflect the policy-facing obs shape.

    Without this, the run_start metadata says (1, 84, 84) while the
    saved model's observation_space is Box(0, 255, (4, 84, 84), uint8),
    creating a false discrepancy that masks real wrapper mismatches.
    """
    from sight_agent.rl.train import _godot_smoke_obs_metadata

    cfg = {
        "env": {
            "observation_mode": "pixel",
            "pixel_channels": 1,
            "pixel_height": 84,
            "pixel_width": 84,
            "frame_stack": 4,
        }
    }
    obs_shape, action_n = _godot_smoke_obs_metadata(cfg)
    assert obs_shape == (4, 84, 84)
    assert action_n == 3


def test_godot_smoke_obs_metadata_state_mode_unaffected_by_frame_stack() -> None:
    """State mode is never frame-stacked; metadata stays (10,)."""
    from sight_agent.rl.train import _godot_smoke_obs_metadata

    cfg = {
        "env": {
            "observation_mode": "state",
            "frame_stack": 4,
        }
    }
    obs_shape, action_n = _godot_smoke_obs_metadata(cfg)
    assert obs_shape == (10,)
    assert action_n == 3
