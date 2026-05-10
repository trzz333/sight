"""H4 Step 1 unit tests for GodotSignalDodgeEnv observation_mode plumbing.

Pure construction tests. No subprocess started, no TCP connection opened,
no live Godot binary required. Validates the H4 plan section 1 contract:

- observation_mode in {"state", "pixel", "both"} with default "state"
- pixel_width / pixel_height / pixel_channels constructor args
- ValueError on invalid observation_mode
- ValueError on non-positive pixel dims
- ValueError when observation_mode in {"pixel", "both"} and headless=True
  (per Grok H3 closure caveat: caller intent honored or rejected, not
  silently transformed; the H4 spike blocks --headless pixel capture)
- observation_space dispatch: state -> Box(-1,1,(10,),float32),
  pixel -> Box(0,255,(1,84,84),uint8),
  both -> Dict({"state": ..., "pixel": ...})

Run:
    pytest tests/rl/test_h4_godot_env_construct.py -v --tb=short
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from sight_agent.rl.godot_env import GodotSignalDodgeEnv


# Dummy paths suffice because construction never touches the filesystem.
_GODOT_EXE = "/nonexistent/godot.exe"
_PROJECT = "/nonexistent/project"


def _make(**overrides):
    """Construct the env with safe defaults; overrides apply on top."""
    kwargs = dict(
        godot_executable=_GODOT_EXE,
        project_path=_PROJECT,
        tcp_port=0,
    )
    kwargs.update(overrides)
    return GodotSignalDodgeEnv(**kwargs)


# --- default mode (H3 invariant) -----------------------------------------


def test_default_observation_mode_is_state():
    """H3 state behavior must be byte-for-byte default. No H4 surface
    leaks into callers that did not opt in."""
    env = _make()
    assert env._observation_mode == "state"
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert float(env.observation_space.low.min()) == -1.0
    assert float(env.observation_space.high.max()) == 1.0


def test_default_action_space_is_discrete_3():
    env = _make()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3


# --- mode validation ------------------------------------------------------


@pytest.mark.parametrize("mode", ["state", "pixel", "both"])
def test_observation_mode_accepts_valid_values(mode):
    """All three documented modes construct without raising. Pixel/both
    require windowed launch; tests pass headless=False for those."""
    headless = mode == "state"
    env = _make(observation_mode=mode, headless=headless)
    assert env._observation_mode == mode


@pytest.mark.parametrize(
    "bad_mode",
    ["", "STATE", "pixels", "rgb", "image", None, 0, 1, True, ["state"]],
)
def test_observation_mode_rejects_invalid_values(bad_mode):
    with pytest.raises(ValueError, match="observation_mode"):
        _make(observation_mode=bad_mode)


# --- pixel dimension validation ------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["pixel_width", "pixel_height", "pixel_channels"],
)
@pytest.mark.parametrize("bad_value", [0, -1, -84, 1.5, "84", None, True, False])
def test_pixel_dims_reject_non_positive_int(field, bad_value):
    with pytest.raises(ValueError, match=field):
        _make(**{field: bad_value}, observation_mode="pixel", headless=False)


# --- headless rejection (Grok caveat) ------------------------------------


def test_pixel_mode_rejects_headless_true():
    """Per docs/sight-h4-plan.md section 1 and the Grok H3 closure caveat,
    pixel mode must raise when headless=True. The env does not auto-flip
    headless. The H4 spike (docs/sight-h4-spike.md) proved Godot 4.6.2
    --headless cannot emit RenderingServer.frame_post_draw."""
    with pytest.raises(ValueError, match="headless"):
        _make(observation_mode="pixel", headless=True)


def test_both_mode_rejects_headless_true():
    with pytest.raises(ValueError, match="headless"):
        _make(observation_mode="both", headless=True)


def test_pixel_mode_accepts_headless_false():
    env = _make(observation_mode="pixel", headless=False)
    assert env._observation_mode == "pixel"
    assert env._headless is False


def test_both_mode_accepts_headless_false():
    env = _make(observation_mode="both", headless=False)
    assert env._observation_mode == "both"
    assert env._headless is False


def test_state_mode_accepts_headless_true():
    """State mode must NOT regress: H3 default of headless=True still
    works for state observations."""
    env = _make(observation_mode="state", headless=True)
    assert env._headless is True


# --- observation_space dispatch ------------------------------------------


def test_pixel_observation_space_default_shape():
    """Pixel default is Box(0,255,(1,84,84),uint8) per H4 plan section 2
    and Decision 3."""
    env = _make(observation_mode="pixel", headless=False)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (1, 84, 84)
    assert space.dtype == np.uint8
    assert int(space.low.min()) == 0
    assert int(space.high.max()) == 255


def test_pixel_observation_space_honors_constructor_dims():
    """Custom pixel dims propagate. Order is (channels, height, width)
    to match SB3 NatureCNN's channel-first convention."""
    env = _make(
        observation_mode="pixel",
        headless=False,
        pixel_width=64,
        pixel_height=64,
        pixel_channels=3,
    )
    assert env.observation_space.shape == (3, 64, 64)
    assert env.observation_space.dtype == np.uint8


def test_both_observation_space_is_dict():
    """Both mode wraps state and pixel spaces in a Dict so callers can
    consume either stream."""
    env = _make(observation_mode="both", headless=False)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Dict)
    assert set(space.spaces.keys()) == {"state", "pixel"}
    state_space = space.spaces["state"]
    pixel_space = space.spaces["pixel"]
    assert isinstance(state_space, gym.spaces.Box)
    assert state_space.shape == (10,)
    assert state_space.dtype == np.float32
    assert isinstance(pixel_space, gym.spaces.Box)
    assert pixel_space.shape == (1, 84, 84)
    assert pixel_space.dtype == np.uint8


def test_state_observation_space_unchanged_from_h3():
    """Constructing with explicit observation_mode='state' yields the
    same observation_space shape/dtype/bounds as the H3 default."""
    default_env = _make()
    explicit_env = _make(observation_mode="state")
    for env in (default_env, explicit_env):
        space = env.observation_space
        assert isinstance(space, gym.spaces.Box)
        assert space.shape == (10,)
        assert space.dtype == np.float32
        assert float(space.low.min()) == -1.0
        assert float(space.high.max()) == 1.0
