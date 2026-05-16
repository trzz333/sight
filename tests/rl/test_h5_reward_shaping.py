"""H5 reward-amendment unit tests.

Covers the pure reward-shaping compute function, the env-layer
integration (shaped vs default path, terminal handling, log schema),
constructor validation, and the config / factory plumbing.

Default-tier pytest. No live Godot, no real subprocess. Mirrors the
shared fake-protocol harness in ``tests/rl/h3_godot_fakes.py``.

Run:
    pytest tests/rl/test_h5_reward_shaping.py -v --tb=short
"""

from __future__ import annotations

import json as _json

import numpy as np
import pytest

from sight_agent import protocol
from sight_agent.rl.godot_env import GodotSignalDodgeEnv
from sight_agent.rl.reward_shaping import (
    DEFAULT_ALPHA,
    DEFAULT_LOOKAHEAD_BAND,
    DEFAULT_SAFE_LATERAL_DISTANCE,
    REWARD_SHAPING_NONE,
    REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    VALID_REWARD_SHAPINGS,
    compute_threat_weighted_clearance,
)

from .h3_godot_fakes import (
    FakeProcess,
    FakeProcessFactoryRecorder,
    FakeTransport,
    FakeTransportFactoryRecorder,
    _reset_ok_payload,
    _step_ok_payload,
)


# --- pure function tests --------------------------------------------------


def test_module_constants_match_proposal():
    assert REWARD_SHAPING_NONE == "none"
    assert REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE == "threat_weighted_clearance"
    assert VALID_REWARD_SHAPINGS == frozenset(
        {REWARD_SHAPING_NONE, REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE}
    )
    # Per docs/h5-reward-amendment-proposal.md section 4.
    assert DEFAULT_ALPHA == 0.05
    assert DEFAULT_LOOKAHEAD_BAND == 270.0
    assert DEFAULT_SAFE_LATERAL_DISTANCE == 180.0


def test_none_reward_state_returns_zero_bonus():
    bonus, weight_sum, count = compute_threat_weighted_clearance(None)
    assert bonus == 0.0
    assert weight_sum == 0.0
    assert count == 0


def test_empty_hazards_above_returns_zero_bonus():
    rs = {"player_x": 100.0, "player_y": 500.0, "hazards_above": []}
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == 0.0
    assert weight_sum == 0.0
    assert count == 0


def test_missing_hazards_key_returns_zero_bonus():
    rs = {"player_x": 100.0, "player_y": 500.0}
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == 0.0
    assert weight_sum == 0.0
    assert count == 0


def test_hazards_below_player_excluded():
    # All hazards below player -> bonus 0.0.
    rs = {
        "player_x": 100.0,
        "player_y": 400.0,
        "hazards_above": [
            {"id": 1, "x": 100.0, "y": 410.0},
            {"id": 2, "x": 200.0, "y": 500.0},
        ],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == 0.0
    assert weight_sum == 0.0
    # The defensive filter excludes them, so count is also 0.
    assert count == 0


def test_single_hazard_directly_above_at_lookahead_boundary():
    # vertical_distance = lookahead_band -> vertical_weight = 0
    # -> bonus = 0.
    rs = {
        "player_x": 360.0,
        "player_y": 508.0,
        "hazards_above": [
            {"id": 1, "x": 360.0, "y": 508.0 - DEFAULT_LOOKAHEAD_BAND}
        ],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == pytest.approx(0.0, abs=1e-9)
    assert weight_sum == pytest.approx(0.0, abs=1e-9)
    assert count == 1


def test_single_hazard_at_player_position_zero_clearance():
    # hazard same y as player, same x -> vertical_weight = 1.0,
    # lateral_clearance = 0.0 -> bonus = 0.
    rs = {
        "player_x": 360.0,
        "player_y": 508.0,
        "hazards_above": [{"id": 1, "x": 360.0, "y": 508.0}],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == pytest.approx(0.0, abs=1e-9)
    assert weight_sum == pytest.approx(1.0, abs=1e-9)
    assert count == 1


def test_single_hazard_at_player_y_and_far_lateral_max_bonus():
    # hazard at player y -> vertical_weight = 1.0; lateral distance
    # exceeds safe_lateral_distance -> lateral_clearance = 1.0;
    # bonus = alpha * 1.0.
    rs = {
        "player_x": 100.0,
        "player_y": 508.0,
        "hazards_above": [
            {"id": 1, "x": 100.0 + DEFAULT_SAFE_LATERAL_DISTANCE * 2, "y": 508.0}
        ],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert bonus == pytest.approx(DEFAULT_ALPHA, abs=1e-9)
    assert weight_sum == pytest.approx(1.0, abs=1e-9)
    assert count == 1


def test_bonus_bounded_in_zero_alpha():
    # Construct several extreme cases and verify bonus stays in [0, alpha].
    cases = [
        {"player_x": 16.0, "player_y": 508.0, "hazards_above": [
            {"id": 1, "x": 700.0, "y": 0.0},
            {"id": 2, "x": 16.0, "y": 508.0},
            {"id": 3, "x": 360.0, "y": 250.0},
        ]},
        {"player_x": 360.0, "player_y": 508.0, "hazards_above": [
            {"id": 1, "x": 360.0, "y": 480.0},
        ]},
        {"player_x": 704.0, "player_y": 508.0, "hazards_above": [
            {"id": 1, "x": 16.0, "y": 500.0},
        ]},
    ]
    for rs in cases:
        bonus, _, _ = compute_threat_weighted_clearance(rs)
        assert 0.0 <= bonus <= DEFAULT_ALPHA, (rs, bonus)


def test_alpha_zero_disables_bonus_magnitude():
    rs = {
        "player_x": 100.0,
        "player_y": 508.0,
        "hazards_above": [{"id": 1, "x": 700.0, "y": 500.0}],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs, alpha=0.0)
    assert bonus == 0.0
    # weight sum still computed since the code path still ran.
    assert weight_sum > 0.0
    assert count == 1


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        compute_threat_weighted_clearance(
            {"player_x": 0.0, "player_y": 0.0, "hazards_above": []},
            alpha=-0.01,
        )


def test_invalid_lookahead_band_raises():
    with pytest.raises(ValueError):
        compute_threat_weighted_clearance(
            {"player_x": 0.0, "player_y": 0.0, "hazards_above": []},
            lookahead_band=0.0,
        )
    with pytest.raises(ValueError):
        compute_threat_weighted_clearance(
            {"player_x": 0.0, "player_y": 0.0, "hazards_above": []},
            lookahead_band=-1.0,
        )


def test_invalid_safe_lateral_distance_raises():
    with pytest.raises(ValueError):
        compute_threat_weighted_clearance(
            {"player_x": 0.0, "player_y": 0.0, "hazards_above": []},
            safe_lateral_distance=0.0,
        )
    with pytest.raises(ValueError):
        compute_threat_weighted_clearance(
            {"player_x": 0.0, "player_y": 0.0, "hazards_above": []},
            safe_lateral_distance=-100.0,
        )


def test_malformed_hazard_entries_skipped():
    rs = {
        "player_x": 100.0,
        "player_y": 508.0,
        "hazards_above": [
            "not a dict",
            {"id": 1, "x": 100.0},  # missing y
            {"x": 100.0, "y": 250.0},  # missing id, still readable
            {"id": 2, "x": None, "y": 250.0},  # None x
        ],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    # Only the third entry has parseable x and y; it is valid.
    assert count == 1
    assert weight_sum > 0.0
    assert 0.0 <= bonus <= DEFAULT_ALPHA


def test_lateral_clearance_uses_absolute_distance():
    # Two hazards at mirrored lateral positions must produce identical
    # bonuses; the shaping does not encode preferred direction.
    rs_left = {
        "player_x": 360.0,
        "player_y": 508.0,
        "hazards_above": [{"id": 1, "x": 200.0, "y": 400.0}],
    }
    rs_right = {
        "player_x": 360.0,
        "player_y": 508.0,
        "hazards_above": [{"id": 1, "x": 520.0, "y": 400.0}],
    }
    b_left, _, _ = compute_threat_weighted_clearance(rs_left)
    b_right, _, _ = compute_threat_weighted_clearance(rs_right)
    assert b_left == pytest.approx(b_right, abs=1e-9)


def test_imminent_hazard_dominates_distant_hazard():
    # One hazard close-above with lateral_clearance=1.0, one hazard
    # far-above with lateral_clearance=0.0. Weighted average should be
    # heavily dominated by the close hazard (high vertical_weight),
    # producing bonus close to alpha.
    rs = {
        "player_x": 100.0,
        "player_y": 508.0,
        "hazards_above": [
            {"id": 1, "x": 600.0, "y": 500.0},  # close, far lateral
            {"id": 2, "x": 100.0, "y": 260.0},  # far, zero lateral
        ],
    }
    bonus, weight_sum, count = compute_threat_weighted_clearance(rs)
    assert count == 2
    assert weight_sum > 1.0  # both contribute
    # The close hazard's vertical_weight is ~ 1 - 8/270 ~ 0.97, lateral 1.0.
    # The far hazard's vertical_weight is ~ 1 - 248/270 ~ 0.08, lateral 0.0.
    # Weighted average ~ (0.97 * 1.0 + 0.08 * 0.0) / (0.97 + 0.08) ~ 0.924.
    # bonus ~ 0.05 * 0.924 ~ 0.046.
    assert bonus == pytest.approx(0.05 * 0.97 / (0.97 + 0.08148), abs=2e-3)


# --- env-layer constructor validation -------------------------------------


def test_constructor_default_reward_shaping_is_none():
    env = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        max_steps=10,
    )
    try:
        assert env._reward_shaping == REWARD_SHAPING_NONE
        assert env._reward_shaping_alpha == DEFAULT_ALPHA
        assert env._reward_shaping_lookahead_band == DEFAULT_LOOKAHEAD_BAND
        assert env._reward_shaping_safe_lateral_distance == (
            DEFAULT_SAFE_LATERAL_DISTANCE
        )
    finally:
        env.close()


def test_constructor_rejects_unknown_reward_shaping():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x",
            project_path="y",
            max_steps=10,
            reward_shaping="bogus_shape",
        )


def test_constructor_rejects_negative_alpha():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x",
            project_path="y",
            max_steps=10,
            reward_shaping_alpha=-0.1,
        )


def test_constructor_rejects_nonpositive_lookahead():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x",
            project_path="y",
            max_steps=10,
            reward_shaping_lookahead_band=0.0,
        )


def test_constructor_rejects_nonpositive_safe_lateral():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x",
            project_path="y",
            max_steps=10,
            reward_shaping_safe_lateral_distance=-1.0,
        )


# --- env-layer integration helpers ----------------------------------------


def _make_env(
    *,
    run_dir=None,
    reward_shaping=REWARD_SHAPING_NONE,
    alpha=DEFAULT_ALPHA,
    lookahead_band=DEFAULT_LOOKAHEAD_BAND,
    safe_lateral_distance=DEFAULT_SAFE_LATERAL_DISTANCE,
):
    proc = FakeProcess()
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    proc_factory = FakeProcessFactoryRecorder(proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    env = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        tcp_host="127.0.0.1",
        tcp_port=8765,
        run_dir=run_dir,
        max_steps=100,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        seed=None,
        headless=True,
        reward_shaping=reward_shaping,
        reward_shaping_alpha=alpha,
        reward_shaping_lookahead_band=lookahead_band,
        reward_shaping_safe_lateral_distance=safe_lateral_distance,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    return env, tx, proc


def _reward_state_payload(
    player_x: float = 100.0,
    player_y: float = 508.0,
    hazards_above: list | None = None,
) -> dict:
    return {
        "player_x": player_x,
        "player_y": player_y,
        "hazards_above": hazards_above if hazards_above is not None else [],
    }


# --- env-layer integration tests ------------------------------------------


def test_default_reward_shaping_returns_godot_reward_unchanged():
    env, tx, _ = _make_env(reward_shaping=REWARD_SHAPING_NONE)
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # Even when Godot supplies a reward_state, the default path
        # ignores it.
        info = {
            "reward_state": _reward_state_payload(
                hazards_above=[{"id": 1, "x": 700.0, "y": 500.0}]
            )
        }
        tx.queue_step(_step_ok_payload(reward=1.0, info=info))
        _, reward, terminated, truncated, _ = env.step(1)
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_shaped_reward_adds_clearance_bonus_on_non_terminal_step():
    env, tx, _ = _make_env(
        reward_shaping=REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    )
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # Player at left wall, hazard at player y, lateral distance huge.
        # vertical_weight ~ 1.0, lateral_clearance = 1.0 -> bonus = alpha.
        info = {
            "reward_state": _reward_state_payload(
                player_x=16.0,
                player_y=508.0,
                hazards_above=[{"id": 1, "x": 700.0, "y": 508.0}],
            )
        }
        tx.queue_step(_step_ok_payload(reward=1.0, info=info))
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.0 + DEFAULT_ALPHA, abs=1e-9)
    finally:
        env.close()


def test_shaped_reward_returns_base_only_on_collision_terminal():
    env, tx, _ = _make_env(
        reward_shaping=REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    )
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # Even with a reward_state that would yield max bonus, a
        # collision terminal step must return the Godot-supplied 0.0
        # with no bonus added.
        info = {
            "reward_state": _reward_state_payload(
                hazards_above=[{"id": 1, "x": 700.0, "y": 508.0}]
            )
        }
        tx.queue_step(
            _step_ok_payload(
                reward=0.0,
                terminated=True,
                truncated=False,
                terminal_reason=protocol.TERMINAL_REASON_COLLISION,
                info=info,
            )
        )
        _, reward, terminated, truncated, _ = env.step(0)
        assert reward == 0.0
        assert terminated is True
        assert truncated is False
    finally:
        env.close()


def test_shaped_reward_preserves_base_and_bonus_on_timeout_truncation():
    env, tx, _ = _make_env(
        reward_shaping=REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    )
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # Godot returns +1.0 on the truncation step (terminated=False).
        # The Python env must add the bonus and return base + bonus,
        # mirroring the proposal's note that truncation does not zero
        # out the reward.
        info = {
            "reward_state": _reward_state_payload(
                player_x=360.0,
                player_y=508.0,
                hazards_above=[{"id": 1, "x": 360.0, "y": 508.0}],
            )
        }
        tx.queue_step(
            _step_ok_payload(
                reward=1.0,
                terminated=False,
                truncated=True,
                terminal_reason=protocol.TERMINAL_REASON_TIMEOUT,
                info=info,
            )
        )
        _, reward, terminated, truncated, _ = env.step(1)
        # Co-located hazard -> lateral_clearance 0.0 -> bonus 0.0;
        # the test asserts that base 1.0 is preserved exactly.
        assert reward == pytest.approx(1.0, abs=1e-9)
        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_shaped_reward_with_missing_reward_state_yields_zero_bonus():
    env, tx, _ = _make_env(
        reward_shaping=REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    )
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # No reward_state under info; bonus must collapse to 0.0.
        tx.queue_step(_step_ok_payload(reward=1.0, info={}))
        _, reward, _, _, _ = env.step(1)
        assert reward == 1.0
    finally:
        env.close()


# --- log-schema regression tests ------------------------------------------


def _read_ndjson_events(path) -> list[dict]:
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(_json.loads(line))
    return events


# H5 amendment proposal section 5 item 13 (regression test for default
# path). The default ``reward_shaping: none`` must produce a
# ``python.ndjson`` ``step`` event with the same schema the pre-amendment
# code produced: ``reward``, ``terminated``, ``truncated``,
# ``terminal_reason``, plus the identity decoration the writer always
# injects. No shaped-mode fields are emitted.
def test_default_path_step_log_omits_shaped_fields(tmp_path):
    env, tx, _ = _make_env(run_dir=tmp_path, reward_shaping=REWARD_SHAPING_NONE)
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=42)
        info = {
            "reward_state": _reward_state_payload(
                hazards_above=[{"id": 1, "x": 700.0, "y": 500.0}]
            )
        }
        tx.queue_step(_step_ok_payload(reward=1.0, frame=1, info=info))
        env.step(1)
    finally:
        env.close()

    events = _read_ndjson_events(tmp_path / "python.ndjson")
    step_events = [ev for ev in events if ev["type"] == "step"]
    assert len(step_events) == 1
    ev = step_events[0]
    # Pre-amendment schema: exactly these keys plus the identity
    # decoration. No shaped-mode fields.
    pre_amendment_keys = {
        "ts_unix",
        "type",
        "run_id",
        "godot_pid",
        "tcp_port",
        "episode_id",
        "frame",
        "reward",
        "terminated",
        "truncated",
        "terminal_reason",
    }
    assert set(ev.keys()) == pre_amendment_keys, (
        f"unexpected step keys under reward_shaping=none: "
        f"{sorted(set(ev.keys()) - pre_amendment_keys)} unexpected, "
        f"{sorted(pre_amendment_keys - set(ev.keys()))} missing"
    )
    assert ev["reward"] == 1.0
    for forbidden in (
        "base_reward",
        "clearance_bonus",
        "threat_weight_sum",
        "active_hazard_count_above_player",
    ):
        assert forbidden not in ev, (
            f"shaped-mode field {forbidden!r} leaked into default-path "
            f"step event; this breaks Phase A-E log byte-equality"
        )


def test_shaped_path_step_log_emits_components(tmp_path):
    env, tx, _ = _make_env(
        run_dir=tmp_path,
        reward_shaping=REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE,
    )
    try:
        tx.queue_reset(_reset_ok_payload())
        env.reset(seed=42)
        info = {
            "reward_state": _reward_state_payload(
                player_x=100.0,
                player_y=508.0,
                hazards_above=[
                    {"id": 1, "x": 600.0, "y": 500.0},
                    {"id": 2, "x": 100.0, "y": 260.0},
                ],
            )
        }
        tx.queue_step(_step_ok_payload(reward=1.0, frame=1, info=info))
        env.step(1)
    finally:
        env.close()

    events = _read_ndjson_events(tmp_path / "python.ndjson")
    step_events = [ev for ev in events if ev["type"] == "step"]
    assert len(step_events) == 1
    ev = step_events[0]
    for required in (
        "base_reward",
        "clearance_bonus",
        "threat_weight_sum",
        "active_hazard_count_above_player",
        "reward",
        "terminated",
        "truncated",
        "terminal_reason",
    ):
        assert required in ev, f"shaped-mode missing field {required!r}"
    assert ev["base_reward"] == 1.0
    assert ev["clearance_bonus"] >= 0.0
    assert ev["clearance_bonus"] <= DEFAULT_ALPHA
    assert ev["reward"] == pytest.approx(
        ev["base_reward"] + ev["clearance_bonus"], abs=1e-9
    )
    assert ev["active_hazard_count_above_player"] == 2


# --- config / factory plumbing tests --------------------------------------


def test_resolve_godot_kwargs_threads_reward_shaping():
    from sight_agent.rl.godot_config import resolve_godot_kwargs

    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "n_envs": 1,
            "godot_executable": r"C:\fake\godot.exe",
            "project_path": r"C:\fake\project",
            "reward_shaping": "threat_weighted_clearance",
            "reward_shaping_alpha": 0.05,
            "reward_shaping_lookahead_band": 270,
            "reward_shaping_safe_lateral_distance": 180,
        }
    }
    out = resolve_godot_kwargs(cfg)
    assert out["reward_shaping"] == "threat_weighted_clearance"
    assert out["reward_shaping_alpha"] == 0.05
    assert out["reward_shaping_lookahead_band"] == 270
    assert out["reward_shaping_safe_lateral_distance"] == 180


def test_resolve_godot_kwargs_omits_reward_shaping_when_absent():
    from sight_agent.rl.godot_config import resolve_godot_kwargs

    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "n_envs": 1,
            "godot_executable": r"C:\fake\godot.exe",
            "project_path": r"C:\fake\project",
        }
    }
    out = resolve_godot_kwargs(cfg)
    for key in (
        "reward_shaping",
        "reward_shaping_alpha",
        "reward_shaping_lookahead_band",
        "reward_shaping_safe_lateral_distance",
    ):
        assert key not in out, (
            f"key {key!r} leaked into kwargs when YAML omitted it; "
            f"this would override the env constructor default"
        )


def test_resolve_godot_kwargs_threads_partial_reward_shaping():
    # Forwarding behaviour: a YAML that only sets ``reward_shaping``
    # without overriding the tunables must still forward the variant
    # selector; the constructor defaults will provide the rest.
    from sight_agent.rl.godot_config import resolve_godot_kwargs

    cfg = {
        "env": {
            "id": "godot:signal-dodge-v0",
            "n_envs": 1,
            "godot_executable": r"C:\fake\godot.exe",
            "project_path": r"C:\fake\project",
            "reward_shaping": "none",
        }
    }
    out = resolve_godot_kwargs(cfg)
    assert out["reward_shaping"] == "none"
    assert "reward_shaping_alpha" not in out
    assert "reward_shaping_lookahead_band" not in out
    assert "reward_shaping_safe_lateral_distance" not in out


def test_factory_make_env_signature_accepts_reward_shaping():
    # Belt-and-suspenders: the factory must declare the new kwargs so
    # ``train.py`` / ``evaluate.py`` can splat resolver output without
    # TypeError. Inspecting the signature is cheaper than instantiating
    # a VecEnv here.
    import inspect

    from sight_agent.rl.factories import make_env

    sig = inspect.signature(make_env)
    for kw in (
        "reward_shaping",
        "reward_shaping_alpha",
        "reward_shaping_lookahead_band",
        "reward_shaping_safe_lateral_distance",
    ):
        assert kw in sig.parameters, f"factory missing kwarg {kw!r}"
        assert sig.parameters[kw].kind == inspect.Parameter.KEYWORD_ONLY
