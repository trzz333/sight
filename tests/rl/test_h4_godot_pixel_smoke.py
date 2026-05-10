"""H4 plan section 8 + section 10 criterion 6 live smoke for pixel mode.

Opt-in via ``-m live_godot``. Launches the real Godot Signal Dodge build
through ``GodotSignalDodgeEnv`` with ``observation_mode="pixel"`` and
``headless=False`` (pixel mode requires a windowed launch per the
H4 spike). Runs ``seed=0`` twice with the same scripted action sequence
and asserts every post-reset and post-step pixel observation is
byte-equal across the two runs (step-by-step trajectory equality per
docs/sight-h4-plan.md section 9 and section 10 criterion 6).

Pre-mode-lock physics-tick variance is permitted per the H3 closure
caveat carried forward into H4. Equality comparison applies only to
observations returned through the locked pixel-mode transport, which is
what ``env.reset()`` / ``env.step()`` return after the H4 protocol
handshake.

The two runs execute sequentially (not concurrently) so they cannot
collide on TCP port allocation or on the windowed Godot OS window.

Excluded from the default run by ``addopts`` in ``pyproject.toml``.

Run:
    pytest tests/rl/test_h4_godot_pixel_smoke.py -m live_godot -v --tb=short
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from sight_agent.rl.godot_env import GodotSignalDodgeEnv


# Scripted action sequence shared by both runs. Mix of stay/left/right so
# the rollout exercises actual scene change rather than freezing on a
# single position; freezing would make first-pixel equality trivially
# match step-by-step equality and weaken the determinism assertion.
# Length 10 keeps the live run short enough to be a smoke gate.
_SCRIPTED_ACTIONS: list[int] = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1]


# --- helpers (copied from test_h3_godot_smoke patterns) -----------------


def _allocate_isolated_tcp_port() -> int:
    """Bind to ``127.0.0.1:0``, capture the kernel-assigned port, then release.

    The TOCTOU window between releasing the socket and Godot binding the
    same port is negligible on loopback inside one dev box; the env's
    ``connect_timeout_s`` retry covers transient races.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _repo_root() -> Path:
    # tests/rl/test_h4_godot_pixel_smoke.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def _godot_exe_or_fail() -> Path:
    raw = os.environ.get("SIGHT_GODOT_EXE", "")
    if not raw:
        pytest.fail(
            "SIGHT_GODOT_EXE is not set. The live_godot acceptance gate "
            "requires the Godot 4.x executable path. Set it at User or "
            "Machine scope, e.g.:\n"
            r'  setx SIGHT_GODOT_EXE "C:\path\to\Godot_v4.x-stable_win64.exe"'
        )
    p = Path(raw)
    if not p.is_file():
        pytest.fail(f"SIGHT_GODOT_EXE={raw!r} does not point to a file.")
    return p


def _project_path_or_fail() -> Path:
    p = _repo_root() / "games" / "signal-dodge"
    if not (p / "project.godot").is_file():
        pytest.fail(
            f"Expected Godot project at {p}; project.godot not found."
        )
    return p


def _assert_pixel_obs_well_formed(
    env: GodotSignalDodgeEnv, obs: np.ndarray
) -> None:
    """Per-observation invariants. Shape, dtype, range, and Box.contains.

    The transport already validates the wire payload (mode literal,
    shape match, dtype literal, encoding literal, data length, per-byte
    [0,255] range, plus the metadata fields). This assert defends the
    env-layer contract: numpy reshape produced the documented (C, H, W)
    shape, dtype is uint8, and gym's own contains() agrees the obs
    sits inside the declared observation_space. Do NOT weaken these to
    bypass a transport-level mismatch; tightening transport validation
    is the correct fix.
    """
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 84, 84), (
        f"expected (1, 84, 84), got {obs.shape}"
    )
    assert obs.dtype == np.uint8, (
        f"expected uint8, got {obs.dtype}"
    )
    assert int(obs.min()) >= 0
    assert int(obs.max()) <= 255
    assert env.observation_space.contains(obs), (
        f"obs not contained in observation_space {env.observation_space}"
    )


def _assert_godot_ndjson_minimum(run_dir: Path) -> None:
    """Parse godot.ndjson and require the H3-live minimum event-type set.

    Same minimum as ``test_h3_godot_smoke.test_live_godot_reset_and_100_step_smoke``.
    H4 does not introduce new required event types at this tier; the
    pixel-source and capture-point literals live in the wire payload's
    ``obs`` dict (validated by the transport on every receive) rather
    than in named NDJSON events.

    ``collision`` / ``death`` / ``run_end`` are not required at this
    tier: terminal events are scenario-contingent and ``run_end`` is
    shutdown-timing-sensitive.
    """
    godot_ndjson = run_dir / "godot.ndjson"
    assert godot_ndjson.is_file(), (
        f"godot.ndjson missing at {godot_ndjson}"
    )
    raw_lines = [
        ln for ln in godot_ndjson.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert raw_lines, f"godot.ndjson at {godot_ndjson} has no events"
    types_seen: set[str] = set()
    for ln in raw_lines:
        rec = json.loads(ln)
        assert isinstance(rec, dict), f"non-object NDJSON line: {ln!r}"
        types_seen.add(str(rec.get("type", "")))
    required = {
        "run_start",
        "controller_connected",
        "controller_hello",
        "controller_reset_received",
        "episode_start",
        "h3_step",
    }
    missing = required - types_seen
    assert not missing, (
        f"godot.ndjson missing required event types {sorted(missing)}; "
        f"saw {sorted(types_seen)}"
    )

    python_ndjson = run_dir / "python.ndjson"
    if python_ndjson.is_file():
        for ln in python_ndjson.read_text(encoding="utf-8").splitlines():
            ln_stripped = ln.strip()
            if not ln_stripped:
                continue
            rec = json.loads(ln_stripped)
            assert rec.get("type") != "error", (
                f"python.ndjson contains error event: {rec!r}"
            )


def _run_pixel_scripted_rollout(
    run_dir: Path, actions: list[int]
) -> tuple[list[np.ndarray], Path]:
    """Launch one live Godot pixel-mode rollout and return per-step obs.

    Returns ``(observations, run_dir)``. The ``observations`` list has
    length ``len(actions) + 1``: index 0 is the post-reset obs, indices
    ``1..N`` are the post-step obs returned by ``env.step(actions[i-1])``.
    If the env terminates/truncates mid-rollout the list ends after the
    final returned obs; callers can still compare prefixes element-wise.

    The env is constructed with ``observation_mode="pixel"`` and
    ``headless=False`` (pixel mode requires a windowed launch per
    docs/sight-h4-spike.md). ``max_steps`` is set well above
    ``len(actions)`` so the env-level max-steps clamp does not
    truncate the scripted rollout; per-step timeout is generous to
    cover the windowed render path on the StrongerJr Intel UHD + RTX
    2060 hybrid GPU configuration.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    godot_exe = _godot_exe_or_fail()
    project_path = _project_path_or_fail()
    port = _allocate_isolated_tcp_port()
    env = GodotSignalDodgeEnv(
        godot_executable=godot_exe,
        project_path=project_path,
        tcp_host="127.0.0.1",
        tcp_port=port,
        run_dir=run_dir,
        # > len(actions) so the env-level max-steps clamp does not
        # truncate this rollout. 4x headroom for any scripted-rollout
        # extension; lower bound 120 keeps us above the H3 live smoke's
        # 100-step ceiling.
        max_steps=max(120, len(actions) * 4),
        connect_timeout_s=20.0,
        step_timeout_s=10.0,
        seed=None,
        headless=False,  # pixel mode requires windowed launch.
        observation_mode="pixel",
        pixel_width=84,
        pixel_height=84,
        pixel_channels=1,
    )
    obs_list: list[np.ndarray] = []
    try:
        obs, info = env.reset(seed=0)
        _assert_pixel_obs_well_formed(env, obs)
        assert isinstance(info, dict)
        # The env's _build_info contract: run_id / episode_id /
        # godot_pid / tcp_port / frame are always present; seed is
        # present on reset only. Per-step info is asserted below.
        assert info.get("seed") == 0
        assert "episode_id" in info
        assert "run_id" in info
        assert "godot_pid" in info
        assert info.get("tcp_port") == port
        obs_list.append(obs.copy())

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            _assert_pixel_obs_well_formed(env, obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert "frame" in info
            assert "terminal_reason" in info
            obs_list.append(obs.copy())
            if terminated or truncated:
                break
    finally:
        env.close()

    _assert_godot_ndjson_minimum(run_dir)
    return obs_list, run_dir


# --- live tier (opt-in) -------------------------------------------------


@pytest.mark.live_godot
def test_live_godot_pixel_same_seed_step_by_step_trajectory_equality(
    tmp_path: Path,
) -> None:
    """Same-seed scripted-action pixel observations match step-by-step.

    ``docs/sight-h4-plan.md`` section 9 and section 10 criterion 6: H4
    closes only if same-seed plus same scripted action sequence
    produces matching pixel observations at every post-mode-lock step
    across two runs, NOT merely first-pixel equality. This test is
    that gate at smoke scale: 1 reset + 10 scripted steps, two
    sequential runs, byte-equality on every observation.

    If this test fails due to genuine viewport nondeterminism (frame
    queue depth, render-thread ordering, compositor variance), the
    correct response is to record artifacts and escalate, not to
    relax the equality assertion to approximate matching. Approximate
    matching defeats the gate.
    """
    first, run1_dir = _run_pixel_scripted_rollout(
        tmp_path / "run1", _SCRIPTED_ACTIONS
    )
    second, run2_dir = _run_pixel_scripted_rollout(
        tmp_path / "run2", _SCRIPTED_ACTIONS
    )

    assert len(first) == len(second), (
        f"rollout length mismatch: run1={len(first)} run2={len(second)}; "
        f"artifacts run1={run1_dir} run2={run2_dir}"
    )
    # At minimum the post-reset observation. If the rollout terminated
    # during the first step there will be exactly 2 entries; either way
    # element-wise comparison is meaningful.
    assert len(first) >= 1, "rollout produced no observations"

    for i, (a, b) in enumerate(zip(first, second)):
        # Reassert per-element invariants before comparison so a shape /
        # dtype regression surfaces here instead of in the equality
        # message (which would be confusing).
        assert a.shape == b.shape == (1, 84, 84)
        assert a.dtype == b.dtype == np.uint8
        assert np.array_equal(a, b), (
            f"pixel observation mismatch at index {i} "
            f"(0 == post-reset, 1..N == post-step). "
            f"run1 mean={float(a.mean()):.3f} max={int(a.max())} "
            f"min={int(a.min())}; "
            f"run2 mean={float(b.mean()):.3f} max={int(b.max())} "
            f"min={int(b.min())}; "
            f"artifacts run1={run1_dir} run2={run2_dir}"
        )
