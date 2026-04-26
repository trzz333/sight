"""Tests for the P3 pure metric aggregator.

Synthetic fixtures only. No I/O, no Godot, no harness. See
docs/sight-p3-metrics.md for the spec these tests pin.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sight_agent.evaluator.metrics import (
    Episode,
    FAILURE_TERMINALS,
    SUCCESS_TERMINALS,
    TERMINAL_EVENTS,
    action_distribution,
    aggregate,
    is_excluded,
    is_win,
    shannon_entropy_bits,
)


# --- fixtures ---------------------------------------------------------------


def _success(eid: str = "ep_success") -> Episode:
    return Episode(
        episode_id=eid,
        terminal="success_budget_reached",
        actions=("left", "right", "noop", "noop", "left", "right", "noop", "noop"),
        wall_time_seconds=4.0,
    )


def _hazard(eid: str = "ep_hazard") -> Episode:
    return Episode(
        episode_id=eid,
        terminal="hazard_collision",
        actions=("left", "left", "noop"),
        wall_time_seconds=1.5,
    )


def _excluded(eid: str = "ep_excluded") -> Episode:
    return Episode(
        episode_id=eid,
        terminal="success_budget_reached",
        actions=("left", "left", "left", "left"),
        wall_time_seconds=2.0,
        ignore_death_active=True,
    )


# --- terminal classification ------------------------------------------------


def test_success_budget_counts_as_win_not_timeout():
    ep = _success()
    assert is_win(ep) is True
    assert ep.terminal in SUCCESS_TERMINALS
    assert ep.terminal != "timeout"

    metrics = aggregate([ep])
    assert metrics["wins"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["terminal_counts"]["success_budget_reached"] == 1
    assert metrics["terminal_counts"]["timeout"] == 0
    assert metrics["failure_counts"]["timeout"] == 0


def test_hazard_collision_counts_as_failure_not_win():
    ep = _hazard()
    assert is_win(ep) is False

    metrics = aggregate([ep])
    assert metrics["wins"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["terminal_counts"]["hazard_collision"] == 1
    assert metrics["failure_counts"]["hazard_collision"] == 1


def test_terminal_events_set_matches_spec():
    assert TERMINAL_EVENTS == (
        "success_budget_reached",
        "hazard_collision",
        "transport_drop",
        "harness_abort",
        "timeout",
        "other",
    )
    assert SUCCESS_TERMINALS == frozenset({"success_budget_reached"})
    assert set(FAILURE_TERMINALS) == set(TERMINAL_EVENTS) - SUCCESS_TERMINALS


def test_unknown_terminal_rejected():
    with pytest.raises(ValueError):
        Episode(
            episode_id="bad",
            terminal="ragequit",
            actions=(),
            wall_time_seconds=0.0,
        )


def test_other_terminal_requires_reason():
    with pytest.raises(ValueError):
        Episode(
            episode_id="bad_other",
            terminal="other",
            actions=(),
            wall_time_seconds=0.0,
        )
    Episode(
        episode_id="ok_other",
        terminal="other",
        actions=(),
        wall_time_seconds=0.0,
        other_reason="manual_stop",
    )


# --- entropy ----------------------------------------------------------------


def test_entropy_uniform_four_actions_is_two_bits():
    h = shannon_entropy_bits({"a": 5, "b": 5, "c": 5, "d": 5})
    assert math.isclose(h, 2.0, rel_tol=1e-9, abs_tol=1e-9)


def test_entropy_two_action_50_50_is_one_bit():
    h = shannon_entropy_bits({"left": 7, "right": 7})
    assert math.isclose(h, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_entropy_single_action_is_zero():
    assert shannon_entropy_bits({"noop": 100}) == 0.0


def test_entropy_empty_distribution_is_zero():
    assert shannon_entropy_bits({}) == 0.0
    assert shannon_entropy_bits({"a": 0, "b": 0}) == 0.0


def test_aggregate_mixed_distribution_entropy():
    """Two episodes: uniform(4)=2.0 bits and 50/50=1.0 bits -> mean 1.5."""
    ep_uniform = Episode(
        episode_id="uniform",
        terminal="success_budget_reached",
        actions=("a", "b", "c", "d", "a", "b", "c", "d"),
        wall_time_seconds=1.0,
    )
    ep_50_50 = Episode(
        episode_id="binary",
        terminal="success_budget_reached",
        actions=("left", "right", "left", "right"),
        wall_time_seconds=1.0,
    )
    metrics = aggregate([ep_uniform, ep_50_50])
    assert math.isclose(
        metrics["action_distribution_entropy_mean_bits"], 1.5,
        rel_tol=1e-9, abs_tol=1e-9,
    )
    assert metrics["action_distribution_counts"] == {
        "a": 2, "b": 2, "c": 2, "d": 2, "left": 2, "right": 2,
    }


# --- ignore-death exclusion -------------------------------------------------


def test_excluded_episode_does_not_contribute_to_aggregates():
    """ignore_death_active=True -> excluded from every aggregate (spec invariant)."""
    ep = _excluded()
    assert is_excluded(ep) is True

    metrics = aggregate([ep])
    assert metrics["total_episodes"] == 0
    assert metrics["excluded_count"] == 1
    assert metrics["wins"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["action_distribution_counts"] == {}
    assert metrics["action_distribution_entropy_mean_bits"] == 0.0
    assert metrics["terminal_counts"]["success_budget_reached"] == 0


def test_excluded_does_not_inflate_mixed_batch():
    """One win + one hazard + one excluded. Excluded must vanish."""
    win_ep = _success()
    fail_ep = _hazard()
    skip_ep = _excluded()

    metrics = aggregate([win_ep, fail_ep, skip_ep])
    assert metrics["total_episodes"] == 2
    assert metrics["excluded_count"] == 1
    assert metrics["wins"] == 1
    assert metrics["win_rate"] == 0.5
    assert metrics["terminal_counts"]["success_budget_reached"] == 1
    assert metrics["terminal_counts"]["hazard_collision"] == 1

    expected_counts = {}
    for ep in (win_ep, fail_ep):
        for a in ep.actions:
            expected_counts[a] = expected_counts.get(a, 0) + 1
    assert metrics["action_distribution_counts"] == expected_counts


# --- regression guard on the IGNORE_DEATH literal ---------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIRS = (
    REPO_ROOT / "src" / "evaluator",
    REPO_ROOT / "src" / "sight_agent" / "evaluator",
)
SCRIPTS_DIR = REPO_ROOT / "scripts"
GUARD_TOKENS = (
    "refus",      # refuse, refusing, refused, refusal
    "rais",       # raise, raised, raising
    "skip",
    "exclud",     # exclude, excluded, excluding, exclusion
    "block",
    "ban",
    "forbid",
    "reject",
    "guard",
    "invariant",
    "spec",
    "p3-metrics",
)


def _is_guard_line(line: str) -> bool:
    low = line.lower()
    return any(tok in low for tok in GUARD_TOKENS)


def _scan(directory: Path, glob: str = "*.py") -> list:
    findings = []
    if not directory.exists():
        return findings
    for path in sorted(directory.rglob(glob)):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "SIGHT_TCP_IGNORE_DEATH" in line and not _is_guard_line(line):
                findings.append((str(path), lineno, line.rstrip()))
    return findings


def test_ignore_death_literal_only_in_guards_under_evaluator():
    """Spec invariant: SIGHT_TCP_IGNORE_DEATH must not appear in metric paths
    under src/evaluator/ or src/sight_agent/evaluator/ except inside an
    explicit refusal-check guard. Fails loudly on any future loosening."""
    findings = []
    for d in EVALUATOR_DIRS:
        findings.extend(_scan(d))
    assert findings == [], (
        f"Unguarded SIGHT_TCP_IGNORE_DEATH occurrences: {findings}"
    )


def test_ignore_death_literal_only_in_guards_in_run_p3_eval_scripts():
    """Same invariant on scripts/run_p3_eval*.py. Script may not exist yet."""
    findings = []
    if SCRIPTS_DIR.exists():
        for path in sorted(SCRIPTS_DIR.glob("run_p3_eval*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "SIGHT_TCP_IGNORE_DEATH" in line and not _is_guard_line(line):
                    findings.append((str(path), lineno, line.rstrip()))
    assert findings == [], (
        f"Unguarded SIGHT_TCP_IGNORE_DEATH in run_p3_eval*.py: {findings}"
    )
