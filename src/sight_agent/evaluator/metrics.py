"""P3 metric aggregator.

Pure Python computation over per-episode records. No I/O, no env var reads.
The companion runner (scripts/run_p3_eval.py, future slice) refuses to start
when SIGHT_TCP_IGNORE_DEATH is set; this module enforces the same invariant
by excluding any episode flagged ignore_death_active=True.

See docs/sight-p3-metrics.md for definitions.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TERMINAL_EVENTS: tuple[str, ...] = (
    "success_budget_reached",
    "hazard_collision",
    "transport_drop",
    "harness_abort",
    "timeout",
    "other",
)
SUCCESS_TERMINALS: frozenset = frozenset({"success_budget_reached"})
FAILURE_TERMINALS: tuple[str, ...] = tuple(
    t for t in TERMINAL_EVENTS if t not in SUCCESS_TERMINALS
)


@dataclass(frozen=True)
class Episode:
    """A single evaluable episode, post-classification.

    Loaders are responsible for parsing raw NDJSON into this shape and setting
    ``ignore_death_active`` from per-run metadata. The metric layer never reads
    environment variables; it trusts the loader contract and enforces the
    spec exclusion invariant by treating any flagged episode as excluded.
    """

    episode_id: str
    terminal: str
    actions: tuple[str, ...]
    wall_time_seconds: float
    ignore_death_active: bool = False
    other_reason: str | None = None

    def __post_init__(self) -> None:
        if self.terminal not in TERMINAL_EVENTS:
            raise ValueError(
                f"unknown terminal {self.terminal!r}; allowed: {TERMINAL_EVENTS}"
            )
        if self.terminal == "other" and not self.other_reason:
            raise ValueError("terminal=other requires a non-empty other_reason")
        if self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be non-negative")


def is_excluded(ep: Episode) -> bool:
    """Spec invariant: any ignore-death run is excluded from all metrics."""
    return ep.ignore_death_active


def is_win(ep: Episode) -> bool:
    return ep.terminal in SUCCESS_TERMINALS


def episode_action_count(ep: Episode) -> int:
    return len(ep.actions)


def episode_wall_time(ep: Episode) -> float:
    return ep.wall_time_seconds


def action_distribution(ep: Episode) -> dict:
    return dict(Counter(ep.actions))


def shannon_entropy_bits(distribution: Mapping[str, int]) -> float:
    """Shannon entropy in bits over a non-negative-integer histogram.

    Returns 0.0 for empty histograms or single-bin histograms (degenerate).
    """
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in distribution.values():
        if count <= 0:
            continue
        p = count / total
        h -= p * math.log2(p)
    return h


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _length_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
    }


def aggregate(episodes: Iterable[Episode]) -> dict:
    """Aggregate metrics over an iterable of Episode records.

    Spec contract:
    - Episodes with ignore_death_active=True are excluded from every aggregate
      and counted only as ``excluded_count``.
    - Win rate counts only success terminals.
    - Failure counts cover the five non-success terminals.
    - Empty input yields zeroed aggregates rather than raising.
    """
    excluded_count = 0
    counted = []
    for ep in episodes:
        if is_excluded(ep):
            excluded_count += 1
            continue
        counted.append(ep)

    total = len(counted)
    wins = sum(1 for ep in counted if is_win(ep))
    win_rate = (wins / total) if total > 0 else 0.0

    action_counts = [episode_action_count(ep) for ep in counted]
    wall_times = [episode_wall_time(ep) for ep in counted]

    batch_action_counts: Counter = Counter()
    per_episode_entropies = []
    for ep in counted:
        dist = action_distribution(ep)
        batch_action_counts.update(dist)
        per_episode_entropies.append(shannon_entropy_bits(dist))

    terminal_counts = {t: 0 for t in TERMINAL_EVENTS}
    for ep in counted:
        terminal_counts[ep.terminal] += 1
    failure_counts = {t: terminal_counts[t] for t in FAILURE_TERMINALS}

    entropy_mean = (
        float(statistics.fmean(per_episode_entropies))
        if per_episode_entropies
        else 0.0
    )

    return {
        "total_episodes": total,
        "excluded_count": excluded_count,
        "wins": wins,
        "win_rate": win_rate,
        "episode_length_actions": _length_stats(action_counts),
        "episode_length_wall_time": _length_stats(wall_times),
        "action_distribution_counts": dict(batch_action_counts),
        "action_distribution_entropy_mean_bits": entropy_mean,
        "terminal_counts": terminal_counts,
        "failure_counts": failure_counts,
    }
