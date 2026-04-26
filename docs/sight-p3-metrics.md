# Sight P3 Metrics Spec

One-page specification for the P3 measurement layer. Implementation is deferred to a later slice. This document fixes the metric set and the SIGHT_TCP_IGNORE_DEATH exclusion invariant.

## Purpose

P3 (measurement layer) defines how gameplay outcomes are quantified. It is not a continuation of Phase B. Phase B's 300-action clean run validated the TCP transport spine and proved transport endurance only; it did not measure agent gameplay performance. P3 metrics describe agent behavior in real episodes where the player can die.

## Scope

In scope: per-episode metrics computed from logs of a complete Godot run with the harness in normal, death-respecting mode.

Out of scope (this spec): ML training curves, leaderboards, multi-agent comparison, dashboards, replay tooling.

## Metric definitions

### Win rate

Fraction of episodes where the agent reached the configured success terminal (e.g., goal reached, score threshold). Computed as wins / total_episodes. Episodes excluded by the IGNORE_DEATH invariant (below) do not count toward total_episodes.

### Episode length, actions

Number of agent actions emitted in the episode, counted as the population of action events in python.ndjson. Reported per-episode and as mean, median, p95 across the eval batch.

### Episode length, wall-time

Seconds from first action timestamp to terminal-event timestamp in godot.ndjson. Reported per-episode and as mean, median, p95.

### Action distribution, counts

Histogram of action types emitted in the episode (e.g., move_left, move_right, jump, noop). Aggregated across the batch as a totals table.

### Action distribution, entropy

Shannon entropy in bits of the per-episode action histogram, H = -sum(p_i * log2(p_i)) over action types with p_i > 0. Reported per-episode and as batch mean. Detects degenerate single-action policies.

### Failure taxonomy

Each non-win episode is labeled with exactly one terminal cause. Allowed values:

- hazard_collision: agent died from an in-game hazard (intended game-over)
- transport_drop: TCP transport lost or unrecoverable mid-episode
- harness_abort: harness-side error or supervisor abort
- timeout: episode exceeded the configured action or wall-time budget without reaching a terminal
- other: any cause not matching the above; must include a short reason string

Reported as a counts table.

## Run artifact shape (high-level)

Per evaluation batch, on disk:

```
runs\eval\<run_id>\
  episodes\<episode_id>\godot.ndjson
  episodes\<episode_id>\python.ndjson
  summary.json
```

python.ndjson is required when the agent emits structured logs. summary.json contains the batch-level aggregates (win rate, length stats, action totals, entropy mean, failure counts) and the list of episode_ids contributing to each.

Detailed schemas for summary.json and per-episode field names are deferred to the implementation slice. This spec fixes the metric set and the invariant; field-level NDJSON keys are an implementation concern.

## Invariant: SIGHT_TCP_IGNORE_DEATH exclusion

Any episode produced by a run with the env var SIGHT_TCP_IGNORE_DEATH set to any non-empty value is invalid for P3 metrics and must not appear in any aggregate. Enforcement points (binding on the future implementation slice):

- scripts/run_p3_eval.py must refuse to start if SIGHT_TCP_IGNORE_DEATH is set in the inherited environment.
- src/evaluator/ must skip and log any per-episode artifact whose run metadata indicates the flag was active.
- A regression test will fail if the literal string SIGHT_TCP_IGNORE_DEATH appears anywhere under src/evaluator/ or in scripts/run_p3_eval*.py outside an explicit refusal-check guard.

This is a hard exclusion, not a warning. SIGHT_TCP_IGNORE_DEATH was a Phase B transport reachability fix; it cannot contribute to gameplay or survivability claims.

## Non-goals (P3 entry)

- No ML training loss, reward, or sample-efficiency metric.
- No dashboard, live charts, or replay tool.
- No production survivability claim derived from the Phase B 300-action transport run.
- No multi-game aggregation; the first eval target is the existing Godot micro-game.

## Review checklist (before any P3 code begins)

- [ ] Metric set above is complete enough for the first reportable result; nothing important missing.
- [ ] Failure taxonomy values are exhaustive for the current game; other is genuinely a fallback.
- [ ] Invariant language on SIGHT_TCP_IGNORE_DEATH is unambiguous and enforceable in code.
- [ ] Non-goals list correctly fences scope to one page of work, not three.
- [ ] Artifact directory layout (runs\eval\) does not collide with runs\diagnostics\ Phase B artifacts.

On checklist pass, the next implementation slice is scripts/run_p3_eval.py plus a minimal src/evaluator/ aggregator. Tests follow in the same slice or immediately after.