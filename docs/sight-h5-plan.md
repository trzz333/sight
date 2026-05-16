# Sight - H5 Plan

H5 phase plan. Authored after the H4 phase-gate closed GREEN
(`docs/grok-h4-final-green.md`) and after the two pre-H5 hardening items
listed in section 6 of the closure doc landed as a single docs-adjacent
slice (transport literal-pinning, per-reset NDJSON obs metadata
persistence). H5 implementation does not start until those tests are
green on `origin/main`.

This doc is the planning artifact. GPT owns refinements. Claude executes
H5 implementation per a follow-on prompt that references this doc.

---

## 1. Purpose

Learning evaluation of the small CNN policy on Signal Dodge or a
successor microgame. H5 is the first phase where learning-quality
evidence is gated. Everything prior to H5 has been wiring, plumbing,
reproducibility, and capture-path proofs.

H4 was smoke / boundary only. The H4 training pair's eval
`mean_reward = 1800.0` reflects deterministic eval rollout survival
under a freshly initialized `CnnPolicy` at Signal Dodge's step-0
hazard density. It is NOT learning evidence. Any acceptance signal
that "training works" must come from H5.

## 2. Hard preconditions

H5 implementation does not start until ALL of the following hold:

1. The transport pixel-obs metadata literals are pinned
   (`src/sight_agent/rl/godot_transport.py::_validate_pixel_obs`):
   `pixel_source == PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT`,
   `capture_point == CAPTURE_POINT_FRAME_POST_DRAW`,
   `headless_allowed is False`. Tests in
   `tests/rl/test_h4_godot_transport_protocol.py` green.
2. The env persists pixel-obs metadata once per reset to
   `python.ndjson` from `src/sight_agent/rl/godot_env.py` as event
   `type = "obs_metadata"` with `episode_id`, `observation_mode`,
   `shape`, `dtype`, `encoding`, `pixel_source`, `capture_point`,
   `headless_allowed`, `viewport_width`, `viewport_height`, plus the
   auto-decorated `run_id` / `godot_pid` / `tcp_port`. Tests in
   `tests/rl/test_h4_godot_env_protocol.py` green.
3. Default-tier pytest suite passes on a fresh clone of the commit
   that lands the two items above.
4. Charter invariants hold per `docs/sight-charter.md` (no live
   commercial games, no platform automation, no bot-detection
   evasion, no Freecash / offerwalls / account farming, loopback-only
   TCP).

## 3. Required baseline suite

H5 acceptance requires evaluation of FOUR policies on the same
microgame profile, same seed set, same eval posture. The negative
controls exist to prove the task itself has signal to extract;
without them, "trained policy reaches X mean reward" is not
interpretable.

1. **stay-only policy** - deterministic action 1 (stay) every step.
   Establishes the floor for a do-nothing baseline.
2. **seeded random policy** - uniform Discrete(3) sampling under a
   fixed RNG seeded per eval seed. Establishes the random baseline.
3. **untrained CnnPolicy** - the SB3 PPO `CnnPolicy` immediately
   after construction (zero training steps). Establishes the
   "weights carry no task knowledge" baseline. This is the H4
   boundary signal.
4. **trained CnnPolicy** - the policy under test, after the H5
   training run.

All four are evaluated with the same `evaluate.py`-equivalent code
path. Any per-policy branching is recorded in the eval summary.

## 4. Required eval posture

- Fixed multi-seed evaluation set. **Minimum 10 seeds.** Prefer 16 or
  20 if total runtime allows. Seeds are recorded in
  `config_effective.yaml` for reproducibility.
- Per-seed reporting: mean reward, median reward, episode length,
  collision rate, timeout rate. Per-seed table in eval summary.
- Aggregate reporting: mean / median / min / max / std across seeds
  per policy.
- **No single-seed acceptance.** A single-seed result is a smoke
  signal, not learning evidence.
- Eval is deterministic per seed: same policy + same seed produces
  byte-equal trajectories. Reuses the H4 same-seed reproducibility
  posture for pixel mode (post-mode-lock observations).

## 5. Non-saturation rule

The negative controls (stay-only, seeded random, untrained
`CnnPolicy`) must not saturate the configured Signal Dodge profile.
If they do, the profile is too easy and the H5 learning signal will
be drowned.

### Threshold (canonical)

A negative control is **saturated** on the configured profile if
either of the following holds across the eval seed set:

- `timeout_rate >= 0.50`, OR
- `mean_episode_length >= 0.80 * max_steps`.

The Signal Dodge profile **fails the H5 non-saturation gate** if any
of the three negative controls (stay-only, seeded random, untrained
`CnnPolicy`) is saturated by this rule.

The exact threshold values used must be recorded in each policy's
evaluation summary under `non_saturation_thresholds`, and the
per-profile pass/fail decision must be recorded under
`saturation_decision` in the H5 evaluation index. Changing the
threshold is a docs-level amendment to this section, not a runtime
knob.

Training a CnnPolicy against a saturated profile is explicitly
disallowed as H5 learning evidence; any reward / length / collision
gap measured against a saturated negative-control floor is not
interpretable as learning.

### On failure

If the profile fails the non-saturation gate:

- H5 MUST add a harder Signal Dodge profile (higher hazard density,
  shorter spawn intervals, faster moving hazards) OR a successor
  microgame before H5 closure.
- The new profile's negative controls must show a measurable gap
  between random and untrained `CnnPolicy` performance, or the
  profile is also too easy.
- The added game must respect the charter ethics armor. No
  commercial-game asset reuse. No platform automation. The successor
  microgame, if added, is built inside `games/` and uses the same H3
  TCP protocol.

## 6. Learning GREEN bar

H5 closes GREEN only if ALL of the following are true on the
configured (post-non-saturation-rule) profile:

- Trained policy beats the best negative control by at least
  **25% mean reward OR 25% mean episode length.** Whichever metric
  is the more sensitive on the chosen profile is the primary; the
  other is reported. (Reward and length are not redundant: a profile
  where every survived step earns `+1` makes them equivalent, but
  any shaping or per-event reward changes that.)
- Trained policy reduces **collision rate by at least 20 percentage
  points** vs the best negative control.
- Acceptance artifacts include, under `runs/rl/<run_id>/`:
  - `python.ndjson` (with `obs_metadata` events per reset)
  - `godot.ndjson`
  - `summary.json` (per-seed and aggregate metrics)
  - `config_effective.yaml`
  - `model.zip` (final SB3 PPO checkpoint)
  - `evaluation/` directory containing one summary per policy
    (stay-only, seeded random, untrained CnnPolicy, trained
    CnnPolicy)
- The same-seed reproducibility test from H4 still passes on the H5
  HEAD (no regression in capture-path determinism).
- A Grok phase-gate packet is produced and verdict is recorded at
  `docs/grok-h5-final-*.md` per the charter phase-gate pattern.

If the GREEN bar is not met, H5 stays open. Reopening H4 is not
required; the H4 closure record stands.

## 7. H5 non-goals

Hard exclusions for the H5 slice. These are not "deferred"; they are
out of scope for H5 closure regardless of how convenient they would
be:

- No commercial games. Signal Dodge or a custom successor microgame
  only.
- No platform automation. No Freecash, no offerwalls, no third-party
  game launchers, no browser scrape pipelines.
- No bot-detection evasion of any kind. Any anti-bot surface
  encountered during evaluation closes the path immediately.
- No reward shaping unless separately justified by an amendment
  to this section. The H5 baseline reward is the existing Signal
  Dodge sparse-survival reward (`+1.0` per non-terminal step,
  `0.0` at collision terminal).
  - **Amendment 2026-05-16 (Jeff-approved):** exactly one
    bounded reward-shaping variant is permitted for H5
    continuation, threat-weighted clearance reward, as
    specified in `docs/h5-reward-amendment-proposal.md`. The
    base `+1/step` survival reward is preserved; the shaping
    coefficient starts at `alpha = 0.05` and the per-step
    bonus is bounded in `[0.0, alpha]`. No target-environment
    change, no observation-channel change, no commercial-game
    scope change, no platform automation, and no
    hyperparameter sweep before the first shaped-reward
    diagnostic. The negative controls (stay-only, seeded
    random, untrained `CnnPolicy`) continue to be evaluated
    under the base sparse-survival reward; the shaped reward
    applies to the trained `CnnPolicy` only. Any further
    reward variant requires a new amendment to this section.
- No GPU dependency for acceptance. CPU PPO must produce the GREEN
  signal. GPU may be used for faster iteration but acceptance runs
  ship reproducibly on the StrongerJr CPU.
- No product framing. H5 is internal evaluation evidence, not a
  release.

## 8. Carry-forward operational constraints

From H4 closure (`docs/grok-h4-final-green.md` section 6) and H3:

- `-s` is required for live pytest under Desktop Commander on
  StrongerJr. Direct `python -m sight_agent.rl.train` is unaffected.
- `SIGHT_GODOT_EXE` must be set inline (User-scope env vars are not
  inherited by Desktop Commander's parent shell).
- `runs/` remains gitignored. Acceptance artifacts live durably on
  disk on StrongerJr but are not committed.
- Pre-mode-lock physics-tick variance from H3 still applies. Same-
  seed reproducibility assertions apply only to post-mode-lock
  observations returned through `env.reset()` / `env.step()`.

## 9. Phase-gate closure packet

After H5 evidence exists on disk, a Grok phase-gate packet is
required before H5 closes. The packet inlines critical evidence and
follows the same pattern as `docs/grok-h4-phase-gate-packet.md`:

- Acceptance criteria checklist (section 6 of this doc, item by
  item).
- Per-seed result tables for all four policies.
- SHA-256 fingerprints for `model.zip`, `summary.json`,
  `config_effective.yaml`, and the eval summaries.
- A short narrative: what changed since H4, what learning evidence
  was produced, what the non-saturation decision was, what the
  GREEN bar margin looked like.
- Charter-invariant re-check.

Grok's verdict is the authoritative closure record. The packet is
the evidence record. Both are committed under `docs/`.

## 10. What this doc is not

- Not a charter amendment. The charter (`docs/sight-charter.md`) is
  the authority on mission, scope, non-goals, ethics armor, and
  phase-gate pattern.
- Not the H5 implementation prompt. The implementation prompt comes
  separately from GPT and references this doc as the acceptance
  authority.
- Not an open-ended research plan. H5 is bounded to learning-
  evaluation evidence on a microgame Sight already owns end-to-end.
