# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 amended. H5 plan section 7 now permits exactly one bounded reward-shaping variant (threat-weighted clearance reward) for H5 continuation. Reward code NOT implemented yet. No training approved against shaped reward until implementation slice lands and tests are green.

**Last commit on HEAD:** `cf427da` docs(h5): reward-amendment proposal and section 7 amendment for threat-weighted clearance reward (chore-refresh follows on push).

**Substantive code/evidence commit:** `cf427da` docs(h5): reward-amendment proposal and section 7 amendment for threat-weighted clearance reward.

**Current task:** Docs-only H5 reward-amendment proposal slice complete. Jeff approved the scope amendment (single bounded variant: threat-weighted clearance reward, base `+1/step` preserved, `alpha=0.05`). `docs/h5-reward-amendment-proposal.md` documents rationale (from `docs/h5-behavior-audit-evidence.md`), exact per-step formula, initial constants (`alpha=0.05`, `lookahead_band=270`, `safe_lateral_distance=180`), implementation constraints (no target-env change, no observation-channel change, no future-collision oracle, no wall-specific penalty, no hyperparameter sweep before first shaped-reward diagnostic, separate per-step logging of `base_reward` and `clearance_bonus`, config-gated with default `reward_shaping: none` byte-identical to pre-amendment), confirmation criteria (action=-1 share < 60%, wall-clamp frame share < 50%, GREEN bar plausible, same-seed trajectories diverge), falsification criteria (collision rate >= 0.90, wall-clamp persists, action=-1 share > 80%), and an explicit no-training-before-code-approval statement. `docs/sight-h5-plan.md` section 7 minimally amended in place with an "Amendment 2026-05-16 (Jeff-approved)" sub-bullet pointing at the proposal doc. Default-tier pytest green (391 passed in 20.57s) with the amended docs in place.

**Next action:** GPT design review of `docs/h5-reward-amendment-proposal.md` section 10 open items (negative-control reward axis question, pre-training smoke for lookahead/clearance constants, optional `active_hazard_count_above_player` log field, regression test for `reward_shaping: none` byte-equality). After GPT review, Claude implements: (a) env-wrapper computation of `clearance_bonus` from existing state dict (player+hazard positions), (b) YAML key `reward_shaping` with default `none` and new value `threat_weighted_clearance` carrying `alpha`/`lookahead_band`/`safe_lateral_distance`, (c) per-step `python.ndjson` fields `base_reward` and `clearance_bonus`, (d) regression test that `reward_shaping: none` is byte-identical to a stored Phase E artifact. Training against shaped reward begins only after the implementation slice is committed and its tests pass.

**Blockers:** None on the docs slice itself. Implementation slice gated on GPT design review of section 10 open items in the proposal doc; the review can land in a normal turn, no external dependency.

**Notes:**

- The amendment is to the H5 plan section 7 only, not to the charter. Charter mission, scope, non-goals, ethics armor, role split, and phase-gate pattern are unchanged. The H5 GREEN bar (section 6), the H5 non-saturation gate (section 5), and the four-policy baseline suite (section 3) are unchanged in structure.
- The H5 pre-training non-saturation gate convention is unchanged: evaluates three negative controls only (`stay_only`, `seeded_random`, `untrained_cnn`). `trained_cnn` does not exist until a training slice produces a `model.zip`. The inherited handoff-precision test in `tests/rl/test_h5_baseline_cli.py` continues to assert this phrasing is preserved here.
- Per-step bonus is bounded in `[0.0, alpha]`. With `alpha=0.05` and an 1800-step max episode, maximum cumulative shaping is `+90.0` against a base of `+1800.0`, so survival remains the dominant gradient signal. The shaping computation lives Python-side in the env wrapper; the Godot reward source remains the existing sparse-survival signal. This keeps Godot side unchanged and makes the shaping ablate-able by config.
- Initial constants are derived from Signal Dodge geometry, not tuned: `lookahead_band = SCREEN_HEIGHT/2 = 270`, `safe_lateral_distance = SCREEN_WIDTH/4 = 180` (well above the 28 px collision alignment threshold). The implementation slice may revise after a pre-training smoke run if the per-step bonus is pathological at these values.
- Operational reminders unchanged: `SIGHT_GODOT_EXE` must be set inline at session start; `-s` required for live pytest under DC; `runs/` remains gitignored. The `godot.ndjson` double-`episode_start` cleanup in `games/signal-dodge/scripts/main.gd` remains a non-urgent target for the next eval-logging revision pass.
