# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 amended. Section 7 permits exactly one bounded reward-shaping variant (threat-weighted clearance reward). Reward code not implemented. No training approved against shaped reward until the implementation slice lands and tests are green.

**Last commit:** `cf427da` docs(h5): reward-amendment proposal and section 7 amendment for threat-weighted clearance reward.

**Current task:** Docs-only H5 reward-amendment proposal slice complete. Jeff approved the scope amendment (single bounded variant, base `+1/step` preserved, `alpha=0.05`). `docs/h5-reward-amendment-proposal.md` specifies rationale tied to the behavior audit, the exact per-step formula, initial constants (`alpha=0.05`, `lookahead_band=270`, `safe_lateral_distance=180`), implementation constraints, confirmation and falsification criteria, and an explicit no-training-before-code-approval statement. `docs/sight-h5-plan.md` section 7 minimally amended in place with a Jeff-approved sub-bullet pointing at the proposal. Default-tier pytest green (391 passed in 20.57s).

**Next action:** GPT design review of `docs/h5-reward-amendment-proposal.md` section 10 open items (negative-control reward axis, pre-training smoke for lookahead/clearance constants, optional `active_hazard_count_above_player` log field, regression test for `reward_shaping: none` byte-equality). After review, Claude implements env-wrapper computation of `clearance_bonus`, the `reward_shaping` YAML key with default `none` byte-identical to pre-amendment behavior, per-step logging of `base_reward` and `clearance_bonus` to `python.ndjson`, and the byte-equality regression test. Training against shaped reward begins only after that implementation slice commits and its tests pass.

**Blockers:** none.

**Notes:**

- Amendment is to H5 plan section 7 only, not to the charter. Charter mission, scope, non-goals, ethics armor, role split, and phase-gate pattern are unchanged. The H5 GREEN bar (section 6), the H5 non-saturation gate (section 5), and the section 3 baseline suite (stay_only, seeded_random, untrained_cnn, trained_cnn) is unchanged in structure.
- Per-step bonus bounded in `[0.0, alpha]`. With `alpha=0.05` and 1800-step max episode, max cumulative shaping is `+90.0` against a base of `+1800.0`, so survival remains the dominant gradient signal. Shaping is computed Python-side in the env wrapper; the Godot reward source is unchanged, which keeps the shaping ablate-able by config.
- Initial constants derived from Signal Dodge geometry, not tuned: `lookahead_band = SCREEN_HEIGHT / 2 = 270`, `safe_lateral_distance = SCREEN_WIDTH / 4 = 180` (well above the 28 px collision alignment threshold). The implementation slice may revise these after a pre-training smoke run.
- Pre-training non-saturation gate convention unchanged. It evaluates three negative controls only (stay_only, seeded_random, untrained_cnn). The inherited precision test in `tests/rl/test_h5_baseline_cli.py` continues to assert this phrasing is preserved here.
- Operational reminders. `SIGHT_GODOT_EXE` set inline at session start. `-s` required for live pytest under DC. `runs/` remains gitignored. Godot `episode_start` double-emit cleanup in `games/signal-dodge/scripts/main.gd` remains a non-urgent target.
