# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (collision-propagation bug fixed; non-saturation gate now PASSES on the H4 pixel profile)

**Last commit:** `1c5c323` chore: refresh handoff hash

**Current task:** The GDScript collision-propagation bug from `docs/h5-collision-propagation-bug.md` is fixed at `30d220d`. `_h3_perform_step` no longer wipes the H3 terminal flags at start-of-step; the sticky flag set by `_on_player_died` now reaches Python on the next step reply. Post-fix smoke on the H4 pixel profile shows all three negative controls terminate well below the 0.80 length-ratio threshold (stay_only=303.0, seeded_random=349.5, untrained_cnn=303.0; timeout_rate=0.0 for all three).

**Next action:** GPT decides whether the next H5 slice is the trained-CnnPolicy training run on the existing H4 pixel profile, or whether closure-margin headroom (H5 plan section 6: 25% reward gap, 20pp collision-rate reduction) still motivates profile hardening. The corrected smoke shows Step 2B as originally scoped is solving a problem the data no longer shows.

**Blockers:** none.

**Notes:**

- Post-fix live verification (seeds 1000,1001): `python.ndjson` for `godot-eval-stay_only` shows 606 step events, 2 with `terminated=true terminal_reason="collision"` (one per seed), episode_id progresses `ep-000001` to `ep-000003`. Pre-fix: 3600 step events all `terminated=false`, stuck `ep-000001`.
- `test_live_godot_pixel_same_seed_step_by_step_trajectory_equality` and `test_live_godot_reset_and_100_step_smoke` both PASSED post-fix; H3/H4 determinism preserved.
- `pytest tests/rl` default tier post-fix: 290 passed, 0 failed, 2 deselected. Includes 4 new tests in `tests/rl/test_h5_collision_propagation.py` covering the between-step contract, reset clear, env-layer step-after-terminal guard, and `GodotRemoteError` propagation.
- Step 2B status: halt lifted but no longer obviously needed; the original "100% saturation" Step 2A-lite result was a measurement artifact of the propagation bug. The H4 pixel profile already provides a usable non-saturation floor.
- Carry-forward operational invariants unchanged: `SIGHT_GODOT_EXE` must be set inline in a `.bat` for live tests; pre-mode-lock physics-tick variance remains out of scope for same-seed determinism; `runs/` stays gitignored; ethics armor unchanged.
