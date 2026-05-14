# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 collision-propagation bug FIXED. Step 2B (additive H5 hard profile YAML and difficulty knob plumbing) was halted pre-implementation pending this fix; the underlying saturation question can now be re-asked truthfully on the H4 pixel profile, but the post-fix smoke shows the H4 profile is NOT saturated (all three negative controls terminate well below the 0.80 length-ratio threshold). Step 2B is no longer obviously needed; the prior Step 2A-lite "100% non-saturation" result was a measurement artifact of the propagation bug, and the corrected measurement shows healthy negative-control termination on the existing H4 profile.

**Last commit:** `30d220d` fix(godot): make H3 collision terminal flag sticky across next step

**Current task:** Land the GDScript collision-propagation fix (`games/signal-dodge/scripts/main.gd`), the four Python-side contract tests (`tests/rl/test_h5_collision_propagation.py`), and this handoff refresh. The fix removes the unconditional start-of-step reset of `_h3_step_terminated` / `_h3_terminal_reason` / `_h3_collision_info` in `_h3_perform_step`; reset clearing remains in `_h3_perform_soft_reset`. The flag is now sticky from `_on_player_died` until the next reset, so a between-step hazard-physics-tick collision is consumed by the very next step reply as `terminated=True`. `_h3_episode_done` still gates step-after-terminal with `bad_request`.

**Next action:** GPT re-asks the original Step 2A-lite saturation question against the corrected post-fix measurement. The H4 pixel profile produced these negative-control results on seeds 1000,1001: stay_only mean_episode_length=303.0, seeded_random=349.5, untrained_cnn=303.0 (all length_ratio≈0.17-0.19, far below the 0.80 saturation threshold; timeout_rate=0.0 for all three). The non-saturation gate now passes without any profile hardening. The remaining downstream H5 work is the trained-CnnPolicy slice plus the four-policy acceptance suite; profile hardening is no longer the obvious next move on the H4 pixel profile. If GPT decides H5 closure requires a harder profile anyway (e.g., to widen the gap between random and trained), the Step 2B knobs (`spawn_interval_frames`, `hazard_speed`, `hazards_per_spawn`) and the new H5 hard YAML remain valid downstream work, just no longer urgent.

**Blockers:** none. Working tree clean post-commit. Origin/main in sync.

**Notes:**

- Live verification of the fix on StrongerJr. Post-fix negative-controls smoke (`h5_baseline_cli ... --mode negative-controls --run-id h5_negative_controls_smoke_postfix --seeds 1000,1001`) returned `passed=True saturated_negative_controls=[]`. `python.ndjson` for stay_only shows 2 step events with `terminated=true terminal_reason="collision"` (one per seed), and `episode_id` advanced from `ep-000001` to `ep-000003`. Pre-fix the same smoke produced 3600 step events all `terminated=false` and a stuck `ep-000001` per seed; the contrast is unambiguous.
- Determinism preserved post-fix. `test_live_godot_pixel_same_seed_step_by_step_trajectory_equality` and `test_live_godot_reset_and_100_step_smoke` both PASSED post-fix on real Godot 4.6.2 (`Godot_v4.6.2-stable_win64.exe`). H3 same-seed reproducibility and H4 same-seed step-by-step pixel byte-equality both still hold.
- Default-tier `pytest tests/rl` post-fix: 290 passed, 0 failed, 2 deselected. Includes the four new tests in `tests/rl/test_h5_collision_propagation.py`. The one transient handoff-wording test (`test_handoff_does_not_describe_pre_training_gate_as_four_policy`) is cleared by this handoff including the substring "non-saturation" again.
- Step 2B status: HALT lifted. Profile-hardening files (H5 hard YAML, godot_config.py difficulty knobs, factories.py wiring, main.gd parameterization, drift-detection tests) were not added in this slice and are no longer obviously needed; the corrected smoke shows the H4 pixel profile already provides a usable non-saturation floor for an H5 trained-policy comparison. GPT decides whether the next slice is the trained-CnnPolicy training run on the existing H4 profile or whether closure margin (per H5 plan section 6: 25% reward gap and 20pp collision-rate reduction) still motivates a harder profile.
- Carry-forward operational invariants: `SIGHT_GODOT_EXE` must be set inline in a `.bat` (User-scope env vars are not inherited by Desktop Commander's parent shell); pre-mode-lock physics-tick variance remains out of scope for same-seed determinism; `runs/` stays gitignored; ethics armor unchanged.


---
