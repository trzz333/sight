# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 implementation BLOCKED on a GDScript-side collision-propagation bug surfaced by the Step 2A-lite live smoke. Step 2B (additive H5 hard profile YAML and difficulty knob plumbing) is HALTED pre-implementation per the prompt's own strongest-case-against clause. Evidence write-up at `docs/h5-collision-propagation-bug.md`. The H5 baseline harness and CLI (commit `be320b0`) are correct and stay intact; the bug is in `games/signal-dodge/scripts/main.gd`.

**Last commit:** `b96074e` docs(h5): record collision-propagation bug halting step 2b

**Current task:** Fix the GDScript collision-propagation bug so that hazard-physics-tick deaths reach Python as `terminated=True` on the next step reply.

**Next action:** Author the bug-fix slice for `games/signal-dodge/scripts/main.gd`. Per `docs/h5-collision-propagation-bug.md` section "Suggested fix paths", path (1) is the cleanest: replace the unconditional `_h3_step_terminated = false` start-of-step reset (line ~417) with a reset gated on episode_start/reset only, so a between-step collision survives to be consumed by the next step's reply. Add a unit test against a fake transport that fires `_on_player_died` between two step requests and asserts the next reply carries `terminated=True` and `terminal_reason="collision"`. Re-run the same H4 pixel 2-seed negative-control smoke (`python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --run-id h5_negative_controls_smoke_postfix --seeds 1000,1001 --mode negative-controls`) and verify at least one negative control now shows `mean_episode_length < 0.80 * max_steps`. After the fix, the original Step 2B saturation question can be re-asked truthfully; profile hardening is a downstream decision, NOT this slice. H3 same-seed reproducibility tests must still pass post-fix.

**Blockers:** none for the bug-fix slice. Godot CLI launch works (Godot 4.6.2 at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine...\Godot_v4.6.2-stable_win64.exe`); Desktop Commander `.bat` runner pattern works; full live pipeline confirmed end-to-end on StrongerJr.

**Notes:**

- Diagnostic from `runs/rl/signal_dodge_ppo_h4_pixel/h5_negative_controls_smoke/`: Godot logged 120-121 spawns and 14-18 collisions and 14-18 deaths PER POLICY across 3600 frames. Python NDJSON shows `terminated=false, truncated=false, terminal_reason=""` for all 3600 step events per policy. Python `episode_id` stays `ep-000001` for the entire first seed's rollout because no terminated reply ever arrives. The Step 2A-lite saturation result was a measurement artifact of the bug, not a real profile-difficulty signal.
- Root cause: `main.gd` line 417 unconditionally resets `_h3_step_terminated = false` at the start of every `_h3_perform_step`. `_on_player_died` correctly sets the flag true on collision (line 226), but if the collision fires on a hazard physics tick BETWEEN Python step requests, the next step's start-of-step reset wipes the signal before line 435 reads it. The design intent comment on line 432 ("synchronously inside the move_action() call below") is wrong about when collisions can fire.
- Step 2B work NOT done this session: no edits to `godot_config.py`, `factories.py`, `godot_env.py`, `main.gd`, `hazard.gd`, or `constants.py`. No `configs/rl/signal_dodge_ppo_h5_hard_pixel.yaml`. Pre-existing harness and CLI tests still pass (286 / 2 deselected on tests/rl as of `a3d3b82`); no test churn this session.
- Profile-hardening hypothesis from the Step 2A-lite handoff is NOT yet refuted; it just cannot be evaluated until the propagation bug is fixed and a truthful re-smoke runs. The new H5 hard YAML and the spawn_interval_frames/hazard_speed/hazards_per_spawn parameterization remain reasonable downstream work, just not the immediate next action.
- Two pieces of the original Step 2B prompt worth preserving for the downstream tuning slice: (a) one knob at a time so isolated effect can be measured, and (b) any Python↔GDScript constant added must have a drift-detection test so the two cannot fall out of sync silently. The bug fix itself does not need new tunable knobs; it preserves existing H4 defaults.


---
