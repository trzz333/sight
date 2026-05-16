# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 paused for reward-amendment proposal. Behavior audit closed; Phase G NOT triggered. No new training approved.

**Last commit on HEAD:** `53223b7` docs(h5): behavior audit evidence for state seed 2 and Phase E seed 2 (chore-refresh follows on push).

**Substantive code/evidence commit:** `53223b7` docs(h5): behavior audit evidence for state seed 2 and Phase E seed 2.

**Current task:** Pause-and-reframe behavior audit complete. Per-step `godot.ndjson` traces from the latest state-comparator seed 2 eval and a Phase E seed 2 pixel eval were classified without retraining. Result: both models converge on the same policy. Phase E `CnnPolicy` outputs `action=-1` 100% of the time across all three representative episodes. State `MlpPolicy` outputs ~78-81% left, ~19-22% right, ~0% stay. Both end at `x=16` (left wall clamp) with `>90%` wall_hug_ratio. Cross-model determinism on identical eval seeds (same lengths on seeds 1000, 1004, 1008) shows the dominant determinant of episode length is the seed (hazard spawn pattern), not the policy. Failure-mode distribution across the 20 representative-model episodes: `wall_hugging_into_collision` 17, `high_frequency_oscillation_into_collision` 1, `survived_to_timeout (still wall-hugging)` 2. Audit closes the diagnostic question: the `+1/step` survival reward has a gradient fixed point at the left wall; the bad in-game behavior is the gradient's correct answer to this reward. Encoder and observation channel are ruled out as the H5 blocker. Training-budget-only is ruled out as the H5 fix. Reward is the proximate cause. Evidence: `docs/h5-behavior-audit-evidence.md`. Audit script: `scripts/h5_behavior_audit.py` (uses existing artifacts only; no training, no eval runs).

**Next action:** GPT to draft an H5 plan section 7 charter amendment proposing a reward-shaping change (distance-bonus, approach-penalty, or lateral-utility-bonus variants are sketched in the evidence doc). No further training sweeps and no further hyperparameter slices until that amendment is on the table for Jeff approval. Claude does not pick a reward shape and does not run more experiments inside the current reward; pause is in effect.

**Blockers:** Pre-amendment lockout. No training is approved while the reward shape is the standing prime suspect. Operational consequence: Phase G remains NOT triggered and the next-experiment-lever question remains open at the charter level, not the operational level. Jeff and GPT decide whether to amend H5 to permit reward shaping or to revise H5 scope.

**Notes:**

- Reward gradient mechanism documented in `docs/h5-behavior-audit-evidence.md` Synthesis section: clamping at `x=16` makes `action=-1` locally stable; survival-only reward has no per-step signal to distinguish skillful avoidance from a lucky safe position; gradient descent therefore punishes exploration away from the wall. Same fixed point for `MlpPolicy` on state and `CnnPolicy` on pixels, which is why the perception-axis levers (frame-stack, state comparator) all failed.
- Secondary Godot-side observation: `godot.ndjson` from eval runs emits two `episode_start` events per `controller_reset_received`, producing alternating empty-episode shells in the trace. The audit script filters by requiring `steps > 0` before assigning eval seeds. No effect on prior eval summaries (those use Python-side `episodes.ndjson` aggregated on terminate/truncate). Cleanup target for `games/signal-dodge/scripts/main.gd` next time eval logging is revisited. Not urgent.
- H5 pre-training non-saturation gate convention reminder: evaluates three negative controls only (stay_only, seeded_random, untrained_cnn). `trained_cnn` does not exist until a training slice produces a `model.zip`. The inherited handoff-precision test in `tests/rl/test_h5_baseline_cli.py` continues to assert this phrasing is preserved here.
- Operational lesson from this session: `start "" /B cmd /c <bat>` background-launch pattern failed silently on this host once (Python alive 3+ minutes with zero stdout, no run dir). Reliable pattern is inline `interact_with_process` invocation against a persistent `cmd.exe`, accepting the MCP 4-minute false-timeout error, and recovering output via subsequent `read_process_output`. Persistent shell PID can also disappear between turns; re-spawning and re-exporting `SIGHT_GODOT_EXE` + `PYTHONUNBUFFERED=1` is the default on resume.
- Handoff convention reminder: `Last commit on HEAD` and `Substantive code/evidence commit` may temporarily lag during a chore-refresh push; resume by running `git log --oneline -5` before claiming HEAD.
