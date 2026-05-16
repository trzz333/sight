# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 state-comparator slice complete; Phase G NOT triggered. Diagnostic-not-selection negative result.

**Last commit on HEAD:** to be refreshed in chore commit (see "Next action").

**Substantive code/evidence commit:** to be set in this session's commit closing the slice (`docs(h5)`-prefixed evidence commit).

**Current task:** H5 state-observation comparator slice closed. GPT-approved diagnostic-not-selection move tested whether PPO can learn Signal Dodge when perception is removed from the loop. Three 10k seeds trained against `configs/rl/signal_dodge_ppo_h5_state_comparator.yaml` (`MlpPolicy`, `obs_shape (10,)`, recipe inherited verbatim from Phase D/F), all `status=ok`, entropy stable at -1.07 to -1.10 across all 40 iterations of every seed (sharp contrast to Phase F seed 1 collapse at iteration 33). Full-mode eval over seeds 1000-1009 pooled across seeds 1, 2, 3 to reward 646.90, length 647.83, collision 0.933, timeout 0.067. State pooled FAILS every GPT success-bar criterion: reward 646.90 < 756.25; length 647.83 < 757.50; collision 0.933 > 0.80. Pooled is below every pixel comparator on reward/length (Phase E 764.87/765.80, Phase F 712.87/713.80) and tied with all pixel comparators on collision. Policy is non-degenerate (variance 183-1800, two full-survival timeouts) but does not avoid hazards better than stay_only. Evidence: `docs/h5-state-comparator-evidence.md`.

**Next action:** GPT to choose next experiment lever. The state-comparator result removes "pixel/perception is the H5 blocker" from the top of the lever queue. Standing evidence points toward reward dynamics or optimization at 10k as the proximate blocker. Open levers remaining: (a) extend timesteps past Phase D's 50k single-seed point with the state recipe to test budget alone; (b) raise `ent_coef` toward 0.05 under either state or frame-stack; (c) propose a reward/profile charter amendment per H5 plan section 7 (largest scope; requires docs-level amendment before any reward-shaping change); (d) augment state with hazard velocities to test the velocity-soundness narrowing called out in the evidence doc. Diagnostic-not-selection. No selection has been made. Per GPT plan and Jeff approval, Claude does not pick.

**Blockers:** None operational. State-comparator slice ran cleanly. One side note: `branch_metadata="ppo_cnnpolicy_loaded_from_disk"` in eval summaries is misleading for `MlpPolicy` runs but mechanically correct (SB3 dispatches via stored policy class). If the state comparator becomes a regular fixture, the eval CLI needs a non-CNN-baking label; this was not done in this slice and is not a current blocker.

**Notes:**

- Phase F frame-stack trigger ambiguity resolution from prior session held: "best frame-stack negative" is the per-metric strongest comparator (highest reward/length negative for reward/length thresholds, lowest-collision negative for collision threshold). Applied this session to the state-comparator success bar without controversy.
- State observation is position-only (player x, last action, three hazards as (x_offset, y_offset, present_flag) — slot 10 ceiling drops hazard 2's present flag). No velocities. A state-PPO failure does NOT falsify any "missing velocity" hypothesis at the pixel layer because state also lacks velocity. Recorded in the evidence doc; relevant if velocity-augmented state is selected as a future lever.
- Operational lesson: `start "" /B cmd /c <bat>` background pattern failed silently on this host (Python alive 3+ min with zero stdout, no run dir created). Reliable pattern is inline `interact_with_process` invocation against a persistent `cmd.exe`, accepting MCP's 4-minute false-timeout error, then recovering output via subsequent `read_process_output` calls. The persistent shell PID can also be lost between turns; spawning a fresh shell and re-exporting `SIGHT_GODOT_EXE` plus `PYTHONUNBUFFERED=1` should be the default on resume.
- Inherited handoff-precision test `test_handoff_does_not_describe_pre_training_gate_as_four_policy` (in `tests/rl/test_h5_baseline_cli.py`) asserts the handoff retains the wording "non-saturation" somewhere; the prior Phase F refresh accidentally dropped it and this rewrite restores it explicitly in this notes line and in the current-task description. The H5 pre-training non-saturation gate evaluates three negative controls only (stay_only, seeded_random, untrained_cnn); trained_cnn does not exist until after a training slice produces a model.zip.
- Handoff convention: `Last commit on HEAD` and `Substantive code/evidence commit` may temporarily lag during a chore-refresh push; resume by running `git log --oneline -5` before claiming HEAD.
