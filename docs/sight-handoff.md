# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 deadly_corridor. Entropy-collapse fix CONFIRMED (reached 500k, past the 375k freeze point). Wall is now an UNDIAGNOSED silent mass kill of every python process. Long jobs moved to Task Scheduler; attempt 4 resumed from 500k and live.

**Last commit:** cf779df vzd: correct the crash diagnosis; move long jobs to Task Scheduler

**Current task:** Attempt 3 reached 500,000 steps, deepest ever for this scenario, with the reward-scale fix holding (HIGH). Then every python process on the box died silently at 17:07:47 on 2026-07-15: trainer, its workers, the supervisor, AND the monitor (which shares nothing with the training tree). **The 3db4777 diagnosis was wrong and is corrected in cf779df.** Evidence against "stochastic ViZDoom engine death": zero `[SafeDoom]` rebuilds in 500k steps; no worker-side traceback (the worker vanished rather than raised, so a native access violation is possible and SafeDoom's `except` cannot see that); the supervisor backstop died too. Ruled out (verified this session): Claude Desktop / MCP restart (up since 07:45), sleep/wake (no Kernel-Power events, wake count 0), Application-level errors (none), vizdoom crash dumps (none). Root cause UNKNOWN (LOW confidence in every candidate). `nhi` (Thunderbolt) 9007/9008 events bracket the death at 16:59 and 17:08 but are not decisive. SafeDoom is kept (cheap, verified on its own path) but is no longer claimed to fix this.

**Next action:** Poll `runs\vzd\ppo_deadly_corridor_s3_shaped\supervisor.log` (timestamped) and `SUP_HEARTBEAT`. **If it dies again, the new instrumentation is the whole point: a stalled heartbeat with NO "leg N exited rc=" line means the supervisor was KILLED, not that the trainer crashed and the loop failed. That single bit decides the next move.** Killed -> the kill is external and system-wide, so investigate Defender/antivirus, MSI vendor software, and power/Thunderbolt (`nhi`) before touching RL code. Crashed-and-not-restarted -> the supervisor loop is at fault, fix it. Target 1.5M total, ~120 fps, so ~2.3h from 17:44. On DONE, read `summary.json`: bar is raw-scenario IQM decisively above 683.9 (same skill, same raw eval). **Watch entropy**: -0.943 at 520k vs -1.26 at 290k, drifting toward collapse though approx_kl 0.0043 / clip_fraction 0.125 say the policy still moves. If it collapses, escalate cheapest-first: vf_coef 0.5 to 0.25, then raise `--ent-coef`, then `share_features_extractor=False`.

**Blockers:** None requiring Jeff.

**Notes:**

- **Long jobs are Task Scheduler tasks now, not detached children.** `schtasks /Run /TN Sight-VZD3` (training, via `runs\vzd\_run_s3_shaped_task.cmd`) and `schtasks /Run /TN Sight-Monitor` (monitor, via `runs\_run_monitor_task.cmd`). Detached children died 3x for the monitor and 3x for training; FOUND-ART ADOPT, Task Scheduler is the built-in packaged answer (NSSM equivalent, adds a dependency). `/RL HIGHEST` needs elevation and was dropped; normal privilege is still OS-owned.
- Monitor verified 200 this session via the task. If it is down, `schtasks /Run /TN Sight-Monitor`, do not hand-launch a detached child.
- **Method lesson, cost 3 runs:** fault injection that exercises the path you built rather than the path that is failing produces confidence, not evidence. The `game.close()` test proved SafeDoom catches a clean engine exit; the real failure never raises a Python exception. Verify against the observed failure signature, not against the fix.
- Resume path is real and exercised: VecNormalize stats restored from the step-matched `.pkl` at 500k. Without `save_vecnormalize=True` a restart re-estimates the return std from 1.0 and re-inflates the returns that collapsed entropy, i.e. the restart would undo the fix it protects. Checkpoints every 50k (`--ckpt-every`).
- Wrapper order matters: SafeDoom must sit INSIDE ShapedCorridorReward so `unwrapped.game` resolves to the rebuilt engine. Shaping skips the rebuild discontinuity via `info["engine_fault"]`. Re-run `runs\vzd\_test_safedoom_fault.py` after any wrapper-order change.
- **Bar is weak, needs seeds not episodes:** 15 of 30 flat episodes were byte-identical at 664.1885 (the untrained smoke float), so IQM 683.9 is close to "untrained policy wanders". Deterministic eval on a deterministic map measures which mode the policy landed in, not a distribution, so IQM over 30 episodes is near-single-sample. A decisive corridor claim needs MULTIPLE SEEDS. Do not over-read a single clear.
- vizdoom 1.3.0, sb3 2.8.0, Python 3.14 (both verified). Game vars: `unwrapped.game.get_game_variable(GV.X)` reads ANY variable regardless of cfg. HITCOUNT/DAMAGE_TAKEN/KILLCOUNT/AMMO2 work; SELECTED_WEAPON_AMMO reads -1, unusable.
- Skill 1 is NOT a useful eval point (untrained evals ~2280). Skill 3 keeps the eval comparable to the flat 683.9. Eval is deliberately RAW so numbers compare across runs; shaped ep_rew_mean ~-1000 is NOT comparable to the flat curve.
- `--resume` makes `--steps` ADDITIONAL; the supervisor owns the total via `--target`. `runs\` is gitignored, so launchers/helpers there are on-disk only. Attempt 2's log archived at `runs\vzd\_archive_s3_shaped_attempt2_train.log`. Full story in `docs\vzd-deadly-corridor-findings.md`.
