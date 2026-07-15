# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 deadly_corridor. Entropy-collapse fix CONFIRMED WORKING. Wall moved from optimization to infra (ViZDoom engine deaths); crash tolerance shipped and verified; attempt 3 in flight under a supervisor.

**Last commit:** 3db4777 vzd: make corridor training survive ViZDoom engine deaths

**Current task:** The reward-scale fix is no longer a hypothesis. Attempt 2 (`runs\vzd\_archive_s3_shaped_attempt2_train.log`, parsed this session) ran 290,816 steps with entropy_loss -2.08 to -1.26 (flat runs hit ~0 by the first quarter), approx_kl 0.001 to 0.004, clip_fraction 0.10, explained_variance 0.003 to 0.51, ep_rew_mean -1160 to -262 monotone. The policy was live and learning the whole way (HIGH). It died on infra: a ViZDoom worker vanished and SubprocVecEnv EOF'd (`BrokenPipeError` WinError 109 -> `EOFError`), killing the run. Second crash (~47k, ~290k), so the method changed rather than the retry count. Shipped: `SafeDoom` env wrapper (catches ViZDoom's own exception types, rebuilds the engine in-process, ends the episode) plus `tools\vzd_supervise.py` (relaunches from newest checkpoint; `--target` is a TOTAL; bails after 3 legs with no checkpoint progress so a deterministic crash gets diagnosed not retried). Checkpoints now every 50k with `save_vecnormalize=True`, and resume restores the return stats matched by step count. Attempt 3 launched fresh (the 250k checkpoint predates save_vecnormalize and has no return stats; resuming without them re-inflates the returns whose scale collapsed entropy). Kept `--n-envs 8` against the pre-registered cut to 4: the cut lowered crash rate, SafeDoom removes the consequence.

**Next action:** Poll `runs\vzd\ppo_deadly_corridor_s3_shaped_train.log`. Supervisor pid 32164 at launch, target 1.5M total, ~118 fps, so ETA ~3.6h from 2026-07-15 ~01:15. Healthy at 8k: value_loss 0.29, entropy_loss -2.07, approx_kl 0.0034 (MEDIUM that it stays healthy past the 375k mark where the flat runs froze; no shaped run has yet reached it). Read entropy across the run via `.venv-c1\Scripts\python.exe runs\vzd\_parse_fields.py runs\vzd\ppo_deadly_corridor_s3_shaped_train.log`; grep `[SafeDoom]` and `[sup]` to count engine deaths survived. On DONE, read `runs\vzd\ppo_deadly_corridor_s3_shaped\summary.json`: bar is raw-scenario IQM decisively above 683.9 (same skill, same raw eval). If entropy collapses again despite normalization, escalate cheapest-first per the found-art ladder: vf_coef 0.5 to 0.25, then raise `--ent-coef`, then `share_features_extractor=False`. If it clears the bar, the corridor teacher is done: record navigation footage, then the ammo-efficiency shaping experiment.

**Blockers:** None requiring Jeff.

**Notes:**

- ViZDoom engine death is a known, decade-old, unfixed upstream pathology (Farama-Foundation/ViZDoom#169: dies stochastically under multi-instance load, rate scales with instance count; #430: the exception only means "the binary died"). Do not try to prevent it. Detect and rebuild. SB3's SubprocVecEnv has no crash recovery by design.
- SafeDoom's rebuild path is verified by fault injection, not by hope: `runs\vzd\_test_safedoom_fault.py` closes the engine under a live env and asserts one rebuild plus a clean 20-step rollout after. Re-run it after any wrapper-order change.
- Wrapper order matters: SafeDoom must sit INSIDE ShapedCorridorReward so `unwrapped.game` resolves to the rebuilt engine. Shaping skips the rebuild discontinuity via `info["engine_fault"]`.
- Monitor http://127.0.0.1:8791/monitor.html (pid 31548 this session, verified 200) serves `runs\vzd`. It has now died twice between sessions; treat uptime as unreliable and relaunch on resume via `runs\_launch_monitor_vzd_root.py`.
- vizdoom is 1.3.0, sb3 is 2.8.0 (both verified this session). Game vars: `env.unwrapped.game.get_game_variable(GV.X)` reads ANY variable regardless of cfg. HITCOUNT/DAMAGE_TAKEN/KILLCOUNT/AMMO2 work; SELECTED_WEAPON_AMMO reads -1 and is unusable.
- Skill 1 is NOT a useful eval point: an untrained policy evals ~2280 there. Skill 3 keeps the eval comparable to the flat 683.9.
- Eval is deliberately RAW (unshaped, unnormalized) so numbers compare across runs. Shaped ep_rew_mean ~-1000 is NOT comparable to the flat curve.
- `--resume` makes `--steps` ADDITIONAL, not a total; the supervisor is what owns the total. `runs\` is gitignored, so helper and launcher scripts there are on-disk only. Full run story and found-art writeup live in `docs\vzd-deadly-corridor-findings.md`.
