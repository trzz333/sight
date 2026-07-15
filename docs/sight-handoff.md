# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 deadly_corridor. Flat reward FAILED TWICE (skill 5, skill 3). Root cause identified and fixed; first shaped+normalized run crashed on infra, attempt 2 in flight.

**Last commit:** 7868b5a vzd: fix corridor entropy collapse (reward-scale). Add --shape-reward/--norm-reward/--ent-coef; flat s3 FAILED (IQM 683.9); shaped+norm run live

**Current task:** Flat skill-3 curriculum FAILED its pre-registered bar: mean 891.1, IQM 683.9 over 30 deterministic episodes (verified from `runs\vzd\ppo_deadly_corridor_s3\summary.json` this session), against a bar of IQM decisively above both 93.6 and ~767. 15 of 30 episodes are byte-identical at 664.1885223388672, the same float the untrained smoke produced, so training never moved the dominant mode. Root cause of both flat failures is entropy collapse from reward scale, not the reward landscape: entropy_loss -2.07 to ~0 by the first quarter, value_loss ~5.5e4, approx_kl and clip_fraction pinned at 0.0 for the back half, i.e. ~1.2M of 1.5M steps ran on a frozen policy. Mechanism is the shared CnnPolicy trunk carrying value-loss gradients into the policy (found-art ADOPT, Pop-Art's target-normalization/lower-layer-gradient equivalence). Fix shipped as `--norm-reward` (VecNormalize, clip 10) plus `--shape-reward`; at 47k steps it moved value_loss 5.5e4 to 0.32, entropy_loss to -1.89, approx_kl to 0.003, clip_fraction to 0.11. That first shaped run then CRASHED at ~47k with `ViZDoomUnexpectedExitException` (dead engine subprocess killed a SubprocVecEnv worker). Attempt 2 relaunched fresh, pid 21264, ~120 fps, 1.5M steps, out `runs\vzd\ppo_deadly_corridor_s3_shaped`. Outcome UNKNOWN: no shaped run has yet passed the ~375k mark where the flat runs froze.

**Next action:** Check `runs\vzd\ppo_deadly_corridor_s3_shaped_train.log` for a repeat `ViZDoomUnexpectedExitException` and for entropy at 300-400k via `.venv-c1\Scripts\python.exe runs\vzd\_parse_fields.py runs\vzd\ppo_deadly_corridor_s3_shaped_train.log`. If it crashed again, that is twice, so change the method: cut `--n-envs` to 4, add a supervisor that auto-resumes from the newest checkpoint, and remove the bare `except Exception` in `ShapedCorridorReward.step` that can hide the first engine error. If it is alive and entropy_loss is still below -1.0 with approx_kl above 0, let it finish, then read `runs\vzd\ppo_deadly_corridor_s3_shaped\summary.json`: bar is raw-scenario IQM decisively above 683.9 (same skill, same raw eval). If entropy collapsed again despite normalization, escalate cheapest-first per found-art: vf_coef 0.5 to 0.25, then raise `--ent-coef`, then `share_features_extractor=False`.

**Blockers:** None requiring Jeff.

**Notes:**

- Monitor http://127.0.0.1:8791/monitor.html (pid 45108) serves `runs\vzd`, pointed at the s3_shaped run. `runs\_launch_monitor_vzd_root.py` was missing `DETACHED_PROCESS` and died with its spawning console; flag added, but it has still died once since, so treat monitor uptime as unreliable and relaunch on resume.
- vizdoom is 1.3.0, not 1.2. Game vars verified: `env.unwrapped.game.get_game_variable(GV.X)` reads ANY variable regardless of cfg (deadly_corridor.cfg declares HEALTH only). HITCOUNT/DAMAGE_TAKEN/KILLCOUNT/AMMO2 work. SELECTED_WEAPON_AMMO reads -1 and is unusable; use AMMO2.
- Skill 1 is NOT a useful eval point: a 2k-step untrained policy evals ~2280 there (walks to the vest unopposed). This killed the old "skill 1 + shaping" plan. Skill 3 chosen so the eval stays comparable to the flat 683.9.
- Eval is deliberately RAW (unshaped, unnormalized) so numbers compare across runs. Shaped ep_rew_mean ~-1000 is NOT comparable to the flat curve, since damage_taken*10 charges ~-1000 per full-health death.
- `--resume` makes `--steps` ADDITIONAL, not a total (SB3 `reset_num_timesteps=False`); argparse help used to claim the opposite and is now fixed. `runs\` is gitignored, so helper scripts there are on-disk only. Full run story and the found-art writeup live in `docs\vzd-deadly-corridor-findings.md`.
