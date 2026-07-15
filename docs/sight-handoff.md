# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 deadly_corridor. Flat reward has now FAILED TWICE (skill 5, skill 3). Root cause found and fixed; shaped+normalized run in flight.

**Last commit:** see `git log -1`. Prior: f19996d docs: VZD-3 deadly_corridor findings.

**Current task:** `runs\vzd\ppo_deadly_corridor_s3_shaped` is training (pid 46604 at launch, detached, ~120 fps, 1.5M steps, ETA ~3.5h from 23:30 CDT 2026-07-14). Cmd: `tools\vzd_ppo_train.py --env-id VizdoomDeadlyCorridor-v1 --doom-skill 3 --steps 1500000 --shape-reward --norm-reward --out runs\vzd\ppo_deadly_corridor_s3_shaped`. Log `runs\vzd\ppo_deadly_corridor_s3_shaped_train.log`.

**THE LOAD-BEARING FINDING (this session):** both flat runs failed from *entropy collapse driven by reward scale*, not from the reward landscape. `entropy_loss` -2.07 -> ~0 by the first quarter, `value_loss` ~5.5e4, `approx_kl` and `clip_fraction` pinned at 0.0 for the back half: the policy became a point mass by ~300k steps and the remaining ~1.2M steps were wasted on a frozen policy. Mechanism: corridor reward is ~1000x defend's scale -> value_loss ~5e4 -> and SB3 CnnPolicy SHARES the features extractor between value and policy heads, so value gradients (x vf_coef 0.5) swamp the entropy bonus (ent_coef 0.01 x ~2.0) and saturate the trunk. This is the project's own "normalize first" rule; the early blocks show the exact signature (kl 0.32/1.19/0.84, clip_frac 0.77/0.80) before freezing. **Shaping alone would also have failed.** Full writeup: `docs\vzd-deadly-corridor-findings.md`.

**Verified fix (HIGH confidence on mechanism, MEDIUM on outcome):** with `--norm-reward --shape-reward` at ~47k steps, value_loss 5.5e4 -> **0.32**, entropy_loss -2.08 -> -1.89 (still exploring), approx_kl **0.001-0.003**, clip_fraction **0.11**. The policy is still exploring where the flat runs had already frozen.

**Next action:** Re-check entropy at 300-400k in the shaped run (the point where flat had already frozen): `.venv-c1\Scripts\python.exe runs\vzd\_parse_fields.py runs\vzd\ppo_deadly_corridor_s3_shaped_train.log`. If entropy_loss is still well below -1.0 and approx_kl > 0, let it finish. If it collapsed again, the next single change is raising `--ent-coef` (flag now exists, default 0.01), NOT more shaping. On DONE, read `runs\vzd\ppo_deadly_corridor_s3_shaped\summary.json`: **bar is raw-scenario IQM decisively above 683.9** (the flat skill-3 IQM, same skill, same raw eval). If it passes, stage 2 is resume-finetune at skill 5 (`--resume runs\vzd\ppo_deadly_corridor_s3_shaped\model.zip --doom-skill 5 --out runs\vzd\ppo_deadly_corridor_s5ft`; `--steps` is ADDITIONAL, verify from log). Owed: a 30-episode UNTRAINED skill-3 anchor (the ~767 in prior handoffs came from a 3-episode smoke and is not a valid IQM comparator).

**Blockers:** None requiring Jeff.

**Notes:**

- Monitor: http://127.0.0.1:8791/monitor.html (pid 6364), serves `runs\vzd` root, pointed at the s3_shaped run. FIXED this session: `runs\_launch_monitor_vzd_root.py` was missing `DETACHED_PROCESS`, so the server died whenever the spawning console went away (it had died twice). Flag added; relaunch with `.venv-c1\Scripts\python.exe runs\_launch_monitor_vzd_root.py`.
- Stage-1 flat s3 result (FAILED): mean 891.1 / **IQM 683.9**, 30 eps. Bar was IQM above 93.6 AND ~767; 683.9 < 767. Bimodal: 15/30 byte-identical at 664.1885223388672 (the SAME float the untrained smoke produced, i.e. the dominant mode never moved) and 4/30 at ~2280 (armor reached).
- vizdoom is **1.3.0**, not 1.2 as older notes said.
- Game vars VERIFIED on 1.3.0: `env.unwrapped.game.get_game_variable(GV.X)` reads ANY variable regardless of what the cfg declares (deadly_corridor.cfg declares HEALTH only). HITCOUNT / DAMAGE_TAKEN / DAMAGECOUNT / KILLCOUNT / AMMO2 all work. **SELECTED_WEAPON_AMMO reads -1 and is unusable; use AMMO2.**
- Skill 1 is trivial and NOT a useful eval point: a 2k-step untrained policy evals ~2280 at skill 1 (walks to the vest unopposed). This is why the old "skill 1 + shaping" plan was dropped.
- `--resume` semantics CONFIRMED: `reset_num_timesteps=False` makes `--steps` ADDITIONAL, not a total. The argparse help used to say the opposite; fixed.
- Shaped train reward is not comparable to the flat curve: ep_rew_mean ~-1000 because damage_taken*10 charges ~-1000 for a full-health death. Eval stays RAW (unshaped, unnormalized) on purpose so it compares to 683.9 / 93.6.
- `runs\vzd\_parse_corridor_log.py` used to hardcode the OLD skill-5 log path and ignore its argument, which produced a false "collapse" reading this session. Fixed to honor `sys.argv[1]`. `runs\vzd\_parse_fields.py` (new) dumps entropy/value_loss/kl/clip trajectories, which is what found the real bug.
- `runs\` is gitignored: helper scripts there are on-disk only, not tracked.
- DC relay failure mode: relay can error while the device still executes; verify via `~/.claude-server-commander/tool-history.jsonl`. Registry last_seen unreliable.
- DC transport dies on blocking calls >= ~4 min; use `*>` file logging and instant reads. Multi-line `python -c` through `interact_with_process` breaks the shell into continuation mode: write a probe script to disk instead.
- deadly_corridor ground truth: VizdoomDeadlyCorridor-v1, cfg skill 5, death_penalty 100, no living_reward, WAD distance shaping toward armor, Discrete(8), timeout 2100 tics.
