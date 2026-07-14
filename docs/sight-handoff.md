# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 curriculum stage 1 in flight (deadly_corridor skill-3 PPO; skill-5 flat baseline complete and FAILED)

**Last commit:** 715028f vzd: generalize PPO trainer/watcher to --env-id and --doom-skill for VZD-3

**Current task:** Stage 1 (skill 3, raw scenario reward, 1.5M steps) is running: at audit time step 38,912, ep_rew_mean 25.5, ep_len_mean 45.6 vs the 14 the skill-5 collapse was pinned at. Out runs\vzd\ppo_deadly_corridor_s3, log runs\vzd\ppo_deadly_corridor_s3_train.log, ETA ~4h from 15:20 CDT launch. Skill-5 flat baseline VERDICT stands: mean 130.5 / IQM 93.6, 14/30 identical episodes, kl and clip_fraction collapsed to 0 early, sprint-and-die local optimum; failure clip runs\vzd\ppo_deadly_corridor\gameplay_fail_s5.mp4 (15.3MB), baseline dir preserved as evidence. FOUND-ART this session (web search "ViZDoom deadly corridor PPO doom_skill curriculum training"): verdict ADAPT. Published recipe is curriculum PLUS game-variable reward shaping, not curriculum alone. Khan 2025 (Computer Animation & Virtual Worlds) reports deadly_corridor learnable through level 5 with both. nicknochnack/DoomReinforcementLearning uses s1..s5 cfg curriculum plus reward = movement + damage_taken_delta*10 + hitcount_delta*200 + ammo_delta*5. Exploration bonuses (RND: callumhay/vizdoom_ppo_rnd; ICM: mehdiboubnan) are the structurally different family behind that. Known risk the prior art exposes: skill-3 success can be combat-free (run to vest), and skill 5 requires killing the first pair, so curriculum-alone transfer may fail at stage 2.

**Next action:** When DONE appears in runs\vzd\ppo_deadly_corridor_s3: eval bar is IQM decisively above both 93.6 (skill-5 collapse) and ~767 (untrained skill-3 smoke), ep_len_mean well above 14. If passed: stage 2 = resume at skill 5 (--resume runs\vzd\ppo_deadly_corridor_s3\model.zip --doom-skill 5 --out runs\vzd\ppo_deadly_corridor_s5ft; --resume treats --steps as ADDITIONAL, verify from log). If stage 2 fails: next single change is the Renotte-style game-variable shaping wrapper (HEALTH already in cfg; verify HITCOUNT/ammo variable names at implementation), coefficients per the cited notebook. After that: RND/ICM exploration bonus. If stage 1 itself fails: skill 1 plus the shaping wrapper together, flat has then failed twice.

**Blockers:** None requiring Jeff.

**Notes:**

- Monitor: http://127.0.0.1:8791/monitor.html serves runs\vzd root (server pid 33916), pointed at the s3 run, decodes UTF-16 PowerShell logs.
- DC relay failure mode: relay can error ("No approval received"/"No devices available") while the device still executes; verified via ~/.claude-server-commander/tool-history.jsonl. Fallback: DC write-only with *> redirects, read via Filesystem MCP. Registry last_seen unreliable.
- vzd_ppo_watch.py output is block-buffered under *> redirect; poll for the artifact, not the log.
- deadly_corridor ground truth: VizdoomDeadlyCorridor-v1, cfg skill 5, death_penalty 100, no living_reward, WAD distance shaping toward armor, Discrete(8), timeout 2100 tics. Untrained skill-3 eval ~767.
- DC transport dies on blocking calls >= ~4 min; *> file logging, instant reads only. runs\vzd\_parse_corridor_log.py parses UTF-16 SB3 logs.
