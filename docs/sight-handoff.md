# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-1 complete (ViZDoom defend_the_center PPO teacher trained, evaluated, packaged; Godot Signal Dodge closed with imitation as the standing solution)

**Last commit:** 09ec084 vzd: teacher results + resume packaging

**Current task:** None in flight. VZD teacher finished at 2.25M total steps (SB3 resume gotcha: reset_num_timesteps=False ADDS --steps to the checkpoint count; documented in the trainer docstring). Eval of record: mean 12.17 / IQM 12.75 kills per episode, 30 deterministic eps, rewards 7-15 with the 15 ceiling hit 13/30. 30s clip at runs\vzd\ppo_defend\gameplay.mp4 (640x480). Resume packaging shipped: README results table (every number re-verified against on-disk summaries this session, Spearman -0.19 and median 573->393 recomputed from the 1M/5M Godot summaries), gamma/critic-collapse story, docs\vzd-ppo-teacher-findings.md.

**Next action:** VZD-2: teacher-to-student distillation on defend_the_center. Roll out the trained teacher (runs\vzd\ppo_defend\model.zip) to generate a demo dataset via a rollout-extract tool (adapt vzd_extract_dataset.py to consume policy rollouts instead of LMP replays), train the BC student with vzd_bc_train.py, eval with vzd_bc_eval.py, report student-vs-teacher mean/IQM. Resume-legible framing: distillation pipeline, what survives compression. Follow-on arc after that: deadly_corridor scenario (visible movement for a better clip) and an ammo-efficiency reward-shaping experiment (Jeff's observation: 26 rounds, 12-15 kills, no ammo term in the reward, so no pressure to conserve).

**Blockers:** None requiring Jeff.

**Notes:**

- defend_the_center action space is turn-left / turn-right / attack only; the player is fixed at center by scenario design. Lack of movement in the clip is the scenario ceiling, not a policy defect. Visible navigation requires a different scenario (deadly_corridor queued).
- Monitor for the vzd run: runs\vzd\ppo_defend\monitor.html served detached on 127.0.0.1:8791 (relaunch via runs\_launch_monitor_vzd.py). Godot monitor server script remains at runs\_monitor_server.py (port collision: only one at a time on 8791).
- PowerShell here-string footgun: the closing '@ terminator must be alone at column 0; commands on the same line wedge the session in continuation mode. Commit messages go through a temp file written by write_file, never inline here-strings through interact_with_process.
- Run-death postmortem stands: never kill_process near a live training run. The one death this cycle was self-inflicted at 772k; resume + checkpoints salvaged it.
- Stale-sentinel rule stands: smoke runs take --out to a scratch dir.
