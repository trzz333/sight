# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-2 complete (teacher-to-student distillation on defend_the_center; VZD-1 teacher and Godot Signal Dodge imitation both stand)

**Last commit:** 9d469bf vzd: VZD-2 teacher-to-student distillation, student matches teacher

**Current task:** None in flight. VZD-2 shipped end to end this session. tools\vzd_rollout_dataset.py rolls the PPO teacher out in its exact training env chain and writes a BC dataset schema-identical to the LMP extractor (student frames mean-gray stride-2 120x160 of the 240x320 RGB screen, labels from game.get_last_action). 100 episodes, 20469 samples, teacher rollout mean 12.55 / IQM 13.26. vzd_bc_eval.py gained --obs rgb2 so student eval matches the dataset derivation. Student (93.4% val action match) evals at mean 12.57 / IQM 13.31 over 30 eps vs teacher of record 12.17 / 12.75. Distillation lossless within noise; both near the 15-kill scenario ceiling. README results table and methods line updated.

**Next action:** VZD-3: bootstrap deadly_corridor (scenario with visible movement) for a better public clip. Reuse the VZD-1 recipe: vzd_ppo_train.py generalized to take --env-id, train the PPO teacher (gamma 0.99, gray stride obs, skip 4, stack 4), eval, record a clip with vzd_ppo_watch.py. deadly_corridor is known-harder (corridor navigation, reward shaping may be needed); if the flat recipe stalls, the pre-registered lever is the scenario's living_reward/death penalty config, not more steps. After VZD-3: the ammo-efficiency reward-shaping experiment on defend_the_center.

**Blockers:** None requiring Jeff.

**Notes:**

- Pairing rule: a student trained from teacher rollouts MUST be evaluated with --obs rgb2. The native GRAY8 render is a different pixel distribution; mixing them invalidates the comparison. Human-demo students stay on native.
- Desktop Commander transport dies on any blocking tool call at/over ~4 minutes, hard ceiling, worse under ViZDoom load. It self-recovers and background processes survive. Launch with file logging (*> redirect), poll with instant reads only, never block on the pipe.
- defend_the_center has no movement actions by design (turn/attack only); stationary clips are the scenario ceiling, not a policy defect. deadly_corridor is the queued fix for visible navigation footage.
- ppo_defend\summary.json recipe string says gray84x112; the model artifact is the authority: obs space (4,60,80), matches current trainer source. Stale label only, numbers in it are good.
- Monitor for vzd runs: runs\vzd\ppo_defend\monitor.html on 127.0.0.1:8791 via runs\_launch_monitor_vzd.py (one server at a time on 8791). PowerShell *> logs are UTF-16; read_file renders them with spacing artifacts but readable.
