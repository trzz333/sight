# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 curriculum stage 1 in flight (deadly_corridor skill-3 PPO; skill-5 flat baseline complete and FAILED)

**Last commit:** 301a529 chore: refresh handoff, VZD-3 deadly_corridor baseline run live at 715028f

**Current task:** Skill-5 flat baseline VERDICT: policy collapse, not reward-scale instability. Eval mean 130.5 / IQM 93.6 over 30 eps; 14/30 episodes identical 94.95 (deterministic sprint-and-die). Training forensics (runs\vzd\_parse_corridor_log.py): ep_len_mean pinned ~14 agent steps (~1.7s) all run, approx_kl and clip_fraction collapsed to 0.0 early and stayed, explained_variance fine (~0.6). Diagnosis: premature convergence to sprint-forward local optimum at skill 5, gradients vanished. Reward normalization demoted (no instability present). Failure clip recorded: runs\vzd\ppo_deadly_corridor\gameplay_fail_s5.mp4 (11.9s, before/after material). Baseline artifacts preserved in runs\vzd\ppo_deadly_corridor (evidence, do not overwrite). CURRICULUM STAGE 1 LAUNCHED 2026-07-12 ~15:20 CDT: --doom-skill 3 --steps 1500000 --out runs\vzd\ppo_deadly_corridor_s3, log runs\vzd\ppo_deadly_corridor_s3_train.log, ~4.3h at ~97 fps. Start entropy -2.07 (healthy, near-uniform).

**Next action:** When DONE appears in runs\vzd\ppo_deadly_corridor_s3: bar is IQM decisively above BOTH the skill-5 collapse (93.6) and the untrained-skill-3 smoke (~767), with ep_len_mean well above 14. If passed: stage 2 = resume the s3 checkpoint at skill 5 (vzd_ppo_train.py --resume runs\vzd\ppo_deadly_corridor_s3\model.zip --doom-skill 5 --out runs\vzd\ppo_deadly_corridor_s5ft; NOTE --resume treats --steps as ADDITIONAL steps, verify from log), then eval, clip, README results table with the fail->curriculum narrative. If stage 1 fails at skill 3: method has now failed twice flat, next is skill 1 AND revisit ent_coef/exploration, not a retry.

**Blockers:** None requiring Jeff.

**Notes:**

- Monitor: http://127.0.0.1:8791/monitor.html re-pointed to the s3 run (serves runs\vzd root, server pid 33916).
- DC relay failure mode: relay can error ("No approval received"/"No devices available") while the device still executes; verified via tool-history.jsonl. Fallback: DC write-only with *> redirects, read via Filesystem MCP. Registry last_seen unreliable.
- vzd_ppo_watch.py output is block-buffered under *> redirect (no -u); log lands at exit, poll for the artifact instead.
- deadly_corridor ground truth: VizdoomDeadlyCorridor-v1, cfg skill 5, death_penalty 100, no living_reward, WAD distance shaping, Discrete(8), timeout 2100 tics. Skill-3 untrained ~767.
- DC transport dies on blocking calls >= ~4 min; *> file logging, instant reads only. PowerShell *> logs are UTF-16 (monitor.html decodes; _parse_corridor_log.py decodes).
