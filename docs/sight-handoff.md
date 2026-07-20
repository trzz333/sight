# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 seed replication (seed 1 of 2). Seed-0 pipeline COMPLETE and combat-verified; seed-1 s3 leg PASSED; seed-1 s5 finetune IN FLIGHT. Machine: MSI Raider 18 HX A14VHG (hostname MSI, RTX 4080 Laptop 12GB) as of 2026-07-19; StrongerJr retired as the Sight box.

**Last commit:** 66cd638 infra: no-popup launch pattern (pythonw + launch_hidden.py, CREATE_NO_WINDOW) for all scheduled tasks; retire raw-IP monitor URLs for hostname http://MSI:8791 after DHCP drift broke Jeff's link

**Current task:** Seed-1 replication of the s3->s5 pipeline is mid-flight. The s3 shaped+norm leg (seed 1, 0->1.5M, Task Scheduler job Sight-VZD3-Seed1) finished clean at 1,501,184 steps in 14,623s (102.6 steps/s) and PASSED: raw skill-3 eval over 30 episodes, mean 2277.36, IQM 2276.58 vs pre-registered bar ~683.9, from `runs\vzd\ppo_deadly_corridor_s3_shaped_seed1\summary.json`. Seed-1 s3 IQM is within ~3 points of seed-0's skill-5 IQM 2279.67, consistent with the corridor's distance-dominated reward ceiling. The s5 resume-finetune (job Sight-VZD3-S5-Seed1, out `runs\vzd\ppo_deadly_corridor_s5_ft_seed1`, seeded from the numbered 1.5M checkpoint + step-matched VecNormalize pkl, --ent-coef 0.05, --seed 1, target 3.0M) is running; supervisor confirmed resume from `ppo_deadly_corridor_1500000_steps.zip`. Expected ~4h at the measured rate. Monitor: http://MSI:8791/monitor_s5_seed1.html.

**Next action:** When Sight-VZD3-S5-Seed1 finishes (check `runs\vzd\ppo_deadly_corridor_s5_ft_seed1\SUP_HEARTBEAT` for "finished clean"), verify `summary.json` raw skill-5 IQM decisively above 93.6, then run the combat probe (`tools\vzd_probe_combat.py` against the s5_ft_seed1 dir, launched detached from DC per the ~50s rule). On probe FIGHT verdict, launch seed 2: copy `_run_s3_shaped_seed1_task.cmd` to a seed2 variant (--seed 2, seed2 out dir and log), register Sight-VZD3-Seed2 via the launch_hidden pattern, run, then its s5 finetune and probe. Seed 2 completing the full pipeline is the bar for the README transfer claim.

**Blockers:** None requiring Jeff.

**Notes:**

- **MONITOR + NO-POPUP RULE (Jeff has asked 5x):** every scheduled task and autostart launches via `pythonw.exe tools\launch_hidden.py <cmd>` (CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP inherited by legs and the doom engine); never point a task straight at a .cmd (visible Windows Terminal tab). Monitor URLs are hostname-based (http://MSI:8791), never raw IPs (DHCP drift broke .108). Serving stack: `tools\serve_monitor.py` (ThreadingHTTPServer, logging suppressed) via schtasks Sight-Monitor + Startup `sight-monitor.vbs`. Verify serving by HTTP-vs-disk byte equality, never status 200.
- **DC session reaping:** keep every Desktop Commander call's wait under ~50s (MCP ~60s timeout -32001 reaps sessions and kills child trees; killed a probe at ep 29/30 once). Anything over ~1 min runs detached; verify by output files, not process status.
- **Long jobs = Task Scheduler, HIGH (4 clean runs: s3, s5 ft, seed1 s3, seed1 s5 launch).** Stalled `SUP_HEARTBEAT` with no "leg N exited rc=" line in `supervisor.log` proves killed, not crashed.
- **Resume mechanics:** `--ent-coef` reapplied after `PPO.load` (silent no-op before fb7ff99). Seed resume dirs from the numbered `_<N>_steps.zip` + step-matched VecNormalize pkl, never `model.zip`.
- **Standing caveats:** README transfer claim gated on seed 2. `combat_probe.json` SHOTS_FIRED/accuracy UNRELIABLE (stale AMMO2 baseline), KILLCOUNT/HITCOUNT/DAMAGE_TAKEN clean. Eval is RAW and cross-run comparable; shaped ep_rew_mean is not. vizdoom 1.3.0, sb3 2.8.0, Python 3.14. `runs\` gitignored (cmds, monitor pages, probe JSONs, demos are disk-only).
