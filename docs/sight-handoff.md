# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 deadly_corridor COMPLETE. Both stages passed and combat-verified. Seeds owed before any README transfer claim.

**Last commit:** 8a94b76 vzd: skill-5 combat probe FIGHT (kills_mean 5.83, 30/30 with a kill, 1 death) - VZD-3 caveat resolved, findings 4b + results table

**Current task:** VZD-3 is done. The skill-5 combat probe (30 deterministic episodes, `runs\vzd\ppo_deadly_corridor_s5_ft\combat_probe.json`) resolved the fight-or-run-past caveat: verdict FIGHT, kills_mean 5.83, 30/30 episodes with a kill, 29/30 full clears at 6/6 kills and 6/6 hits, damage_taken_mean 58.8, 1 death (the same -115.9 episode as the eval). The probe reproduces the finetune eval IQM 2279.67 exactly from independent code, re-verifying the eval itself. The skill-5 score is combat, not a distance artifact. Findings section 4b + results table written and committed at 8a94b76. Demo mp4 recorded (`runs\vzd\demos\corridor_s5_ft.mp4`, 47.9 MB) and the public reel cut (`runs\vzd\demos\corridor_reel.mp4`, 80s, s3 clip + s5 clip, labeled, honest about the 1 death). Everything decisive remains SINGLE SEED (train seed 0).

**Next action:** Start the seed replication that licenses the README transfer claim. Copy `runs\vzd\_run_s3_shaped_task.cmd` to `_run_s3_shaped_seed1_task.cmd`, change to `--seed 1` and `--out runs\vzd\ppo_deadly_corridor_s3_shaped_seed1` (train script exposes `--seed`, verified; supervisor log path in the cmd must also point at the seed1 log). Register and run as Task Scheduler job `Sight-VZD3-Seed1` (schtasks /create /tn Sight-VZD3-Seed1 /tr the cmd /sc once, then /run), verify launch via `supervisor.log` + `SUP_HEARTBEAT` advancing, recreate a monitor page by copying `monitor_s5.html` and rewiring LOG/SUM/TARGET (1.5M). ~2.3h. When it passes the raw skill-3 eval, run its s5 resume-finetune (seed dir seeded from the numbered checkpoint, `--ent-coef 0.05`, per the fb7ff99 pattern), then its combat probe. Repeat for seed 2 if wall clock allows. 2 extra seeds through the full s3->s5 pipeline is the bar findings 4b sets.

**Blockers:** None requiring Jeff.

**Notes:**

- **MONITOR:** foreground server pattern holds. Currently serving: python PID 15776, `0.0.0.0:8791`, rooted at `runs\vzd`, verified this session by fetching the s5 train log over HTTP and matching latest `total_timesteps` 3,001,184 to disk (the run's final value). Jeff's LAN URL http://192.168.68.108:8791/monitor_s5.html (Wi-Fi) or http://192.168.1.178:8791 (Ethernet). Verify by log content, never status 200. Dies on reboot; the robust always-on service (auto-restart, correct run auto-selected) is still an open item. `monitor*.html` are gitignored, recreate by copy+rewire.
- **NEW INFRA LESSON: Desktop Commander session reaping kills child jobs.** The first probe attempt died at episode 29/30 with no traceback: an MCP request that exceeds the ~60s transport limit returns error -32001 and DC reaps its sessions, which can kill the session's process tree. Rules: keep every DC call's wait under ~50s and poll; launch anything longer than ~1 min detached from DC session lifetime (PowerShell `Start-Process` with redirected output for short jobs, Task Scheduler for long ones); verify by output files, not process status. This is the same killed-not-crashed class as the old runs-2/3 mass kill (root cause of that one still UNKNOWN).
- **Long training jobs: Task Scheduler, MEDIUM+ (3 clean runs: s3 shaped, s5 finetune, and it is the pattern for the seed runs).** If a job dies, `supervisor.log` + `SUP_HEARTBEAT`: stalled heartbeat with no "leg N exited rc=" line proves killed, not crashed.
- **Resume mechanics for the seed pipeline:** `--ent-coef` is reapplied after `PPO.load` since fb7ff99 (was a silent no-op before; the s5 pass depended on it). Seed a resume dir from the numbered `_<N>_steps.zip` checkpoint, never `model.zip`, or VecNormalize stats are lost and returns re-inflate.
- **Standing caveats:** all decisive results are train seed 0, n=1; seeds owed before the README makes a transfer claim. `combat_probe.json` SHOTS_FIRED/accuracy UNRELIABLE (stale AMMO2 baseline at reset), KILLCOUNT/HITCOUNT/DAMAGE_TAKEN clean. Eval is RAW, comparable across runs; shaped ep_rew_mean is not. vizdoom 1.3.0, sb3 2.8.0, Python 3.14. `runs\` is gitignored (probe JSONs, demos, reel are disk artifacts, not in git).
