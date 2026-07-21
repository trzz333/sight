# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 seed replication (seed 1 of 2). Seed-0 pipeline COMPLETE and combat-verified; seed-1 s3 leg PASSED; seed-1 s5 finetune IN FLIGHT. Machine: MSI Raider 18 HX (hostname MSI, RTX 4080 Laptop 12GB); StrongerJr retired.

**Last commit:** b7b2426 infra: supervisor watchdog + instance lock; persistent window watcher

**Current task:** Seed-1 s5 finetune is running healthy at 2,750,000 of 3,000,000 steps (~97 steps/s, checkpoint cadence normal) under the watchdog supervisor, resumed from the 2.6M checkpoint. Two prior death/freeze incidents this leg: a system sleep kill (7/20 01:39) and a 15h full-tree wedge at ~0 CPU with every process alive (cause UNKNOWN; watchdog now converts any recurrence into a kill+relaunch after 30 min without checkpoint progress). The recurring terminal popups Jeff reported 7x are root-caused with a live window watcher, not theory: (a) task \BrotherPrinterHealthcheck, hourly + at logon since 5/20, launching powershell.exe directly so a Windows Terminal window flashes every fire (watcher timestamp matched task LastRunTime to the second); (b) same defect in \Workbench-ClaudeDesktopBootRecover and \WBReach; all three repointed through pythonw launch_hidden with schedules preserved, first post-repoint hourly fire ~02:25 UNVERIFIED; (c) vizdoom 1.3.0 engine windows titled "VIZDOOM 1.3.0 (ZDOOM 2.8.1+)" appear at leg start and at SafeDoom mid-run rebuilds, UNFIXED. Sight's own launch chains produced zero windows under the watcher. Combat probe auto-fires on clean finish via the chained wrapper cmd.

**Next action:** Silence the vizdoom engine windows: in tools\vzd_ppo_train.py env construction force window_visible False / render_mode None (found-art first: check vizdoom 1.3.0 gymnasium wrapper render_mode semantics), verify with the window watcher that a fresh leg spawns zero VIZDOOM windows. Then, when SUP_HEARTBEAT reads "finished clean": verify summary.json raw skill-5 IQM decisively above 93.6, read the auto-produced combat_probe.json, and on FIGHT verdict launch seed 2 (copy _run_s3_shaped_seed1_task.cmd to a seed2 variant, register Sight-VZD3-Seed2 via launch_hidden with the 30-min repetition trigger and battery-safe settings, chain its probe). Seed 2 completing the full pipeline is the bar for the README transfer claim.

**Blockers:** None requiring Jeff. (Optional, not blocking: LAN monitor access needs an admin firewall rule; command given in session log. On-box monitor works at http://MSI:8791/monitor_s5_seed1.html.)

**Notes:**

- **POPUPS (Jeff asked 7x; root-caused 7/21):** offenders were non-Sight scheduled tasks launching powershell/.cmd directly (BrotherPrinterHealthcheck hourly was the big one) plus vizdoom engine windows. Persistent window watcher (Startup sight-window-watch.vbs + running) logs every process that acquires a visible window to runs\vzd\window_watch.log with pid/path/title. Any future popup report: read that log first, never theorize. Verify printer task's ~02:25 fire logged nothing.
- **Resilience stack:** supervisor watchdog (30 min no-checkpoint-progress kills leg tree), SUP_LOCK single-instance, 30-min repetition trigger on training tasks (self-heal after sleep; IgnoreNew + lock prevent stacking), battery-safe settings (AllowStartIfOnBatteries, DontStopIfGoingOnBatteries, no time limit), probe chained into the wrapper cmd with existence guard. Apply all of it to seed 2.
- **Monitor:** hostname URL http://MSI:8791 on the box. Jeff's Brave bookmark pointed at dead 192.168.68.108 (box now .68.102 and .1.178). LAN access from other devices blocked: both network profiles Public, no inbound 8791 rule; rule creation is Jeff-owned admin. Serving verified by HTTP-vs-disk hash equality.
- **DC reaping:** every Desktop Commander wait under ~50s; longer runs detached via pythonw launch_hidden (now multi-arg); verify by output files, never process status. NEVER bare Start-Process on python.exe or a .cmd.
- **Resume mechanics + standing caveats:** --ent-coef reapplied post PPO.load; seed resumes from numbered _<N>_steps.zip + step-matched VecNormalize pkl, never model.zip. README transfer claim gated on seed 2. combat_probe.json SHOTS_FIRED/accuracy UNRELIABLE, KILLCOUNT/HITCOUNT/DAMAGE_TAKEN clean. Eval is RAW and cross-run comparable. vizdoom 1.3.0, sb3 2.8.0, Python 3.14. runs\ gitignored.
