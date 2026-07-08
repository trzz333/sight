# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Godot eval-of-record, curriculum recipe. 5M staged probe COMPLETE, plateau confirmed, from-scratch vs imitation scope call now live for Jeff. Bar 930.27.

**Last commit:** `180ee32` sd-godot: 5M curriculum probe complete (plateau); threaded monitor server; below-normal launcher

**Current task:** The 5M curriculum probe (`g99curr_godot_5M_s0`, 5M steps, 8 envs, 93.5 steps/s, ~14.9h at BELOW_NORMAL priority) is complete. All numbers HIGH, anchored to `runs\sd_godot\g99curr_godot_5M_s0_summary.json` read this session. Held-out greedy eval (30 seeds 5000-5029, clean env): IQM 451.6 (95% CI [341.2, 647.6], scipy trim_mean 0.25 + BCa), mean_len 573.93, beats_bar false. 7/30 seeds clear the bar, 1 maxes the 1800 cap, 7 die before step 300. Contrast with the 1M curriculum ref (IQM 550.8 CI [407.3, 704.1], mean 635.03): CIs overlap heavily, so no significant change either way, but the >760 climb gate is decisively falsified. 5x the training budget bought zero improvement; the from-scratch curriculum recipe plateaus around IQM ~450-550, roughly half the bar. Final-log EV in the summary is -1.5723 vs +0.9426 at 1M, but mid-run EV was ~0.58 and the summary field grabs the last minibatch readout; whether the critic degrades after the curriculum fully anneals (at 3.5M) is LOW confidence and untested. Per the pre-registered decision gate the remaining 4 seeds were NOT launched. Also this session: corrected a prior handoff error (the reboot-killed first 5M attempt reached 1,335,296 steps, ~27%, not ~2%; evidence log renamed `g99curr_godot_5M_s0_console.killed-1p335M.log`); replaced the single-threaded monitor server, which wedged on browser keep-alive polling under load, with a ThreadingHTTPServer (`runs\_monitor_server.py`); launched the run via `runs\_launch_g99curr_5M.py` with BELOW_NORMAL_PRIORITY_CLASS, and all 8 Godot children inherited it (load mitigation worked; the run survived to completion).

**Next action:** No compute pending the Jeff scope call in Blockers. On resume without a decision: run the zero-cost per-seed diagnostic comparing 1M vs 5M held-out lengths on the shared seeds 5000-5029 (both summaries on disk) to characterize whether 5M reshuffled which seeds succeed or reproduced the same bimodal split; this sharpens the scope brief. Do not launch any training runs.

**Blockers:** Jeff scope call, now live: from-scratch PPO with the proven curriculum+gamma-0.99 recipe plateaus at IQM ~450-550 at both 1M and 5M budgets (bar 930.27), while imitation clears reliably (BC 1737.3, PPO-ft 1710.5, both use demonstrations). Options are Jeff's to weigh: accept imitation as the standing solution, keep pursuing from-scratch with structurally new methods (exploration/credit-assignment, not more steps), or park. More budget on the current recipe is ruled out by this contrast.

**Notes:**

- 5M artifact `runs\sd_godot\g99curr_godot_5M_s0_summary.json` (gitignored, on disk): IQM 451.6 CI[341,648], mean 573.93, 7/30 clear, 1 cap, 7 die <300. 1M ref: IQM 550.8 CI[407,704], mean 635.03, EV +0.9426.
- Monitor (standing ritual, start every session): `.venv-c1\Scripts\python.exe runs\_launch_monitor.py` then http://127.0.0.1:8791/monitor.html. ThreadingHTTPServer; never use bare `python -m http.server`, it wedges single-threaded under browser polling. Re-point monitor.html LOG/SUM/TARGET at the live run.
- Long-run launch pattern proven: detached Popen, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS; Godot children inherit priority; machine stays usable. Template `runs\_launch_g99curr_5M.py`.
- Trainer still checkpoints nothing (saves only at completion). Deferred deliberately; add tested SB3 CheckpointCallback only before the next multi-hour run, if any.
- `runs\` gitignored (infra scripts force-added). rliable via scipy trim_mean(0.25) + BCa bootstrap, n_resamples 10000, seed 0.
