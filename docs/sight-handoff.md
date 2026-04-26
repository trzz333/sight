# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Live harness audit complete; implementation pending GPT reconciliation. Loader at 5052a96 frozen for this slice.

**Last commit:** 787b9f0 handoff: P3 live-harness audit complete; implementation pending GPT plan

**Current task:** Read-only audit of wip/p3-live-harness-pre-plan at 13b2e06 against GPT's 12-point live-harness plan finished. Eight points match. Three hard mismatches: no explicit --live flag (live is the default path), no runs\\diagnostics\\p3-live-<batch>\\ split (stdout/stderr land inside the eval episode dir), and the harness uses snapshot-and-copy from Godot user://runs instead of writing godot.ndjson via SIGHT_GODOT_LOG_PATH (the env var does not exist anywhere in repo on either branch). One soft mismatch: src/sight_agent/harness/tcp_client.py duplicates four helpers already living in scripts/run_phase_b.py. One pre-existing gap on main: GDScript never emits success_budget_reached and never tracks actions_budget; loader derives success from decision_count >= actions_budget AND apply_ratio >= 0.9.

**Next action:** GPT reads the audit findings in this handoff and the original 12-point plan, then publishes the reconciled live P3 harness plan covering the GDScript slice that emits success_budget_reached and honors SIGHT_GODOT_LOG_PATH, Python --live flag semantics, runs\\diagnostics\\ split, and whether tcp_client.py helpers stay or fold back into per-script files. After plan, Claude lands the implementation slice from main, GDScript first.

**Blockers:** None on main. Informational: live-launch gate still active; no live-mode code lands on main before GPT's reconciled plan.

**Notes:**

- Audit per-file verdicts vs wip/p3-live-harness-pre-plan@13b2e06: docs/sight-handoff.md discard (stale rewind, predates quarantine); scripts/run_p3_eval.py rewrite (salvage refuse_if_ignore_death, IgnoreDeathRefusal, build_child_env with SIGHT_GODOT_LOG_PATH added, episode_id_for_index, wire_run_id, episode_dir, meta.json round-trip); src/sight_agent/harness/tcp_client.py adopt-with-edits (keep connect_with_retry, wait_for_port_bind, send_json_line, PROTOCOL_VERSION; drop build_hello/build_action/build_decision/action_for_seq as duplicates of run_phase_b.py); tests/test_run_p3_eval_live.py adopt-with-edits (pure-helper tests port cleanly; preflight and never-binds tests need rewriting for the new path layout).
- Smallest clean implementation slice from main, in order: (1) GDScript actions_budget plumbing in tcp_controller.gd plus success_budget_reached emission in main.gd plus SIGHT_GODOT_LOG_PATH support in logger.gd with user://runs fallback; (2) Python --live flag (mutually required with --from-artifacts, no implicit live default); (3) runs\\diagnostics\\p3-live-<batch_run_id>\\ for stdout/stderr and raw subprocess metadata; (4) child env carries SIGHT_GODOT_LOG_PATH=<ep_dir>\\godot.ndjson, snapshot/copy logic deleted; (5) keep meta.json plus harness_status plumbing for harness_abort cases; (6) regression test that child env contains SIGHT_GODOT_LOG_PATH and never SIGHT_TCP_IGNORE_DEATH.
- Loader at src/sight_agent/evaluator/episodes.py already accepts harness_status parameter; _HARNESS_ABORT_STATUSES = {abort, aborted, error, errored, failed, failure, harness_abort}. WIP value harness_status="ok" is not in the set so it falls through to the normal cascade. Loader still derives success from decision_count >= actions_budget AND apply_ratio >= 0.9; works as fallback but does not satisfy GPT plan point 6. GDScript fix in slice 1 is required for source-of-truth correctness.
- Open questions for GPT before the plan: (a) is --live a positive required flag, with refusal unless --live or --from-artifacts is given; (b) when SIGHT_GODOT_LOG_PATH is set does logger.gd write only there or both there and user://runs\\*.ndjson; (c) does meta.json stay adjacent to ndjson in the eval episode dir or move to runs\\diagnostics\\.
- Environment unchanged. C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). Doc edits use atomic Python re.subn + os.replace. cmd.exe shell avoids PowerShell pipe mangling. Last clean pytest from prior session: 81 passed, 1 deselected. No tests run this session (audit was read-only). Charter cosmetic whitespace drift in docs/sight-charter.md was reverted via git restore at session start.
