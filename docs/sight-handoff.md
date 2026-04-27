# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Live harness audit complete; implementation pending GPT reconciliation. Loader at 5052a96 frozen for this slice. Handoff tooling at 741d080 provisional and not re-dogfooded.

**Last commit:** 1fb6a47 handoff: record handoff tooling recovery

**Current task:** Recovered cleanly from a dogfood glitch in tools\handoff_update.py. CRLF anchor fix committed at 62a396e. The dead session ran a partial pass that injected PLACEHOLDER_HASH PLACEHOLDER_SUBJECT into the Last commit field of the handoff before failing; that doc damage was reverted via git restore. Root handoff_update.json (dogfood input payload) was removed. The "second regex bug" the dead session was diagnosing remains unverified. P3 audit state preserved at 787b9f0; nothing about the live-launch gate has changed.

**Next action:** Either bug-hunt and re-dogfood tools\handoff_update.py in a fresh session, or return to the P3 live harness slice once GPT publishes the reconciled plan. Per GPT, no live-mode code lands on main before that plan. The handoff tooling task is paused, not abandoned.

**Blockers:** None. Live-launch gate still active. tools\handoff_update.py is provisional and must not be used as authoritative until validated.

**Notes:**

- 741d080 added tools\handoff_update.py and canonical user-skill source at tools\skills\sight-handoff\SKILL.md. Upload the SKILL via Claude.ai Settings > Skills to sync into /mnt/skills/user/ if it changes.
- Dogfood glitch injected PLACEHOLDER_HASH PLACEHOLDER_SUBJECT into the Last commit field. This session restored docs\sight-handoff.md and committed only the CRLF anchor fix to the script at 62a396e (field patterns [^\n]* to [^\r\n]*, Notes anchor gained \r? before each newline).
- Do not use tools\handoff_update.py to update docs\sight-handoff.md until a fresh bug-hunt confirms it works end to end on CRLF files. Until then, manual base64 plus os.replace remains the canonical doc-write method.
- Root handoff_update.json was the dogfood input payload, not project state. Removed. Add to .gitignore in a later session if the script gets re-dogfooded.
- P3 context preserved: WIP audit at 13b2e06 vs GPT 12-point plan with 8 matches, 3 hard mismatches (no explicit --live flag, no runs\diagnostics\p3-live-<batch>\ split, snapshot-copy from user://runs instead of SIGHT_GODOT_LOG_PATH), 1 soft mismatch (tcp_client.py duplicates run_phase_b.py helpers), 1 main-side gap (GDScript never emits success_budget_reached). Per-file verdicts and ordered six-step implementation slice in commit message at 787b9f0. SIGHT_TCP_IGNORE_DEATH refusal guard intact in scripts\run_p3_eval.py.
