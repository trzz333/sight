# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress (GDScript foundation landed). Loader at 5052a96 and metrics core remain frozen for this slice. Handoff tooling at 741d080 still provisional and unused.

**Last commit:** 3cbd915 feat(godot): P3 GDScript foundation slice for live eval

**Current task:** GDScript foundation slice shipped on main. logger.gd honors SIGHT_GODOT_LOG_PATH (writes only to that absolute path when set, parent dir created on demand, user://runs untouched). tcp_controller.gd tracks distinct first-applied seqs in _applied_count and exposes applied_count(). main.gd reads SIGHT_P3_ACTIONS_BUDGET only in TCP mode (empty/missing/non-positive disables the terminal) and emits success_budget_reached with frame, applied_count, actions_budget, optional run_id, optional episode_id (when SIGHT_EPISODE_ID set), then ends the run, stops TCP, and quits. Live-launch gate still active.

**Next action:** GPT publishes the reconciled live P3 Python harness plan against the GDScript surface now on main (SIGHT_GODOT_LOG_PATH, SIGHT_P3_ACTIONS_BUDGET, SIGHT_EPISODE_ID, success_budget_reached event shape). Once published, Claude implements the Python slice on a fresh branch behind an explicit positive --live flag.

**Blockers:** None. Live-launch gate active. tools\handoff_update.py still provisional; manual base64 plus os.replace remains canonical for docs\sight-handoff.md updates. Desktop Commander read_file is hazardous on Markdown in this repo (rewrites CRLF to LF, auto-links URLs, escapes HTML entities, doubles backslashes). Read Markdown docs via Filesystem MCP read_text_file only.

**Notes:**

- 3cbd915 GDScript foundation. SIGHT_GODOT_LOG_PATH override in logger.gd, _applied_count and applied_count() in tcp_controller.gd, success_budget_reached terminal in main.gd that sets _alive=false and returns early. SIGHT_TCP_IGNORE_DEATH refusal guard in scripts\run_p3_eval.py untouched and unweakened.
- success_budget_reached payload contract: frame, applied_count, actions_budget, run_id (when non-empty), episode_id (when SIGHT_EPISODE_ID set). run_start meta also records actions_budget when positive in TCP mode.
- Working-tree line endings preserved per file: logger.gd CRLF, tcp_controller.gd and main.gd LF. Git index is LF for all three. Patches preserved per-file endings to keep the diff minimal.
- DC read_file Markdown hazard recurred mid-session and was restored via git restore on docs\sight-charter.md and docs\sight-handoff.md before any new work landed. New rule codified: never use DC read_file on Markdown in this repo; use Filesystem MCP read_text_file.
- Checks this session: pytest 81 passed and 1 deselected (live-gated test). git diff --check clean. No Godot launch. GDScript syntax remains untested at the engine level until the next headless or live run.
