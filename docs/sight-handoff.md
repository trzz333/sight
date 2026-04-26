# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 complete (Phase B 300-action TCP transport gate clean; closeout review packet drafted, awaiting Grok sanity check per charter)

**Last commit:** TBD

**Current task:** P2 closeout review packet has been produced for Jeff to relay to Grok per the charter's "Each gate: Grok sanity check before proceeding" rule. Functional acceptance is at d167f32 (SIGHT_TCP_IGNORE_DEATH test flag plus promoted scripts/run_phase_b_live.py). Docs hygiene at 2032467 (auto-link noise removed from docs/sight-handoff.md, Last commit field aligned with HEAD-success). pytest reverified at 2032467: 46 passed, 1 deselected, 0.51s. Transport-endurance result preserved as transport-only; gameplay survivability is explicitly deferred to P3.

**Next action:** Jeff pastes the Grok sanity-check prompt from the prior chat turn into a Grok session and relays the verdict back. On PASS or PASS-WITH-NOTES, Claude lands the smallest P3 entry slice first, which is a one-page docs/sight-p3-metrics.md spec, before any P3 harness or evaluator code.

**Blockers:** none. P2 closeout itself is gated on Grok PASS; no work is blocked at the charter or repo level.

**Notes:**

- P2 closeout recommendation. Close P2. Minimal agent loop and TCP transport spine exist and are exercised end-to-end; transport verified clean at 300 actions; logger hardened against truncation; tcp_controller has regression-pinned seq=0 sentinel; tests green. Gameplay survivability is not P2 scope; it sits in P3 measurement layer.
- SIGHT_TCP_IGNORE_DEATH double-gated in main.gd. Line 36 reads the flag only when _tcp_mode is already true; line 138 suppression branch is `if _tcp_mode and _tcp_ignore_death`. Default Godot runs and non-TCP runs cannot reach the suppression path even if the env var is set. TCP wire schema unchanged. The flag is a reachability fix for transport endurance, not a survivability weakening.
- Proposed smallest P3 entry slice. (a) docs/sight-p3-metrics.md one-page spec for win rate, episode length (actions and wall-time), action distribution (counts and entropy), failure taxonomy (hazard collision, transport drop, harness-side abort, other), with explicit ban on SIGHT_TCP_IGNORE_DEATH in any path that contributes to these metrics. (b) scripts/run_p3_eval.py sibling to run_phase_b_live.py, no IGNORE_DEATH, --episodes N, summary.json. (c) src/evaluator/ aggregator with matplotlib charts. (d) pytest fixture asserting metric correctness plus regression test that fails if the literal SIGHT_TCP_IGNORE_DEATH appears under src/evaluator/ or in run_p3_eval*.py.
- 300-action clean run artifacts present at runs\\diagnostics\\phase_b_live_20260426T074225\\. run_id phase-b-20260426T074225, joined_count=300, all unmatched=0, duplicate_applied_seq_count=0, run_id_mismatch=false, seq_zero_applied=true, tcp_death_ignored_count=4. godot.ndjson 124243 B, python.ndjson 64290 B. Promoted harness scripts\\run_phase_b_live.py replays the run with both env vars set.
- Environment notes for next session. git, python, cmd not on this PowerShell session's PATH; use absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). PowerShell Set-Content -Encoding UTF8 writes a BOM by default; use [System.IO.File]::WriteAllText with a UTF-8 encoder constructed with $false to avoid contaminating commit messages. Desktop Commander edit_block was unreliable on this file in the prior session; prefer atomic Python re.subn + os.replace for non-trivial edits.
