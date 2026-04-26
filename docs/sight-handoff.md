# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Per-episode loader and runner scaffold landed at 5052a96. Live Godot harness plan pending GPT. Pre-plan live-harness work preserved on branch wip/p3-live-harness-pre-plan; main is clean.

**Last commit:** f3d06bf handoff: P3 live-harness quarantined on wip branch; main clean

**Current task:** Pre-plan P3 live-harness work (uncommitted on main at session start) is now quarantined on branch wip/p3-live-harness-pre-plan at commit 13b2e06 and pushed to origin. Main is clean and the no-live-launch-before-GPT-plan gate is intact. Loader at src/sight_agent/evaluator/episodes.py and runner scaffold at scripts/run_p3_eval.py from 5052a96 remain authoritative on origin/main.

**Next action:** GPT delivers the live P3 harness plan (Godot launch sequencing, per-episode artifact write paths, terminal-event detection signal, success registration beyond survival, runs\\eval\\ vs runs\\diagnostics\\ split, per-action-class taxonomy). After plan, Jeff and Claude review the WIP branch against it and decide: adopt, rewrite, or discard.

**Blockers:** None on main. Informational: live-launch gate still active; no live-mode code lands on main before GPT plan.

**Notes:**

- Pre-plan P3 live-harness work preserved on branch wip/p3-live-harness-pre-plan at commit 13b2e06: scripts/run_p3_eval.py modified +505/-63 vs 5052a96, src/sight_agent/harness/**init**.py and tcp_client.py added, tests/test_run_p3_eval_live.py added. Unreviewed against any plan.
- SIGHT_TCP_IGNORE_DEATH refusal guard lives only in scripts/run_p3_eval.py at HEAD on main. Loader and aggregator read no env vars. WIP-branch run_p3_eval.py is unverified against this invariant.
- run_p3_eval.py committed exit codes at 5052a96 on main: 0 success, 2 ignore-death refusal, 3 missing --from-artifacts path, 4 no episode artifacts found, 5 live mode requested (not implemented). WIP branch may extend these; do not rely on the table until reviewed.
- Charter amended at 2c2fb00 with timer-economy macro sandbox future target between Phase Gates and Success Criteria. Ethics non-goals reaffirmed inside the new section. Future target only; no current scope change.
- Environment: C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). Doc edits use atomic Python re.subn + os.replace. cmd.exe shell avoids PowerShell pipe mangling. Last clean pytest from prior session: 81 passed, 1 deselected.
