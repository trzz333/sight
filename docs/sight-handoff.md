# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Charter amended with timer-economy macro sandbox future target. Loader and runner scaffold landed at 5052a96. Live Godot harness pending GPT plan; pre-plan live-harness work preserved on branch wip/p3-live-harness-pre-plan; main is clean.

**Last commit:** 267d33b handoff: charter timer-economy target; live-harness uncommitted state surfaced

**Current task:** Charter now records a long-horizon timer-economy macro sandbox as a future permitted target environment after the P3 and P5 foundations land (custom Godot or local sandbox first; ethics boundary reaffirmed; no live commercial automation). The previously landed P3 per-episode loader at src/sight_agent/evaluator/episodes.py and runner scaffold at scripts/run_p3_eval.py from 5052a96 remain authoritative on origin/main. Working tree carries uncommitted, unreviewed live-harness work (scripts/run_p3_eval.py modified +504/-63, untracked src/sight_agent/harness/tcp_client.py and **init**.py, untracked tests/test_run_p3_eval_live.py) that violates the active gate forbidding live-launch code landing before GPT plans the slice. This work was already on disk at session start and was not authored or committed this session. Pytest not run this session against the dirty tree.

**Next action:** GPT plans the live P3 harness slice (Godot launch sequencing, per-episode artifact write paths, terminal-event detection signal, success registration beyond survival, runs\\eval\\ vs runs\\diagnostics\\ split, per-action-class taxonomy). Before any live-mode code lands as a commit, Jeff decides what to do with the uncommitted live-harness work currently in the working tree: review and adopt after plan, stash to a side branch, or discard.

**Blockers:** None on main. Pre-plan P3 live-harness work preserved on branch wip/p3-live-harness-pre-plan (commit 13b2e06: scripts/run_p3_eval.py modified +505/-63 vs 5052a96, src/sight_agent/harness/tcp_client.py and **init**.py added, tests/test_run_p3_eval_live.py added). Live-launch gate still active: no live-launch code lands on main before GPT delivers the live P3 harness plan.

**Notes:**

- Charter amendment landed at 2c2fb00. New section Future target: timer-economy macro sandbox sits between Phase Gates and Success Criteria. Ethics non-goals reaffirmed verbatim inside the new section.
- P3 loader and scaffold at 5052a96 remain frozen for this slice. [metrics.py](http://metrics.py) purity preserved. Last clean pytest run from prior session: 81 passed, 1 deselected.
- SIGHT_TCP_IGNORE_DEATH refusal guard lives only in scripts/run_p3_eval.py at HEAD. Loader and aggregator read no env vars. Working-tree run_p3_eval.py is unverified against this invariant until reviewed.
- run_p3_eval.py committed exit codes at 5052a96: 0 success, 2 ignore-death refusal, 3 missing --from-artifacts path, 4 no episode artifacts found, 5 live mode requested (not implemented). The dirty in-tree version may extend these; do not rely on the table until that diff is reviewed.
- Environment: absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). For doc edits use atomic Python re.subn + os.replace. Desktop Commander cmd.exe shell avoids PowerShell pipe mangling. Recurring chat-roundtrip contamination of docs\\sight-handoff.md (&gt;/&lt; entities, backslash escapes, auto-links): revert with git checkout -- before working.
