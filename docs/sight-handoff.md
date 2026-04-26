# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Pure metric-core slice landed; harness slice pending GPT plan.

**Last commit:** TBD

**Current task:** Pure P3 metric aggregator landed at src/sight_agent/evaluator/metrics.py with a 14-test fixture suite at tests/test_metrics.py. The module computes win_rate, episode length in actions and wall-time (mean, median, p95), action distribution counts, Shannon entropy in bits, and terminal and failure counts over the six terminal events. Episodes flagged ignore_death_active=True are excluded from every aggregate per the spec invariant; the module reads no env vars itself. A regression test fails if the literal SIGHT_TCP_IGNORE_DEATH appears under src/evaluator/, src/sight_agent/evaluator/, or scripts/run_p3_eval*.py outside an explicit refusal-check guard. Full pytest: 60 passed, 1 deselected (up from 46/1). Phase B reconcile evaluator untouched.

**Next action:** GPT reviews the metric-core API and plans the next slice. Default queued slice is scripts/run_p3_eval.py with a SIGHT_TCP_IGNORE_DEATH refusal-to-start guard, plus a per-episode NDJSON loader that constructs Episode records and surfaces ignore_death_active from run metadata. No new harness code lands before GPT plan.

**Blockers:** none.

**Notes:**

- Metric-core landed at be18519. New files src/sight_agent/evaluator/metrics.py (5621 B) and tests/test_metrics.py (8705 B). Pytest 60 passed / 1 deselected.
- SIGHT_TCP_IGNORE_DEATH is banned from P3 metric paths. The aggregator reads no env vars; the Episode loader contract carries an ignore_death_active flag set from run metadata. Regression guard test scans both src/evaluator/ and src/sight_agent/evaluator/ plus scripts/run_p3_eval*.py for unguarded occurrences of the literal.
- Path-mapping note. Spec language uses src/evaluator/ ; the actual canonical Python package is src/sight_agent/evaluator/ (only the src tree is under setuptools package-dir). Tests scan both locations so the spec wording stays valid even if a future src/evaluator/ ever gets created.
- Still open for GPT to consider before more P3 code lands: per-action-type-class distribution above the primitive level; runs\\eval\\ vs runs\\diagnostics\\ layout split for P3 eval batches; per-game success terminal registration mechanism beyond survival.
- Environment notes preserved. Use absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). For doc edits use atomic Python re.subn + os.replace. For code files in REPL, use ``code = """..."""`` then a single-line ``open(path, "w", encoding="utf-8", newline="\r\n").write(code)``; multi-line ``with`` blocks barf silently in REPL line-mode.
