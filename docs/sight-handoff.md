# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P3 in progress. Loader and runner scaffold landed; live Godot harness pending GPT plan.

**Last commit:** 5052a9665d3e8f0c147bd314567e277dc05d2a1d

**Current task:** P3 per-episode loader landed at src/sight_agent/evaluator/episodes.py and runner scaffold at scripts/run_p3_eval.py. Loader parses godot.ndjson and python.ndjson per episode, reads run metadata from the Godot run_start event, sets Episode.ignore_death_active when run_start carries ignore_death, ignore_death_active, tcp_ignore_death, or env-block mirrored SIGHT_TCP_IGNORE_DEATH (spec exclusion); counts actions from python decision events ordered by seq; computes wall-time from first decision ts to classified terminal ts. Terminal classification cascade: harness_abort > hazard_collision > success_budget_reached > transport_drop > timeout > other(reason). Apply-ratio threshold for transport_drop is 0.9; success_budget_reached requires actions_budget hit AND apply_ratio>=0.9. Runner scaffold refuses to start when SIGHT_TCP_IGNORE_DEATH is non-empty in inherited env (exit code 2 via IgnoreDeathRefusal). --from-artifacts mode reads runs\eval\<run_id>\episodes\<eid>\(godot|python).ndjson, runs loader -> aggregate, writes summary.json. Live Godot launch deliberately not implemented. 21 new tests at tests/test_episodes.py. Full pytest: 81 passed, 1 deselected (up from 60/1). Phase B reconcile evaluator and Phase B runner untouched.

**Next action:** GPT reviews the loader/scaffold API and plans the live P3 harness slice (scripts/run_p3_eval.py live mode that launches the Godot game, drives it for --episodes runs at --actions-budget per episode, writes per-episode godot.ndjson plus python.ndjson, and emits summary.json). No live-launch code lands before GPT plan. Open design questions queued: per-action-type-class distribution above the primitive level; runs\eval\ vs runs\diagnostics\ layout split; per-game success terminal registration mechanism beyond survival.

**Blockers:** none.

**Notes:**

- Loader+scaffold landed at 5052a9665d3e8f0c147bd314567e277dc05d2a1d. New files src/sight_agent/evaluator/episodes.py and scripts/run_p3_eval.py and tests/test_episodes.py (21 tests). Pytest 81 passed / 1 deselected. Phase B and metrics.py purity preserved.
- SIGHT_TCP_IGNORE_DEATH refusal guard lives in scripts/run_p3_eval.py only. The loader and aggregator read no env vars; loader trusts artifact run_start metadata. Regression guard tests still scan src/evaluator/, src/sight_agent/evaluator/, and scripts/run_p3_eval*.py for unguarded occurrences of the literal.
- run_p3_eval.py exit codes: 0 success, 2 ignore-death refusal, 3 missing --from-artifacts path, 4 no episode artifacts found, 5 live mode requested (not implemented).
- Still open for GPT to consider before more P3 code lands: per-action-type-class distribution above the primitive level; runs\\eval\\ vs runs\\diagnostics\\ layout split for P3 eval batches; per-game success terminal registration mechanism beyond survival.
- Environment notes preserved. Use absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). For doc edits use atomic Python re.subn + os.replace. For code files in REPL, use ``code = """..."""`` then a single-line ``open(path, "w", encoding="utf-8", newline="\r\n").write(code)``; multi-line ``with`` blocks barf silently in REPL line-mode.
