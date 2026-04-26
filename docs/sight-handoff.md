# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 closed (Grok GREEN, no material concerns); P3 entry, spec drafted and review-patched, awaiting GPT final pass before any harness or evaluator code.

**Last commit:** efab7e5 docs: clarify Signal Dodge win/terminal semantics in P3 metrics spec

**Current task:** docs/sight-p3-metrics.md was patched per GPT review to clarify Signal Dodge win/terminal semantics. The spec now declares an explicit Terminal events list (success_budget_reached, hazard_collision, transport_drop, harness_abort, timeout, other), recasts the failure taxonomy as the derived non-success view, and narrows timeout to mean harness hard-ceiling exceeded or terminal classification lost (never normal successful survival to the configured budget). The SIGHT_TCP_IGNORE_DEATH exclusion invariant is unchanged in force. Spec is still roughly one page.

**Next action:** GPT runs the final review pass against docs/sight-p3-metrics.md. On PASS, Claude implements the smallest harness slice next session, starting with scripts/run_p3_eval.py and a minimal src/evaluator/ aggregator plus tests. No P3 code lands before GPT PASS.

**Blockers:** none.

**Notes:**

- P2 formally closed. Grok verdict GREEN, no material concerns, no required changes. Functional acceptance d167f32; docs hygiene 2032467; closeout state 28da97a; hash refresh 96aa2ef.
- SIGHT_TCP_IGNORE_DEATH is banned from all P3 metric-contributing paths. Three enforcement points written into the spec: scripts/run_p3_eval.py refusal-to-start when the env var is set, src/evaluator/ skip-and-log on per-episode artifacts whose run metadata indicates the flag was active, and a regression test that fails if the literal string appears under src/evaluator/ or in scripts/run_p3_eval\*.py outside an explicit refusal-check guard. Patch verified the string count in the spec is unchanged at 7 occurrences.
- Phase B 300-action clean run remains a transport-only result. Gameplay survivability is now P3 scope. Run artifacts at runs\\diagnostics\\phase_b_live_20260426T074225\\ are preserved as the transport reference.
- GPT review pass identified Signal Dodge survival semantics as the one ambiguity worth patching pre-implementation; that patch is now landed. Remaining open spec choices for GPT's final pass: whether action distribution should also be reported per action-type-class above the primitive level; artifact directory layout runs\\eval\\ vs runs\\diagnostics\\ separation; whether per-game success terminals beyond survival need a registration mechanism in the spec.
- Environment notes preserved. Use absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). Atomic Python writes preferred over Desktop Commander edit_block on docs files. PowerShell Set-Content -Encoding UTF8 writes a BOM and contaminates commit messages.
