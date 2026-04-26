# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 closed (Grok GREEN, no material concerns); P3 entry, spec drafted, awaiting review before any harness or evaluator code.

**Last commit:** 4885a97 docs: add P3 metrics spec and record P2 closeout

**Current task:** docs/sight-p3-metrics.md drafted as the smallest P3 entry slice. The spec defines win rate, episode length (actions and wall-time), action distribution (counts and Shannon entropy in bits), and a five-value failure taxonomy (hazard_collision, transport_drop, harness_abort, timeout, other). It includes a hard SIGHT_TCP_IGNORE_DEATH exclusion invariant binding on the future implementation slice, plus a review checklist that gates any P3 code. Repo state was clean and HEAD was 96aa2ef before the docs commit.

**Next action:** Jeff reviews docs/sight-p3-metrics.md against the embedded review checklist. On checklist pass, Claude lands the smallest harness slice next session, starting with scripts/run_p3_eval.py and a minimal src/evaluator/ aggregator plus tests. No P3 code lands before checklist pass.

**Blockers:** none.

**Notes:**

- P2 formally closed. Grok verdict GREEN, no material concerns, no required changes. Functional acceptance d167f32; docs hygiene 2032467; closeout state 28da97a; hash refresh 96aa2ef.
- SIGHT_TCP_IGNORE_DEATH is banned from all P3 metric-contributing paths. Three enforcement points written into the spec: scripts/run_p3_eval.py refusal-to-start when the env var is set, src/evaluator/ skip-and-log on per-episode artifacts whose run metadata indicates the flag was active, and a regression test that fails if the literal string appears under src/evaluator/ or in scripts/run_p3_eval*.py outside an explicit refusal-check guard.
- Phase B 300-action clean run remains a transport-only result. Gameplay survivability is now P3 scope. Run artifacts at runs\\diagnostics\\phase_b_live_20260426T074225\\ are preserved as the transport reference.
- Open spec choices likely to draw Jeff feedback during review: failure taxonomy completeness for the current Godot micro-game; whether action distribution should also be reported per action-type-class above the primitive level; artifact directory layout runs\\eval\\ vs runs\\diagnostics\\ separation.
- Environment notes preserved. Use absolute paths C:\\Program Files\\Git\\cmd\\git.exe and C:\\Users\\maste\\AppData\\Local\\Python\\bin\\python.exe (Python 3.14.4). Atomic Python writes preferred over Desktop Commander edit_block on docs files. PowerShell Set-Content -Encoding UTF8 writes a BOM and contaminates commit messages.
