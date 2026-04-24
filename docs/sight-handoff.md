# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P1 closing -> P2 ready (agent loop scaffold next)

**Last commit:** pending (game selected: Signal Dodge, spec recorded)

**Current task:** First game chosen. Spec recorded at `docs/signal-dodge-spec.md`. P2 agent-loop scaffolding is the next prompt.

**Next action:** GPT defines P2 scaffold plan (capture -> perception -> policy -> controller -> logger wiring, plus Godot project layout for `games/signal-dodge/`). Jeff approves. Then Claude scaffolds in a dedicated prompt.

**Blockers:** none. Signal Dodge spec has named open parameters (spawn interval N, spawn edges, hazard velocity, play area and sprite dimensions, player speed) that P2 needs defaults for before the first build.

**Notes:**

- Primary host: **STRONGERJR** (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB).
- ILDC (DESKTOP-QIPEQM0) is now primary for Council. Sight folder on ILDC can be removed after this push lands.
- Convention: first session on a new primary host updates this handoff with `Primary host: <hostname>` and commits. Non-primary sessions are read-only.
- Workbench clone not needed yet; NDJSON logger pattern can be pulled read-only into `C:\Projects\Workbench-ref` when P2 reaches `src/logger/`.
- Chunking rule: one phase per Desktop Claude prompt (no continue button, tool-call budget deaths are silent).
- Ethics armor intact: custom Godot micro-game, Jeff owns assets, no live-service or commercial-game surface area.
