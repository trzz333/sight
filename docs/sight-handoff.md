# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P1 (days 1-14, scaffold + game selection)

**Last commit:** pending (handoff: Sight moves to StrongerJr)

**Current task:** Repo transfer ILDC -> StrongerJr complete. Awaiting Godot micro-game selection from GPT.

**Next action:** GPT proposes 3 Godot micro-game concepts matched to StrongerJr specs. Jeff picks one. Then P2 agent-loop scaffolding begins.

**Blockers:** none

**Notes:**

- Primary host: **STRONGERJR** (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB). Moved from ILDC on 2026-04-24 for the dGPU and RAM.
- ILDC (DESKTOP-QIPEQM0) is now primary for Council. Sight folder on ILDC can be removed after this push lands.
- Convention: first session on a new primary host updates this handoff with `Primary host: <hostname>` and commits. Non-primary sessions are read-only.
- Charter at `docs/sight-charter.md` is the source of truth for scope, non-goals, and phase gates.
- Ethics armor: README non-goals up front, `docs/ethics.md` verbatim, CONTRIBUTING.md hard rejections, MIT LICENSE (copyright Jeff).
- Remote: GitHub `trzz333/sight`, public, main. GPT plans; Claude executes and commits; Grok on trigger only; Jeff decides direction.
- Chunking rule: one phase per Desktop Claude prompt (no continue button, tool-call budget deaths are silent).
