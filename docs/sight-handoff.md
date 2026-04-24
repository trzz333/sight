# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (agent loop scaffold)

**Last commit:** pending (Signal Dodge minimal build landed)

**Current task:** Signal Dodge Godot 4 project scaffolded at `games/signal-dodge/`. Builds the harness only. No agent integration yet.

**Next action:** Jeff opens `games/signal-dodge/project.godot` in Godot 4.3+ and runs F5 to verify. Expected: player square bottom-center, red squares fall from top every 0.5s, collision quits the process, NDJSON written to `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`. Commit any UIDs Godot regenerates on first save. Then GPT defines the agent-side P2 scaffold (capture, perception, policy, controller, logger reader) in `src/`.

**Blockers:** none. Godot 4.3+ must be installed on StrongerJr to verify. If the first-run reveals any scene-format rejection (UID, ext_resource path), Claude revises.

**Notes:**

- Primary host: **STRONGERJR** (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB).
- Renderer set to `gl_compatibility` so RTX 2060 and any integrated GPU both work out of box.
- All movement and spawning in `_physics_process` for determinism across machines (per spec).
- Logger is an autoload. Writes to `user://runs/run_<ts>.ndjson`. One event per line. Schema documented in `games/signal-dodge/README.md`.
- `player_tick` logs at 60 Hz by spec. ~1800 lines per 30 s run. Evaluator will downsample in P3.
- Chunking rule: one phase per Desktop Claude prompt (no continue button, tool-call budget deaths are silent).
- Ethics armor intact: custom micro-game, Jeff-owned, no live-service surface area.
