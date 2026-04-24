# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (Godot verify pending; Python agent layer scaffold queued)

**Last commit:** fde2a2f skills: sight-handoff bootstrap templates use subtext pattern

**Current task:** Session confirmed Godot 4.3+ still not installed on StrongerJr (`winget list GodotEngine.GodotEngine` returned no match). No live NDJSON captured. Per one-phase-per-prompt rule, the Python agent layer was not scaffolded this round; GPT's scaffold spec (TCP loopback IPC on 127.0.0.1, `src/sight_agent/{capture,perception,policy,controller,logger,evaluator}`, reconciler joining `python.decision.seq` to `godot.controller_cmd_applied.seq`) is queued as a separate prompt.

**Next action:** Jeff installs Godot 4.3+ on StrongerJr via `winget install GodotEngine.GodotEngine`, runs `godot --path C:\Projects\Sight\games\signal-dodge` to capture first live NDJSON under `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`, then pastes GPT's queued Python scaffold prompt into a fresh Desktop Claude session.

**Blockers:** Godot 4.3+ not installed on StrongerJr. Live verification cannot proceed until Jeff runs the winget install.

**Notes:**

- Primary host STRONGERJR (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB).
- GPT scaffold locked: TCP loopback (port 8765 unless repo claims otherwise), JSON-line protocol, deps mss + numpy + opencv-python + pytest, fake providers so unit tests pass without Godot.
- Evaluator reconciler uses frame as canonical simulation coordinate; timestamps for latency diagnostics only.
- Do not tune Signal Dodge constants before first live NDJSON. Adjustment order after first run: spawn interval, hazard speed, sprite sizes last.
- Ethics armor intact; custom Godot micro-game, Jeff-owned, no live commercial surface.
