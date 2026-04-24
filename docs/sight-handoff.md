# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (in-Godot agent loop live; Python layer deferred)

**Last commit:** pending (agent loop wired)

**Current task:** Signal Dodge now runs an autonomous in-Godot rule agent. Pipeline: capture -> perceive -> decide -> move_action, driven by `Main._physics_process`. Player input from keyboard removed. `seed(42)` locks hazard spawn sequence. Logger schema expanded with `agent_tick` events.

**Next action:** Jeff installs Godot 4.3+ on StrongerJr and runs `godot --path games/signal-dodge` (or F5 in editor). Confirm the run quits on collision and an NDJSON lands in `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`. Commit the regenerated scene UIDs Godot writes on first save. Then GPT sizes the Python agent layer (capture via screenshot, perception via OpenCV, same rule policy for baseline parity).

**Blockers:** Godot not installed on StrongerJr. Cannot run headless verification this session. All code is static-reviewed; first live run is Jeff's verification step. `winget install GodotEngine.GodotEngine` or scoop is the shortest path.

**Notes:**

- Primary host: **STRONGERJR** (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB).
- Agent pipeline is pure-ish: `capture(player, hazards) -> state`, `perceive(state) -> threat-or-none`, `decide(threat, player_x, screen_w) -> {-1,0,+1}`. Easy to port to Python, easy to unit test.
- Rule policy is intentionally weak (nearest-aligned dodge, no look-ahead). Expected survival under random dense spawn: a few seconds. This is a wiring proof, not a good agent.
- `Main._physics_process` is the single authoritative tick driver. Player and Hazard expose methods; Main calls them. No per-child _physics_process competing for order.
- Python agent layer (capture via mss, perception via OpenCV) is the next P2 sub-phase, after Godot-side verification.
- Chunking rule: one phase per Desktop Claude prompt.
- Ethics armor intact.
