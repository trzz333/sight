# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (Python agent layer scaffolded; Godot live verify still pending)

**Last commit:** 542b09d P2: Python agent layer scaffold + Godot TCP controller mode

**Current task:** GPT's locked spec for the external Python agent layer is implemented under `src/sight_agent/{capture,perception,policy,controller,logger,evaluator}` with a TCP loopback IPC contract on `127.0.0.1:8765`. `games/signal-dodge/scripts/tcp_controller.gd` is added; `main.gd` opt-ins via `SIGHT_TCP_MODE=1` env var, default loop unchanged. `pytest` runs 33/33 green without Godot installed; live MSS smoke is gated behind `pytest -m live_mss`.

**Next action:** Jeff installs Godot 4.3+ on StrongerJr (`winget install GodotEngine.GodotEngine`), runs `godot --path C:\Projects\Sight\games\signal-dodge` once to capture first in-Godot NDJSON under `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`, then launches the same project with `$env:SIGHT_TCP_MODE = "1"` to verify the TCP path end-to-end against a Python client.

**Blockers:** Godot 4.3+ still not installed on StrongerJr. Live verification, determinism check, and first-pass difficulty assessment all gated on this.

**Notes:**

- Godot logger schema is `{t, type, ...}` and lacks `run_id`; reconciler joins on `seq` so this is fine for now, but unifying `run_id` and `run_dir` across both sides is queued for the next slice.
- Reconciler accepts both `player_tick` (TCP-mode) and `agent_tick` (legacy in-Godot) so the existing harness keeps working.
- Determinism convention: in-Godot loop targets exact `survival_frames` under `seed=42`; Python visual loop targets exact `survival_frames` first, jitter is reportable but not a tuning trigger.
- Do not change Signal Dodge constants before first live NDJSON. Adjustment order if needed: spawn interval, hazard speed, sprite sizes last.
- Ethics armor intact. Loopback bind only, no external network surface; charter ethics rules unchanged.
