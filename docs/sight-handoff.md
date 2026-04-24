# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (in-Godot agent loop live; Godot verification and Python agent layer pending)

**Last commit:** 9d0054a skills: sight-handoff description passes skill-packager validator

**Current task:** Signal Dodge Godot harness built with an autonomous in-Godot rule agent pipeline (capture -> perceive -> decide -> controller -> logger), driven entirely by `Main._physics_process`. Everything committed and pushed to `trzz333/sight`. The `/sight-handoff` skill is iterated and packaged as a `.skill` file for Claude Desktop upload.

**Next action:** Jeff does two things in parallel. (1) Install Godot 4.3+ on StrongerJr (`winget install GodotEngine.GodotEngine` is shortest) and run `godot --path C:\Projects\Sight\games\signal-dodge` to verify the harness emits real NDJSON to `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`. (2) Upload `sight-handoff.skill` through Claude Desktop Settings so `/sight-handoff` triggers reliably in future sessions.

**Blockers:** Godot 4.3+ not installed on StrongerJr. Live verification blocked until installed. Not blocking forward planning; GPT can size the Python agent layer in parallel.

**Notes:**

- Primary host: STRONGERJR (i7-10750H 6C/12T, 64 GB RAM, RTX 2060 4 GB).
- After Godot verify, GPT sizes the Python agent layer (screenshot capture, OpenCV perception, same rule policy for baseline parity with the in-Godot agent).
- `Last commit` convention: references the HEAD just before the handoff commit. The handoff commit itself is implicit (the reader is already viewing its output).
- Chunking rule active. One phase per Desktop Claude prompt.
- Ethics armor intact. Custom Godot micro-game, Jeff-owned, no live commercial surface.