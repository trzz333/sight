# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (Godot 4.6.2 installed, BOM strip landed, phase (a) blocked on two newly exposed Godot 4.5+ incompatibilities)

**Last commit:** 055ee3e P2: strip BOM from Godot text resources

**Current task:** Godot 4.6.2 installed on StrongerJr via `winget install GodotEngine.GodotEngine -e --silent --disable-interactivity` (no admin escalation; absolute exe at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe`). All 8 BOM-bearing text resources stripped (3 .tscn, 4 .gd, project.godot) via explicit-file Python script in %TEMP% (no recursive PowerShell, no Bitdefender block this round). 36/36 pytest still green. Phase (a) launched and surfaced two parse-time blockers that did not exist in the Godot 4.3 era when the scaffold was authored.

**Next action:** GPT decision required on the two blockers below before Claude touches main.gd or player.gd. Suggested combined commit shape `P2: Godot 4.6 compat (rename Logger autoload, type clampf in player.gd)`. After that single edit lands, re-run phase (a) with absolute Godot exe path and SIGHT_TCP_MODE unset.

**Blockers:**

1. Godot 4.5 introduced a native `Logger` class for engine log interception (4.5 release notes; Godot forum reports identical `Static function "x" not found in base "GDScriptNativeClass"` parse errors when migrating older projects). Project's autoload `Logger="*res://scripts/logger.gd"` now collides at parse time. 7 `Logger.` call sites in main.gd (lines 36, 96, 103, 121, 130, 136, 137) plus 1 line in project.godot need a rename. GPT to approve main.gd edit and pick autoload key (suggested SightLog | RunLog | EventLog).
2. `player.gd:22` `var dir := clamp(float(action), -1.0, 1.0)` triggers `INFERRED_DECLARATION` warning-as-error in Godot 4.6 (variant `clamp` returns Variant; type inference yields Variant; default project warning level escalates). Minimal fix: `var dir: float = clampf(float(action), -1.0, 1.0)`. One file, one line.

**Notes:**

- BOM strip method: explicit-file Python script in %TEMP% with literal 8-path list, no recursion, no PowerShell. Bitdefender did not flag. Verified post-strip via Python that no listed file starts EF BB BF.
- Godot user_data dir is `%APPDATA%\Godot\app_userdata\Signal Dodge\`. `logs\godot.log` captured the parse errors verbatim; `runs\` directory was never created because Logger autoload `_ready` never fired.
- winget `--silent --disable-interactivity` install completed without admin; PATH alias `godot` not created (admin-only behavior). Use absolute exe path; bare `godot` will not resolve in this shell.
- `agent.gd`, `hazard.gd`, `logger.gd`, `player.gd` had BOM and lost it; `main.gd` and `tcp_controller.gd` were already BOM-free from later-session edits in editors that do not emit BOM.
- Loopback bind only, no external network surface. Charter ethics rules unchanged.
