# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (instrumentation hardened; live verify still gated on Godot install)

**Last commit:** 4266b8f P2: instrumentation patch (first-applied seq, run_id surfacing)

**Current task:** Pre-live-verify instrumentation patch landed. `tcp_controller.gd` now logs `controller_cmd_applied` at most once per new `seq` (first-applied-frame semantics via `_last_logged_seq`) and stamps the hello-supplied `run_id` onto every controller\_\* event after the handshake. `reconcile()` is defensive on duplicate seqs (keeps the first applied frame) and exposes `duplicate_applied_seq_count`; `evaluate()` adds `godot_run_id`, `python_run_id`, `run_id_mismatch` (None when either side is legacy-silent). 36/36 pytest green without Godot installed.

**Next action:** Jeff installs Godot 4.3+ on StrongerJr (`winget install GodotEngine.GodotEngine`), runs the default in-Godot harness first via `godot --path C:\Projects\Sight\games\signal-dodge` to confirm legacy `agent_tick` NDJSON still emits cleanly, then re-runs with `$env:SIGHT_TCP_MODE = "1"` against a small Python client driving `TcpController` to capture paired `python.ndjson` + `godot.ndjson` for evaluator join.

**Blockers:** Godot 4.3+ not installed on StrongerJr. In-Godot verify, TCP visual-loop verify, and determinism band assessment all gated on this.

**Notes:**

- First-applied-frame is enforced by `_last_logged_seq` in tcp_controller.gd; reconciler still defends the contract so legacy or regressed logs do not silently last-write-wins.
- `controller_cmd_applied` now carries `run_id` after hello; `controller_hello` writes it explicitly. Other Godot events (run_start, spawn, agent_tick, death) remain run_id-silent for now; evaluator tolerates that via legacy-mode None on `run_id_mismatch`.
- New evaluator outputs: `godot_run_id`, `python_run_id`, `run_id_mismatch` (True | False | None), `duplicate_applied_seq_count`. None means at least one side did not log run_id, so mismatch cannot be determined.
- [main.gd](http://main.gd) intentionally not modified this round. Stamping run_id onto run_start/spawn/death from [main.gd](http://main.gd) needs the autoload Logger to know the run_id, which is a wider change deferred until first live evidence justifies it.
- Loopback bind only, no external network surface. Charter ethics rules unchanged.
