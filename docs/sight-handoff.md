# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H2 (implementation pushed; acceptance, out-of-band eval, fresh-clone repro, and Grok H2 packet pending). H3 not started.

**Last commit:** ebb89b4 feat(rl): add H2 reusable train and eval harness

**Current task:** H2 acceptance run, then matching out-of-band eval, then fresh-clone repro, then build `docs/grok-h2-phase-gate-packet.md`. None of those are done. Implementation and tests are in. An untracked draft of the Grok packet exists in the working tree but has not been committed.

**Next action:** Run H2 acceptance: `python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h2.yaml`.

**Blockers:**

- Claude Desktop GPU/driver crash environment on Jeff's primary box. Intel iGPU has fallen back to Microsoft Basic Display Adapter; Intel 31.0.101.2115 was crashing pre-fallback. Decision pending from Jeff: install Intel 31.0.101.2140 elevated, source 26.20.100.7642 base package, or run Claude Desktop with `--disable-gpu`. Detail in `C:\Projects\ops\claude-desktop-crash-ledger.md`.
- Untracked `docs/grok-h2-phase-gate-packet.md` in working tree from a prior session. Decide whether to commit, edit, or discard before resuming H2 acceptance.

**Notes:**

- 48/48 tests passing in `tests/rl` against `ebb89b4`. Telemetry posture verified clean: no TensorBoard, W&B, MLflow, Comet, or network imports anywhere under `src/sight_agent/rl/`.
- Substantive scope: `train.py` modified to route through new `factories` and `artifacts`, summary schema bumped to 2 with `kind=train|eval`, `config_hash`, `artifact_paths`, model checkpoint when `checkpoint.enabled`. H1 `events_ndjson` field retained for backward compat.
- H1 closure documented in `docs/grok-h1-final-green.md` (Grok GREEN, relayed by Jeff). Verbatim Grok text not captured in repo by design.
- Backup of the pre-commit WIP remains at `C:\Users\maste\AppData\Local\Temp\sight-h2-wip-recovery`. Do not delete without Jeff approval.
- `constraints/rl-cpu.txt` header references "H2 acceptance and fresh-clone repro runs" which have not happened yet. Forward-looking. One-line header tweak when chunk-2 commits land.
