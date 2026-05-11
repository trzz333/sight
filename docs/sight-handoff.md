# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 acceptance complete on StrongerJr; H4 phase-gate packet drafted and awaiting Grok review. H4 Steps 1 through 8 landed; Step 9 acceptance evidence captured this round. Default test gate at `tests/rl` reads 228 passed, 2 deselected (the 2 `live_godot` opt-ins). Live H4 pixel trajectory equality test passes (1 reset + 10 scripted steps, byte-equal across two same-seed runs). Two same-seed 128-step CPU PPO `CnnPolicy` training runs against `configs/rl/signal_dodge_ppo_h4_pixel.yaml` both exit 0 with all 8 expected artifact entries (`summary.json`, `events.ndjson`, `config_effective.yaml`, `model.zip`, `godot-train/{godot,python}.ndjson`, `godot-eval/{godot,python}.ndjson`). Identical step-128 PPO metrics to all printed digits across both runs; identical eval mean_reward (1800.0 at step 64 and 128 in both runs); identical `config_effective.yaml` SHA-256. Packet at `docs/grok-h4-phase-gate-packet.md`.

**Last commit:** `9e4bbae` docs(h4): add grok h4 phase gate packet and refresh handoff for step 9 acceptance (on top of `0567fec` fix(rl): isolate godot tcp port and absolute godot log path for h4 live train)

**Current task:** H4 Step 9 acceptance complete. Draft of `docs/grok-h4-phase-gate-packet.md` recommends GREEN verdict pending Jeff relay to Grok. Two follow-up YELLOW-candidate caveats surfaced: (1) transport validates pixel-source metadata field **types** but not specific literal values; (2) pixel-source metadata is not persisted to NDJSON, so artifact-only audit currently relies on transport-validation-survival plus source-code inspection. Both are small follow-up patches and not blocking for the boundary gate.

**Next action:** Relay packet to Grok for review. If GREEN, H5 (learning evaluation of small CNN policy on Signal Dodge) becomes the next phase under the existing charter. If YELLOW, close out the two pixel-source-metadata caveats with a small transport + NDJSON patch round and a YELLOW closure doc in the H1 pattern. If RED, address Grok's specific concerns.

**Blockers:**

- Live H4 train smoke and live trajectory equality test require `SIGHT_GODOT_EXE` set inline in the parent shell (Desktop Commander does not inherit User-scope env vars). Operational; same as H3 acceptance.
- Pytest stdin capture under Desktop Commander breaks `subprocess._make_inheritable` on Windows; the live trajectory equality test must be run with `-s`. Same family as the H3 `subprocess.PIPE` deadlock; not patched in this round per the bootstrap "no code changes unless required" instruction. Operational.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Sight runs on the standalone DC remote MCP, unaffected.

**Notes:**

- **Trajectory equality is binding evidence.** 11 byte-equal pixel observations across two same-seed runs of the scripted mix-action rollout (`[1,0,2,1,0,2,1,0,2,1]`) pass `np.array_equal` with no tolerance. This is the H4 plan section 10 criterion 6 gate.
- **Identical PPO step-128 metrics across both training runs.** approx_kl 8.7335706e-05, entropy_loss -1.0985275506973267, value_loss 165.138671875, policy_gradient_loss 0.001090841367840767, explained_variance -0.001621842384338379, loss 81.49404907226562. CnnPolicy forward and backward are deterministic on this machine at seed 0 under this CPU build. Per H4 plan and bootstrap guidance, not a learning claim.
- **`config_effective.yaml` byte-identical** across run1 and run2 (sha256 `cea7867a...d347`). `events.ndjson` and `model.zip` are not byte-identical (timestamps and per-run ids), which the bootstrap explicitly allows.
- **Pixel-source metadata** is validated at type level on every receive in `godot_transport.py` lines 637-657 (`pixel_source` str, `capture_point` str, `headless_allowed` bool, viewport_w/h positive int). 128+ pixel-mode receives per run completed without raising `GodotProtocolError`. Specific literal values (`"godot_windowed_viewport"`, `"RenderingServer.frame_post_draw"`, `headless_allowed = false`) are documented in `tcp_controller.gd` lines 580-596 and `docs/sight-h4-plan.md` Decision 4 but not pinned in the transport check.
- **Eval mean_reward 1800.0** in both runs at both eval checkpoints is consistent with a 1800-step deterministic eval rollout under freshly initialized CnnPolicy and Signal Dodge's step-0 hazard density, not with policy learning. Learning quality is H5.
