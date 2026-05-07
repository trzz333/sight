# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Implementation Sequence step 2 (Godot bidirectional protocol plumbing) complete in tcp_controller.gd. Step 3 (Signal Dodge soft reset in main.gd) is next.

**Last commit:** `f03aa1c` feat(godot): H3 step 2 bidirectional protocol plumbing in tcp_controller.gd

**Current task:** H3 step 2 closed. Ready for step 3 of `docs/sight-h3-plan.md` Implementation Sequence.

**Next action:** Step 3: implement in-process soft reset in `games/signal-dodge/scripts/main.gd` per plan Decision 2 (clear hazards, reset frame counter, reset death state, reset score/survival, reposition player, reseed Godot RNG from the Python-provided episode seed, return initial observation). Step 3 must also resolve the latent main.gd parse error blocker (see notes) before any Godot-side validation can run. Plan section 7 has the full request/response field lists; tcp_controller.gd public API in step 2 (`mode()`, `has_pending_h3_request()`, `take_pending_h3_request()`, `send_reset_ok()`, `send_step_result()`, `send_error()`) is the seam main.gd will consume.

**Blockers:**

- Latent `main.gd` parse error blocking Godot 4.6.2 parse validation of the project. Lines 110 and 118 fail `Cannot infer the type of "ac"/"rid"` against the new Godot 4.6.2 stricter type inference (`var ac := _tcp.applied_count()` and `var rid := _tcp.run_id()` cannot infer type because `_tcp` is declared as untyped Variant). Pre-existing, unrelated to step 2, but blocks any headless Godot project parse. Must be resolved at the start of step 3 before live validation, e.g. by typing `_tcp` properly or by giving these vars explicit type annotations.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only, not Sight evidence blocker. Sight sessions run on standalone DC remote MCP (deviceId 64416a67-1bdb-42fc-bf1a-48f988e6901d).

**Notes:**

- Step 2 mode-dispatch design per GPT directive: single listener, mode tripwire (`MODE_UNSET` -> `MODE_LEGACY` | `MODE_H3`) locks on first class-identifying field. Cross-mode rejection: legacy `protocol` field in H3 mode produces `error protocol_version_mismatch`; H3 `protocol_version` field in legacy mode is logged and dropped (legacy channel has no response surface). Pipeline overrun (second pending request before main.gd consumes) produces `error bad_request`. Legacy v1 hello/action path preserved verbatim.
- Step 2 introduces only protocol plumbing: `hello`, `reset`, `step` request parsing, field validation, discrete-action validation, single pending request slot, and `reset_ok` / `step_result` / `error` response helpers. No soft reset, no observation builder, no Python transport per directive scope.
- Validation gate: `pytest tests/rl -v --tb=short` 48 passed (baseline unchanged - step 2 adds no Python tests). Python protocol smoke import + key-constant assertions PASS. Godot-side parse validation deferred to step 3 because of the main.gd blocker above.
- `H3_PROTOCOL_VERSION=2` literal duplicated in tcp_controller.gd (mirrors src/sight_agent/protocol.py); the Python module remains the authoritative contract. Drift between the two is a bug.
- HEAD progression this round: `f3bb57e` (handoff hash) -> `f03aa1c` (H3 step 2 code) -> handoff hash refresh (this commit).
