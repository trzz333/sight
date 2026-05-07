# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Implementation Sequence step 1 (protocol constants) complete. Section 10 acceptance criteria amended pre-step-1 to formalize the 10 technical gates vs 3 closure checks split (closure of the flag carried in the previous handoff).

**Last commit:** `b61aefc` docs(h3): clarify acceptance gates and fallback authorization

**Current task:** H3 step 1 closed. Ready for step 2 of `docs/sight-h3-plan.md` Implementation Sequence.

**Next action:** Step 2: extend `games/signal-dodge/scripts/tcp_controller.gd` to parse `reset` and `step` requests and emit `reset_ok`, `step_result`, and `error` responses. New listener path keys on `protocol_version` (H3_PROTOCOL_VERSION=2). Legacy hello (field `protocol`) on the H3 listener path should produce an `error` with code `protocol_version_mismatch`. Plan section 7 has full request/response field lists; `src/sight_agent/protocol.py` REQUIRED_FIELDS_* sets are the authoritative field contract.

**Blockers:**

- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only, not Sight evidence blocker. Sight sessions run on standalone DC remote MCP (deviceId 64416a67-1bdb-42fc-bf1a-48f988e6901d).

**Notes:**

- H3 plan Section 10 amendment landed as `22cf0e2`. Pruning to 10 technical gates + 3 closure checks closes the flag carried in the previous handoff.
- `src/sight_agent/protocol.py` is constants only: no parsing, no transport, no tests of its own. Steps 2 (Godot) and 5 (Python transport) will exercise it. Smoke import + assertion check passed; tests/rl 48 passed unchanged.
- `H3_PROTOCOL_VERSION=2` with field name `protocol_version`. Legacy controller stays at `sight_agent.constants.PROTOCOL_VERSION=1` with field `protocol`. Field-name divergence is the intentional tripwire against accidental cross-mode hellos on the same Godot listener.
- Active runtime gate still in force per plan section "Claude execution boundary" and now formalized in plan section "Fallback authorization" (`b61aefc`): NDJSON log-tailing and subprocess-per-episode each require specific minimum evidence before GPT can consider authorizing; subprocess fallback also requires an explicit acceptance-criteria patch.
- HEAD progression this round: `fafa460` -> `22cf0e2` (acceptance split) -> `ab4f76e` (protocol module) -> `dfb50a4` (handoff refresh) -> `b61aefc` (fallback authorization) -> handoff hash refresh (this commit).
