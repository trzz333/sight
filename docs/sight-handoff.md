# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Step 9 (default-tier stub transport smoke tests) closed and validated. Step 10 (live Godot smoke test behind the `live_godot` pytest marker) is next.

**Last commit:** `50ea24d` test(rl): add h3 godot stub smoke coverage

**Current task:** H3 step 9 closed. `pytest tests/rl -v --tb=short` 118 passed in 11.60s (111 prior + 7 net new smoke tests; the in-file fakes block in `test_h3_godot_env.py` was extracted to `tests/rl/h3_godot_fakes.py` net-zero on test count). Ready for step 10.

**Next action:** Step 10 per H3 plan implementation sequence: introduce the `live_godot` pytest marker (registered in `pytest.ini` or `pyproject.toml`, default-excluded) and add a single live smoke test under `tests/rl/test_h3_godot_smoke.py` (or a sibling file) that locates the Godot binary via `SIGHT_GODOT_EXE`, launches `games/signal-dodge`, runs `reset(seed=0)` and 100 steps or until terminal, and asserts no malformed protocol messages plus expected event mix in the Godot NDJSON. Default `pytest tests/rl -v --tb=short` must continue to skip it; opt-in command is `pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short`. Step 10 needs a pre-flight from GPT only on marker registration shape (pytest.ini vs pyproject.toml) since the rest is mechanical.

**Blockers:**

- Step 10 will be the first time live Godot launches under H3. `SIGHT_GODOT_EXE` must point at the StrongerJr Godot 4.6.2 binary; the binary itself is already installed and parse-validated. No new install gate.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only, not a Sight evidence blocker. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- New file `tests/rl/h3_godot_fakes.py`: shared fakes (`FakeTransport`, `FakeProcess`, `FakeProcessFactoryRecorder`, `FakeTransportFactoryRecorder`, `_reset_ok_payload`, `_step_ok_payload`). Mechanical extraction from `test_h3_godot_env.py`; no behaviour change. Underscore-prefixed payload helpers preserved verbatim to keep the move mechanical; both consumer files import them by explicit name. `tests/rl/__init__.py` already existed so the relative import `from .h3_godot_fakes import ...` works under pytest collection.
- `tests/rl/test_h3_godot_env.py` shrinks from 862 to 674 lines via the extraction. Fixtures (`fake_proc`, `fake_transport`, `env`) and all 32 test bodies stay in place; only the in-file fakes block was replaced with the import. No env or transport behaviour change; full test count preserved.
- New file `tests/rl/test_h3_godot_smoke.py`: 7 default-tier smoke tests reading as an acceptance overview of the H3 plan section 8 contract. Spaces match plan (Box(-1,1,(10,),float32) + Discrete(3)), `reset(seed=0)` returns `(obs, info)` with obs in observation_space, `step(1)` returns the Gym 5-tuple, ten-step rollout has no protocol drift, forced collision -> terminated True with reward 0.0 and `terminal_reason=collision`, forced timeout -> truncated True with `terminal_reason=timeout`, `close()` is idempotent. A small `_build_smoke_env()` helper replaces fixtures to keep the smoke file flat.
- No production code changed in step 9. `factories.py`, `godot_env.py`, `godot_transport.py`, `train.py`, `evaluate.py`, and `godot_config.py` are byte-for-byte unchanged. No `live_godot` pytest marker introduced (that lands with step 10). No `pytest.ini` change.
- HEAD progression: `dca2af3` (step 8 config plumbing) -> `9cca81a` (handoff post step 8) -> `50ea24d` (step 9 stub smoke coverage) -> handoff refresh (this commit). Default RL gate: 118 passed in 11.60s on StrongerJr. No live Godot path was touched in this step.
