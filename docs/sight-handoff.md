# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 planning, pre-H5 hardening landed. H4 closed GREEN by Grok per `docs/grok-h4-final-green.md`. The two pre-H5 hardening items called out in section 6 of the closure doc are now implemented and tested (transport literal-pinning, per-reset NDJSON obs metadata persistence). H5 plan is committed at `docs/sight-h5-plan.md`. H5 implementation is authorized but not started; baseline / evaluation suite is GPT's next planning target.

**Last commit:** `9c80fec` feat(rl): harden h4 pixel metadata before h5 planning

**Current task:** H5 baseline / evaluation implementation not started.

**Next action:** Implement the H5 baseline evaluation suite per `docs/sight-h5-plan.md` sections 3 through 6 (four-policy baseline, multi-seed eval posture, non-saturation check, 25% reward-or-length / 20pp collision-rate GREEN bar). Implementation prompt to be authored by GPT.

**Blockers:** none.

**Notes:**

- Transport literal-pinning lands in `_validate_pixel_obs` after the existing type checks: `pixel_source == PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT`, `capture_point == CAPTURE_POINT_FRAME_POST_DRAW`, `headless_allowed is False`. Any deviation raises `GodotProtocolError`.
- Pixel-mode `env.reset()` now emits one `obs_metadata` NDJSON event per reset with the full metadata set (no `obs.data`). State mode emits nothing new. Audits can run from `python.ndjson` alone.
- `pytest tests/rl --tb=short -q` is 238 passed / 2 deselected (was 228 / 2 at H4 closure). Targeted slice (transport + env protocol tests) is 54 passed.
- H5 plan at `docs/sight-h5-plan.md` enforces a non-saturation rule: if negative controls (stay-only, seeded random, untrained `CnnPolicy`) mostly reach `max_steps`, the current Signal Dodge profile is too easy and must be hardened or replaced before H5 closure.
- Operational constraints carry forward: `-s` for live pytest under Desktop Commander, inline `SIGHT_GODOT_EXE`, `runs/` gitignored, pre-mode-lock physics-tick variance applies only to pre-lock observations.


---
