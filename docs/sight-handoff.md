# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 closed GREEN by Grok. H5 is now authorized under the existing charter but not started. The H4 closure record is at `docs/grok-h4-final-green.md`; the H4 evidence packet is at `docs/grok-h4-phase-gate-packet.md`. Grok found no blocking defects; two metadata items (literal-pinning in `godot_transport.py`, NDJSON metadata persistence in `godot_env.py`) are tracked as pre-H5 hardening candidates, not blockers.

**Last commit:** `7f377e6` docs(h4): record grok h4 green verdict and prepare for h5 planning

**Current task:** H4 closure recorded. Prepare H5 planning. No code work in this round.

**Next action:** GPT scopes the H5 plan (learning evaluation of the small CNN policy on Signal Dodge or its successor microgame). Two pre-H5 hardening items are open and may be folded into H5 setup, run as a parallel cleanup slice, or deferred with the caveats explicitly recorded: (1) pin `pixel_source == "godot_windowed_viewport"`, `capture_point == "RenderingServer.frame_post_draw"`, `headless_allowed == False` literal values in `src/sight_agent/rl/godot_transport.py`; (2) persist the obs metadata dict once per reset to `python.ndjson` from `src/sight_agent/rl/godot_env.py`. Both are small patches. Decision is GPT's framing then Jeff's call.

**Blockers:** none for H4 closure or H5 entry.

**Notes:**

- Grok GREEN with no blocking defects. All 11 acceptance criteria from `docs/sight-h4-plan.md` section 10 satisfied. Charter invariants hold with zero drift.
- Pre-H5 hardening candidates flagged but not blocking: literal-value pinning in transport, per-reset metadata persistence to NDJSON. Recommended follow-up only.
- Operational reminders from H4 carry forward: pytest live trajectory test needs `-s` under Desktop Commander; `SIGHT_GODOT_EXE` must be set inline because Desktop Commander does not inherit User-scope env vars; `runs/` stays gitignored.
- Eval mean_reward of 1800.0 in H4 training pair is NOT a learning signal. Learning quality is H5's gate.
- Pre-mode-lock physics-tick variance carries forward from H3. Same-seed reproducibility assertions apply only to post-mode-lock observations.
