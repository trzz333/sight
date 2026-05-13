# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 implementation in progress. Baseline / evaluation harness landed. H4 closed GREEN by Grok per `docs/grok-h4-final-green.md`; pre-H5 hardening landed in `9c80fec`. H5 plan at `docs/sight-h5-plan.md` is the acceptance authority; section 5 now pins the canonical non-saturation thresholds. Training and full 16-seed evidence run not yet started.

**Last commit:** `814457f` feat(rl): add h5 baseline evaluation harness

**Current task:** H5 full baseline evidence and training not yet run.

**Next action:** Run the 16-seed four-policy baseline / non-saturation check (seeds 1000-1015) against the configured Signal Dodge pixel profile, then decide whether the profile must be hardened before H5 training can be treated as learning evidence. If the non-saturation gate passes, training is the next slice; if it fails, the slice after that is a harder profile or successor microgame per `docs/sight-h5-plan.md` section 5.

**Blockers:** none.

**Notes:**

- Non-saturation threshold pinned in `docs/sight-h5-plan.md` section 5: a negative control is saturated if `timeout_rate >= 0.50` OR `mean_episode_length >= 0.80 * max_steps`; the profile FAILs the gate if any of stay-only / seeded-random / untrained_cnn is saturated.
- Four policy evaluators implemented in `src/sight_agent/rl/h5_baseline.py`: `StayOnlyPolicy`, `SeededRandomPolicy`, `build_untrained_cnn_policy`, `build_trained_cnn_policy`. Trained-policy branch is unit-tested for `model.zip` path validation only this slice; first real end-to-end exercise is the later training/eval slice.
- Seeded random policy uses an independent policy-side `numpy.random.default_rng` seeded as `derive_policy_seed(eval_seed) = eval_seed + 1_000_000`. Per-seed rows record both `seed` and `policy_seed` for audit.
- Evaluation artifacts write to `runs/rl/<run_id>/evaluation/<policy>/summary.json` (per-policy) and `runs/rl/<run_id>/evaluation/index.json` (run-level decision). `obs.data` is never copied into evaluation summaries; the H4 metadata-only audit posture carries forward.
- `pytest tests/rl --tb=short -q` is 258 passed / 2 deselected (was 238 / 2 at H4 closure). No live Godot 2-seed smoke ran this session; `SIGHT_GODOT_EXE` was not exercised under Desktop Commander.


---
