# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (Phase F frame-stack diagnostic sweep complete, negative result, Phase G NOT triggered)

**Last commit on HEAD:** `a1ac842` chore: refresh handoff with correct HEAD SHA after phase F evidence commit

**Substantive code commit:** `4ae429f` docs(h5): phase F frame-stack diagnostic sweep evidence

**Current task:** Phase F three-seed diagnostic sweep over the `(4, 84, 84)` observation contract is complete. Pooled trained_cnn over train seeds {1, 2, 3} returns reward 712.87, length 713.80, collision 0.933, timeout 0.067, worse than Phase E aggregate (764.87 / 765.80 / 0.933) on reward and length and equal on collision. Phase G triggers fail under any reading of "best frame-stack negative." Failure mode is policy degeneration to stay-only behavior (seed 1 entropy collapse from iteration 33, seed 3 trained_cnn per-seed lengths byte-identical to stay_only baseline). Frame-stack contract itself propagates correctly (`run_start.env_smoke.obs_shape=[4, 84, 84]` verified). Phase F evidence at `docs\h5-phase-f-frame-stack-evidence.md`.

**Next action:** GPT to choose the next experiment lever from the four documented in the evidence doc: raise `ent_coef`, raise `total_timesteps` past the iteration-32 collapse point, reshape per-step reward (currently +1/step survival incentivizes stay), or change a different perception axis. Not a Claude decision.

**Blockers:** None operational. Open decisions for GPT: (1) select the next experiment lever, (2) resolve the "best frame-stack negative" definitional ambiguity in the Phase G trigger spec (hardest-to-beat = stay_only at 605/606, lowest-collision = three-way tie at 1.0, or lowest reward/length = untrained_cnn at 372/373).

**Notes:**

- Test suite: 307 passed, 2 deselected. Inherited handoff-precision test `test_handoff_does_not_describe_pre_training_gate_as_four_policy` now green; new handoff body preserves "non-saturation" wording.
- All Phase F metrics in the evidence doc are read directly from JSON artifacts in `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\`. No remembered or hand-calculated numbers.
- The dominant blocker at 10000 timesteps is policy collapse to the trivial stay solution, not perception. The Phase F hypothesis "single-frame perception is the blocker" is neither falsified nor supported by this result.
- Seed 3 train hit a transient `GodotTransportError: recv timed out after 5.0s` at step 0 (Godot startup race). Retry from cleared run directory with same seed and config_hash completed `status=ok`. If this recurs on the next sweep, add a startup retry loop to `src\sight_agent\rl\godot_transport.py` rather than relying on manual retries.
- Handoff convention this session: `Last commit on HEAD` matches HEAD at file-write time; `Substantive code commit` references the most recent non-chore commit and is stable across chore refreshes. A chore-refresh commit creates a one-commit lag because a commit cannot self-reference its own SHA in its content.
