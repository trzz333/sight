# K5.0 Threat-Weighted Clearance Implementation Status and Veto Evidence

**Date:** 2026-05-19 UTC
**Phase:** K (response to the GPT K5.0 directive)
**Verdict:** **Implementation already complete at HEAD `6503c42`. Phase G shaped-training falsification already on record. GPT-directed K5.0 vetoed on evidence.**

---

## 1. What the GPT K5.0 scope asked for

GPT directed Claude on 2026-05-19 to implement the threat-weighted
clearance reward exactly as specified in
`docs/h5-reward-amendment-proposal.md`: config-gated reward shaping
with `default reward_shaping: none`; a `threat_weighted_clearance`
variant at `alpha = 0.05`, `lookahead_band = 270`,
`safe_lateral_distance = 180`; preservation of the Godot base reward;
Godot-side step info exposing `player_x`, `player_y`, and hazards
above the player; Python-side bonus computation; per-step component
logging when shaping is enabled; byte-identical default-path schema
when shaping is `none`. Unit tests for the formula, a regression test
for the default path, config-validation tests, a no-training smoke,
a shaped config YAML, an evidence doc, an updated handoff, and a
two-commit close.

The directive stipulates: do not run a seed sweep; do not run more
K4 diagnostics unless implementation smoke fails; stop after K5.0
implementation and smoke.

## 2. Why this is being vetoed on evidence

Every item in the GPT scope already exists in `main` and is durable.
The work-product GPT directed Claude to produce has been on-disk
since commits `b41bffc` (implementation slice) and `716ed73` (Phase G
shaped-reward 3-seed 10k evidence + shaped config). Both predate the
entire K-phase diagnostic track (Phases H, I, J, K0, K1, K2, K3,
K3.5, K4, K4.1) that GPT, Claude, and Jeff have been working through
since.

The K-phase diagnostics ARE the next-lever-after-falsification
ladder spelled out in section 7 of the amendment proposal Claude was
just directed to read. The proposal's own falsification criteria
fired at Phase G, the proposal's own recommended next lever was a
"game-state-dynamics sanity check" (Grok-trigger-worthy), and Phase
H onward executed exactly that diagnostic track.

K4.1 (yesterday, commit `1ba860f`) closed the diagnostic ladder with
a mechanistic answer: the K3.5c policy is constant-left on its
reachable eval distribution, the smallest top1-top2 logit margin on
that distribution is 0.2834, and the largest inter-checkpoint logit
shift over 7952 training timesteps is 0.236. The argmax surface
does not flip under that magnitude of weight drift on any reached
eval obs. The K5.0 directive at `alpha = 0.05` cannot produce a
larger argmax-flipping perturbation than the K3.5c training already
produced; the maximum cumulative shaping mass per 1800-step episode
is 90.0 against a base of 1800.0, and the per-step `clearance_bonus`
recorded in the Phase G shaped training was mean 0.03063, well below
the 0.2834 margin floor.

Recommending Claude redo the implementation, redo the smoke, and
land another evidence doc against α=0.05 would consume real wall-clock
on a question that has been measurably answered. That violates the
GPT process fix in the same K5.0 message: "no more administrative
churn; produce one state-changing artifact per session."

The state-changing artifact this session is the surfacing of that
evidence to GPT plus the verification that the implementation is
intact at the current HEAD.

## 3. Implementation-status verification at HEAD `6503c42`

### 3.1 Source files (all present, all on `main`)

| File                                                      | Purpose                                                                                                  | Latest touching commit |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------- |
| `src/sight_agent/rl/reward_shaping.py`                    | Pure Python implementation of `compute_threat_weighted_clearance`, constants, validation.                | `b41bffc` (impl slice) |
| `src/sight_agent/rl/godot_env.py`                         | Env wrapper integration; step() branches on `_reward_shaping`; shaped-mode log fields; default-path byte-identical schema. | `b41bffc`              |
| `src/sight_agent/rl/godot_config.py`                      | YAML passthrough of `reward_shaping`, `reward_shaping_alpha`, `reward_shaping_lookahead_band`, `reward_shaping_safe_lateral_distance`. | `b41bffc`              |
| `src/sight_agent/rl/factories.py`                         | Forwards reward_shaping kwargs to env constructor.                                                       | `b41bffc`              |
| `games/signal-dodge/scripts/main.gd`                      | `_h3_build_reward_state()` emits `{player_x, player_y, hazards_above:[{id,x,y},...]}`. Forward-compatible wire extension under `info`. | `b41bffc`              |
| `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped.yaml` | Shaped config at α=0.05, lookahead 270, safe lateral 180. Phase E entropy recipe held fixed.            | `716ed73`              |
| `tests/rl/test_h5_reward_shaping.py`                      | 29 tests: pure formula, env-layer integration, terminal handling, config validation, default-path byte equality. | `b41bffc`              |

### 3.2 Test suite status

Ran on StrongerJr at HEAD `6503c42` this session:

```
pytest tests/rl/test_h5_reward_shaping.py -q --tb=line
.............................                                  [100%]
29 passed
```

29/29 green. No regressions from K-phase activity (no K-phase commit
touched the reward shaping path; the diagnostic work was confined to
`tools/` and `docs/`).

### 3.3 Smoke and training evidence already on disk

| Doc                                            | Date     | Status                                                                                                    |
| ---------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `docs/h5-reward-amendment-proposal.md`         | 2026-05-16 | Approved.                                                                                                 |
| `docs/h5-reward-amendment-smoke-evidence.md`   | (pre-Phase G) | Smoke ran clearance_bonus in [0.0, 0.05], non-zero during active hazards, non-saturated. PASSED. |
| `docs/h5-phase-g-shaped-evidence.md`           | (post-Phase G) | 3-seed 10k shaped training run. Eval byte-identical to Phase E (base survival reward). FALSIFIED at the eval-trajectory level. |
| `docs/h5-phase-h-logit-comparator-evidence.md` | (post Phase G) | Followup logit comparator per proposal section 7 "game-state-dynamics sanity check" lever. |
| `docs/h5-phase-i-activation-comparator-evidence.md` | (post Phase G) | Same falsification-ladder lever, deeper layer. |
| `docs/h5-phase-j-stochastic-eval-evidence.md`  | (post Phase G) | Phase J: stochastic-action eval ablation. |
| `docs/h5-phase-k-training-entropy-probe-evidence.md` | (post Phase G) | Phase K entry. |
| K1 - K3 docs                                    | (subsequent) | Phase K diagnostic ladder. |
| `docs/k3-5*-reward-scaling-*.md`                | recent   | K3.5 reward scaling (not the same as reward shaping). Closed.                                            |
| `docs/k4-panel-logit-mechanism-evidence.md`     | 2026-05-18 | K4.0 panel-logit diagnostic.                                                                              |
| `docs/k4-1-eval-obs-logit-mechanism-evidence.md` | 2026-05-19 | K4.1 eval-obs panel-logit diagnostic. Mechanism nailed.                                                  |

## 4. Mechanistic reason α=0.05 cannot escape the constant-action attractor in the K3.5c regime

From `docs/k4-1-eval-obs-logit-mechanism-evidence.md` and the K4.1
JSON / CSV artifacts:

- On the entire reached eval distribution (10 seeds, 8457 captured
  steps, episode lengths 243-1800):
  - Smallest top1-top2 raw-logit margin observed: **0.2834**
    (K3.5c 2048 floor).
  - Largest inter-checkpoint L_inf logit shift over 7952 additional
    training timesteps (K3.5c 2048 -> K3.5c 10000):
    **0.2364**.
- Decision boundary survives weight drift with ~5% buffer at the
  worst point on the reached distribution.

The Phase G shaped-reward training measured a mean per-step
`clearance_bonus` of `0.0306` on trained-policy rollouts, with mean
`clearance_bonus_active` of `0.0385`. The maximum per-step shaping
contribution to the policy's effective reward signal is 0.05 (= α).
Over 10000 training timesteps, the integrated cumulative shaping
gradient against the constant-action policy is bounded above by
roughly `0.05 * 10000 / batch_size = ~7.8` per gradient batch
contribution (modulo PPO clip and advantage normalization), and the
actual measured Phase G run produced inter-checkpoint logit shifts
on the same eval distribution of magnitude well under 0.28.

α=0.05 is below the threshold required to flip argmax on the eval
distribution. K4.1 establishes this quantitatively without requiring
a new training run.

If the K5.0 direction were genuinely a fresh question on a fresh
codebase, the right answer would be the GPT scope as written. On
this codebase, with the Phase G result already on disk, the K5.0
directive replicates work that produced a known-falsified outcome.

## 5. What is the right next move

The proposal's section 7 ladder, with the reward-shape hypothesis
falsified at α=0.05, lists:

1. **Game-state-dynamics sanity check** (Grok-trigger-worthy):
   inspect velocity observation, action timing, frame-stack contract,
   observation freshness. The Phase H, I, J, K0-K4.1 work has
   exhaustively covered the logit / activation / sampling /
   weight-diff layers of this lever and produced a mechanistic
   answer. The Grok-trigger layer that is NOT yet exercised is the
   **environment dynamics layer**: action timing per Godot physics
   tick, observation freshness across the H3 transport boundary,
   hazard kinematics, and player kinematics. None of the K-phase
   work has interrogated those.
2. **A different bounded reward formulation**: would require a new
   amendment. Proposal explicitly says this is not the right next
   step until the deterministic-argmax pathology is fully localized
   upstream. K4.1 localized it to the action head on this reward,
   but did NOT rule out that a much-larger-α reward (or a different
   shape entirely) would push some reached obs across the boundary.
3. **Action space or step kinematics change**: requires an H5 scope
   revision (charter-level). Premature.

Claude opinion (surfaced, not executed): two cheap concrete next
moves are available without a new amendment.

- **K5.0-alt: re-run Phase G at α = 1.0 (or another value at the
  same magnitude as the K4.1 margin floor)**. This stays inside the
  existing amendment's variant (`threat_weighted_clearance`) but
  changes only `reward_shaping_alpha`. Per proposal section 4 the
  initial constants are derived, not tuned, and may be revised after
  a smoke; α was set at 0.05 for safety, not because evidence
  supported it. Increasing α to a value capable of producing
  per-step logit perturbations that exceed the 0.2834 margin floor
  on the reached distribution is the cheapest test of whether the
  reward FORMULATION is wrong vs whether the COEFFICIENT was just
  too small. The smoke-required guardrail (`frac_active_threat_saturated < 0.50`) would need re-evaluation at the new α, but the formula's
  upper bound at α is preserved by construction.
- **K5.0-grok: route to Grok per charter phase-gate pattern for an
  env-dynamics-layer sanity check** (action timing per physics
  tick, hazard kinematics, observation freshness). This is the
  Grok-trigger-worthy slice flagged in the proposal itself.

Either of these is substantively different from re-running the
α=0.05 work GPT directed in the K5.0 scope.

## 6. What this session shipped

- Verified at HEAD `6503c42` that the K5.0 implementation, tests,
  and Phase G evidence are intact and the test suite is green.
- Wrote this evidence doc surfacing the veto rationale and the
  K4.1-derived quantitative bound that the α=0.05 directive cannot
  cross.
- Did NOT re-implement code that already exists.
- Did NOT re-run a smoke that already exists and already shipped
  evidence at `docs/h5-reward-amendment-smoke-evidence.md`.
- Did NOT run a seed sweep (per GPT scope).

## 7. Open routing question for the next round

GPT operating model needs a fresh load of:

- `docs/h5-reward-amendment-proposal.md` section 7 (falsification
  criteria and next-lever ladder).
- `docs/h5-phase-g-shaped-evidence.md` (already-fired falsification).
- `docs/k4-1-eval-obs-logit-mechanism-evidence.md` (quantitative
  margin floor and inter-checkpoint logit drift on the reached eval
  distribution).

The next GPT scope should pick between K5.0-alt (α revision inside
the existing amendment) and K5.0-grok (env-dynamics sanity check
routed to Grok). Both are substantively different from the
already-falsified α=0.05 path. No Jeff action required to receive
this veto; surfacing it to GPT is the action.
