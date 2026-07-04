# Phase N / C3 (reward-ranked elite-BC self-imitation) - Screen Findings: NEGATIVE

Method: reward-ranked elite behavioral cloning from filtered self-play (the C3
self-imitation paradigm). Small SB3 MlpPolicy actor (5,059 params), raw mean
episode length objective, 60 iterations, seed-0 first per the pre-registered
screen. Each iteration the top reward-ranked self-play episodes form a rolling
elite buffer; the actor is trained by BC on that buffer. Held-out gate unchanged:
`tools\c1_es_eval.py` on seeds 1000-1009, survival bar 930.27, near-miss floor 880.

## Verdict: C3 screen NEGATIVE (seed-0 clear miss, both vectors, seeds 1-2 not staged)

Anchored to `runs\phase_n\c3_screen_verdict.json` and
`runs\phase_n\c3_screen_s0\c3_report.json` on disk (runs\ is gitignored; numbers
folded here as the citable record).

| vector | held-out mean (1000-1009) | action mix L/S/R | max frac | gate |
| --- | --- | --- | --- | --- |
| best_actor_vec (dev-best, iter 17) | 606.0 | .281/.438/.281 | .438 | FAIL |
| best_final_vec (last iter)         | 443.2 | .328/.344/.328 | .344 | FAIL |

Seed-0 dev-best (in-training, dev seeds) was 790.1 at iter 17 of 60; the run's own
best never reached 930.27 even on dev seeds, and the held-out number is lower still
(606.0). 606.0 is below the 880 near-miss floor, so this is a clear miss, not a
near-miss. Per the pre-registered screen logic, seeds 1 and 2 do not stage
(`staged_seeds_1_2: false`). Sentinel: `EXIT 0 C3-NEGATIVE-seed0-clear-miss`.
Elapsed 6936s (~1h56m), 60 iters, Python 3.14.6 / SB3 2.8.0 / torch 2.11.0+cpu.

No action-collapse this time: the failing policies keep a balanced or stay-heavy
mix (max frac .438), so C3 fails on raw competence, not on the single-action
degeneracy that killed C1. The elite buffer stayed diverse; BC on it produced a
policy that survives ~600 steps held-out and cannot push past that.

## Cross-paradigm result: the wall is from-scratch, not capacity

Held-out best across the three Phase N paradigms is monotonically declining, not
converging on the bar:

- C1 (separable CMA-ES over the actor): 906.4
- C2 (CMA-MAE quality-diversity archive): 845.7
- C3 (reward-ranked elite-BC self-imitation): 606.0

Increasing machinery sophistication (point ES then QD archive then self-imitation
BC) did not move the held-out wall; it lowered the ceiling. The same 5,059-param
actor architecture clears the bar comfortably when trained on demonstrations
(BC 1737.3, PPO-finetune-from-BC 1710.5). So this is not a policy-capacity wall.
It is a from-scratch credit-assignment / exploration wall on Signal Dodge at this
compute budget (seeds-per-gen 2, ~60 iters, high between-seed variance). Five
structurally distinct from-scratch methods (on-policy PPO with VecNormalize,
offline value-RL / DiscreteCQL, CMA-ES, CMA-MAE, elite-BC self-imitation) have now
failed to clear the constant-action survival baseline; the bar is cleared only
when demonstrations are supplied.

## found-art note

C3 was itself the found-art ADAPT from C2's close (search named there:
"self-imitation learning Oh 2018", "ranked behavioral cloning self-play"), reusing
the unchanged actor, rollout/worker pool, and held-out gate. The adaptation was
sound and cheaply executed; it did not clear the wall. No new build was warranted
and none was made.

## Phase N stopping rule: CLOSED FINAL NEGATIVE

The pre-registered rule (stated in `phase-n-c1-es-findings.md` and
`phase-n-c2-findings.md`): three structurally distinct paradigms, one honest shot
each at the reliability gate, then FINAL NEGATIVE if none clear.

- C1 (CMA-ES): spent NEGATIVE.
- C2 (CMA-MAE): spent NEGATIVE.
- C3 (elite-BC self-imitation): spent NEGATIVE.

All three shots taken, all three closed against the same held-out >=930.27 gate.
**Phase N is CLOSED FINAL NEGATIVE.** This is an honest, reproducible negative
result: small-policy from-scratch RL does not clear the constant-action survival
baseline on Signal Dodge across five methods, at hobby-lab compute, with this
actor. Demonstration-seeded learning (BC, PPO-finetune-from-BC) remains the only
route that clears the bar, and by the charter it does not count as from-scratch
mission success.

Mission continuation is now a direction/scope question (Jeff-owned): continue
against a new target environment, redefine what counts as success, or record the
from-scratch-on-Signal-Dodge arc as answered and concluded. No further from-scratch
method on Signal Dodge is warranted: the method has changed five times and the wall
has not moved, so the one structurally different lever left is the environment, not
a sixth algorithm.
