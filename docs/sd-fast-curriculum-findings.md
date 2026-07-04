# sd-fast start-state curriculum (from-scratch lever)

Scope call ruling (Jeff, this session): KEEP PURSUING FROM-SCRATCH. So the
imitation-vs-from-scratch decision is settled in favor of continuing from
scratch on a structurally NEW lever (every pre-registered lever had failed:
CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet exploration, PBRS reward
geometry; contract forbids retrying a failed method harder).

## Lever

found-art verdict ADAPT. Generalized problem: from-scratch deep RL trapped by a
deceptive local optimum (a passive constant-action policy already scores ~746
replica / 845.7 Godot) plus long-horizon credit assignment (+1/step survival
credited across ~800 steps). Web search (curriculum learning / deceptive reward
/ hard exploration) surfaced the un-tried branch: a curriculum, distinct from
the exploration-pressure family (CMA-MAE, NoisyNet) and the reward-geometry
family (PBRS) that already failed. Prior art: curriculum learning (Bengio 2009),
reset-state / return-to-states (Go-Explore, Ecoffet 2019), adaptive task
generation (arXiv 2007.00350), and a direct analog escaping a strong local
optimum via curriculum (arXiv 2410.16790, 42%->66%).

Implementation: `tools\sd_fast_ppo_curriculum.py`. A start-state curriculum,
`CurriculumSDF` subclass of `SignalDodgeFast` (base env left byte-identical, so
the eval harness and the imitation number are untouched). At reset, inject
`curriculum_n_init` hazards above the player (headroom 100px, no reset
collision, no insta-death). `AnnealCurriculum` callback anneals the count
linearly from n_init_max=6 to 0 over the first anneal_frac=0.7 of training, so
the run ENDS on the true clean-start distribution. Everything else is the m21
recipe verbatim (gamma 0.999, gae 0.95, n_steps 512, batch 512, ent 0.01, lr
3e-4, 8 envs, MlpPolicy [64,64], VecNormalize). Eval is UNCHANGED: greedy on the
standard clean-start env, held-out seeds 5000-5029, via `sd_fast_ppo.evaluate`,
so numbers are directly comparable to the m21 none arm. Reward stays "none".

Smoke-verified this session: clean-start (curriculum off) obs byte-identical to
base env; n_init_max=6 injects 6 hazards with no reset collision; live mutation
of curriculum_n_init reflected in reset; 20k-step end-to-end train+eval ran.

## Result so far (seed 0 of 2)

Matched to m21 none seed 0 (same recipe, seed, 5M budget, eval block). Anchor:
`runs\sd_fast\sd_fast_m21curr_s0_5M_summary.json` (read this session).

| metric | m21 none s0 | curriculum s0 |
|---|---|---|
| eval mean (seeds 5000-5029) | 1119.4 | 1743.07 |
| std | 593.3 | 287.5 |
| seeds at 1800 cap | (n/a in summary) | 28/30 |
| action fracs L/S/R | 0.367/0.222/0.411 | 0.396/0.215/0.389 |
| clears 930.27 bar | yes (mean) | yes, decisively |

Curriculum s0 is the first from-scratch Signal Dodge policy in the project to
clear the bar at imitation-grade level (1743.07 vs BC replica 1764.6, PPO-ft
1571.2), with diverse three-way dodging and roughly half the baseline variance.
Per-seed on the 30 held-out seeds: 28/30 reach the 1800 cap, 29/30 clear the
930.27 bar; only one seed (length 198) fails. HIGH (anchor: the lengths array in
the summary json, read this session).

## Status: INTERIM, seed 1 in flight

Seed 1 (`sd_fast_m21curr_s1_5M`, matched to m21 none s1 baseline 598.0, the
WEAK baseline seed) launched detached in the same chain and was still training
at handoff (trainer pid 25072 live, no summary on disk yet). Seed 1 is the real
test of the lever's variance-reduction claim: baseline s1 was the worst seed
(598), so lifting it is what would show the curriculum fixes reliability, not
just luck on an already-decent seed. Do NOT claim the lever succeeds until seed
1 lands and, ideally, until the arm is extended to >=5 seeds for a run-level
IQM/CI matched to the from-scratch none arm.

## Next

1. Collect `sd_fast_m21curr_s1_5M_summary.json` when the detached run finishes.
2. If seed 1 also clears, extend to seeds 2-4 (chain more curriculum runs) for a
   5-seed arm, then run the rliable IQM/CI/POI vs the m21 none arm (adapt
   `tools\sd_fast_reliability.py` run lists to include the curriculum arm).
3. If the 5-seed curriculum arm clears reliably on the replica, THAT is the
   recipe worth porting to a Godot 5M eval-of-record (bar 930.27), which had
   been blocked because no from-scratch recipe cleared reproducibly.
