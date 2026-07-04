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

## Result: 5-seed arm COMPLETE, judged by rliable

All five curriculum seeds trained (5M, m21-verbatim + start-curriculum) and
eval'd greedy on the held-out block 5000-5029. The `tools\sd_fast_reliability.py`
curriculum arm reload-eval reproduced every summary mean exactly (independent
cross-check, so the summaries are trustworthy). Per-seed held-out means:

| seed | m21 none | curriculum | curr clears bar? |
|---|---|---|---|
| 0 | 1119.4 | 1743.1 | yes (af 0.40/0.22/0.39, diverse) |
| 1 |  598.0 | 1704.3 | yes (af 0.78/0.13/0.10, skewed but dodging) |
| 2 |  887.7 |  887.7 | no (collapse) |
| 3 |  669.9 |  669.9 | no (collapse, below best-constant 746.3) |
| 4 |  643.1 | 1591.6 | yes (af 0.42/0.20/0.38, diverse) |

rliable (Agarwal 2021, seed-level bootstrap, BOOT=50000):

| arm | IQM | 95% CI | clears 930 | pool mean |
|---|---|---|---|---|
| none | 733.6 | [613.0, 1042.2] | 1/5 | 783.6 |
| curriculum | 1394.5 | [742.5, 1730.1] | 3/5 | 1319.3 |

Comparison curr vs none: IQM diff **+661.0**; P(IQM_curr > IQM_none) [paired seed
bootstrap] = **0.970**; POI [rliable Mann-Whitney] = **0.743**. HIGH (anchor:
`tools\sd_fast_reliability.py` output this session; cache
`runs\sd_fast\reliability_eval_cache.json`).

## Verdict: BETTER THAN NONE, NOT YET RELIABLE ENOUGH TO PORT

The curriculum is a large, real improvement over the from-scratch none arm (+661
IQM, P=0.970, 3/5 vs 1/5 clears). But it is NOT reliable in the absolute sense
the port decision requires. Two of five seeds (s2 887.7, s3 669.9) fall well
below the 930.27 bar, and the curriculum IQM 95% CI [742.5, 1730.1] straddles
the bar (lower bound 742.5 < 930.27). The earlier two-seed interim (s0, s1 both
~1700) overstated reliability: it was a lucky pair. Porting a recipe that fails
40% of seeds to the expensive Godot 5M eval-of-record would likely reproduce the
coin-flip there. HOLD the port. (The verdict gate in the harness is
P(IQM)>=0.975; 0.970 misses it, but the load-bearing reason to hold is the 2/5
sub-bar seeds and the bar-straddling CI, not the 0.005 threshold gap.)

## Root-cause finding: a shared constant-collapse attractor

Self-audit surfaced that several per-seed length arrays are BYTE-IDENTICAL across
genuinely different trained models: none-s3, curr-s3, shaped-s0, shaped-s1 all
produce the exact same 30 episode lengths (mean 669.9), and shaped-s3 differs by
one step (670.0). Verified against `reliability_eval_cache.json` (element-wise
equality True). This is not a cache bug; it is a real failure mode: from-scratch
PPO on Signal Dodge converges, on a subset of seeds, to ONE specific near-constant
policy that yields ~670 on the held-out block. So the wall is VARIANCE: some
seeds escape the basin, some do not. The curriculum raises the escape rate from
1/5 to 3/5 but does not guarantee escape. HIGH.

## Mechanism: NOT entropy collapse (hypothesis tested and REFUTED)

The first-pass guess (premature entropy collapse, fix by raising ent_coef) was
FALSIFIED by a direct probe. Loaded the trained policies and measured mean policy
entropy over a 2000-state batch (nats, max 1.099): good seeds s0 0.552, s4 0.669;
failing seeds s2 0.639, s3 0.756. The WORST performer (s3, 670) has the HIGHEST
entropy of all. The failing policies are not more deterministic than the winners,
so an entropy bonus targets the wrong mechanism. This is why introspection is not
verification: the "collapse" read came from skewed action fractions, but skewed
argmax-in-rollout does not imply low per-state entropy. Corrected picture:
multi-modal convergence. Seeds settle into different policy basins and only some
basins dodge competently (winners go L-heavy and diverse; s2 went R-heavy, s3
stayed spread and never commits). It is an optimization / credit-assignment
variance problem. HIGH (anchor: probe this session; entropy numbers above).

found-art (search "PPO high seed variance reduce reliability, some seeds converge
good others fail"): the reliability wall for sparse long-horizon PPO is
value-estimation variance, converged across independent sources. arXiv 2301.05104:
sparse reward -> the critic never gets good value estimates for the rare good
states -> high-variance policy training. arXiv 2311.02129: reports the exact
"dichotomous convergence" (a performant group and a failed group of seeds).
arXiv 2111.04504: the lock-in mechanism, early noisy advantages boost one action
and it runs away. This recipe runs gamma 0.999 (effective horizon ~1000) on an
1800-step survival task, so the critic must regress near-undiscounted survival
~1000 steps out (VecNormalize also normalizes returns with gamma 0.999), a large
early-variance source. Verdict ADAPT: cut the discount, not the exploration.

## Next: gamma-0.99 variance-reduction arm (IN FLIGHT)

One-knob change on the same curriculum scaffold: `--gamma 0.99` (effective horizon
~1000 -> ~100 steps), which is plenty for a reactive dodging task and sharply
lowers value-target variance. Not a twice-failed lever (those were CMA-ES,
CMA-MAE, elite-BC, budget 5M, NoisyNet, PBRS reward geometry); the discount is
none of them. 5-seed arm `sd_fast_m21curr_g99_s{0..4}_5M` launched detached this
session (chain log `runs\sd_fast\curr_g99_chain.log`), gamma 0.99, everything
else m21 + curriculum verbatim. Judge with `tools\sd_fast_reliability.py` (the
g99 arm is wired in, guarded on all 5 models being on disk; port gate = clears
5/5 AND IQM CI lower bound > 930.27). If g99 clears reliably, port to a Godot 5M
eval-of-record. If it lifts but is still short, next un-tried knobs are
`anneal_frac` 0.7 -> 0.9 or higher `n_init_max` (hold the scaffold longer for
slow seeds). The retired ent_coef idea is NOT the next move; the probe refuted it.
