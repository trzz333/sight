# Fast Signal Dodge replica + budget-at-speed experiment

Date: 2026-07-04. Single-voice. Post-Phase-N / MinAtar-adopt. Mission env
(Signal Dodge) still open at the 930.27 bar.

## Question

The MinAtar spike showed the SB3 PPO stack clears a benchmark from scratch at
5M steps / ~6000 steps/s. Phase M's from-scratch PPO on Godot Signal Dodge
failed at only 1M steps/seed. The one axis between them never tested is
budget-at-speed, because Godot Signal Dodge runs ~60 steps/s (measured) so a
5M multi-seed sweep is ~23h single-env. Does budget alone close the gap?

## found-art verdict: BUILD (fast replica)

Generalized problem: "fast vectorizable gym env for a 1D falling-hazard dodge
game." No external library is Signal-Dodge-specific, and Godot-fidelity
requires owning every constant, so BUILD a pure-Python replica rather than
adapt. MinAtar itself is a fast pure-Python env; that is why it affords 5M
steps. The port of the MinAtar lesson is to make Signal Dodge equally fast.

`src\sight_agent\rl\sd_fast.py` (`SignalDodgeFast`). Every constant derived
verbatim from `games\signal-dodge\scripts\{main,player,hazard}.gd` and the
scene collision shapes: player speed 5 px/step, half 16, y fixed 508, x clamp
[16,704]; hazard speed 3.333 px/step, half 12, spawn y -24 at x~U(12,708) every
30 frames, cull y>564; AABB collision |dx|<28 and |dy|<28; obs is the identical
10-dim `_h3_build_observation`; reward "none" (+1/step, 0 on collision).

## Throughput (evidence: session tool output)

- Godot Signal Dodge, state, headless, random policy: 59.8 steps/s single env
  (`tools\sd_throughput_probe.py`).
- Replica random policy: 237,910 steps/s single env (`tools\sd_fast_validate.py`).
  ~4000x. 5M steps trains in ~9-10 min including PPO overhead (8981 steps/s
  PPO-bound, 8 DummyVecEnv).

## Fidelity (evidence: `tools\sd_fast_validate.py` output)

Replica constant-action means (500 seeds) vs Godot K5.2 (10 seeds 1000-1009):

| action | replica 500-seed | Godot K5.2 10-seed | analytic |
|--------|-----------------:|-------------------:|---------:|
| stay   | 518.8            | 606.0              | ~524     |
| left   | 732.2            | 845.7              | -        |
| right  | 746.3            | 689.7              | -        |

The replica constant_stay (518.8) matches the first-principles geometric
analytic (~524) almost exactly. Godot survives ~10-15% longer than strict
geometry (its Area2D collision is slightly more forgiving than a strict AABB).
This is a SAFE transfer direction: a policy that learns to dodge on the
stricter replica survives at least as long on the more-forgiving Godot. Godot
10-seed numbers carry std ~450 so they cannot discriminate finer than this.
K5.2 layers 0-5 already confirmed exact kinematics, spawn cadence, and obs
freshness, so the only residual is collision forgiveness. Replica accepted as a
faithful strict-geometry model; eval of record stays Godot.

## Result: budget alone does NOT clear (evidence: `sd_fast_s0_5M_summary.json`)

One from-scratch PPO seed, MinAtar-winning recipe verbatim (n_steps 128, batch
256, n_epochs 4, gamma 0.99, gae 0.95, clip 0.2, ent_coef 0.01, vf_coef 0.5,
lr 2.5e-4, 8 envs, MlpPolicy [64,64]), reward "none", 5,000,000 steps, eval
greedy over held-out replica seeds 5000-5029:

- mean_len 669.93, std 432.14, actions L 1.00 / S 0.00 / R 0.00. Collapsed to
  constant-left. diversity_ok false. Below replica best-constant (746).
- The 100k-step checkpoint was diverse (L 0.42 / R 0.58, mean 577); by 5M it
  had collapsed to a single action. More budget made it worse, not better.
- explained_variance stayed ~0 the entire run; the critic never fit.

Budget-at-speed with the raw MinAtar recipe did not clear and collapsed to the
same constant-left attractor Phase G/K/M hit. Confidence HIGH (disk summary,
disk log, reproducible tool).

## Self-correction: this run is not a clean budget isolation

I ported MinAtar's recipe wholesale. That dropped two things Phase M2.1 needed
on THIS env: VecNormalize (M2.1 reached explained_variance 0.85-0.94 with it; I
got ~0 without it) and gamma 0.999 (I used 0.99). So the collapse is most likely
the known M2 critic defect plus low gamma, NOT proof budget is irrelevant. The
correct control is Phase M2.1's exact recipe (gamma 0.999 + VecNormalize) at 5M
on the replica, isolating budget as the single variable vs the strongest prior
attempt (M2.1: 1M, IQM 418, diverse, sub-baseline). Reaching for "MinAtar's
recipe" instead of "the strongest prior attempt's recipe at 5x budget" was the
wrong control choice.

## Next

Run the clean budget isolation: M2.1 recipe (reward none, gamma 0.999,
VecNormalize(norm_obs, norm_reward), ent_coef 0.01, 8 envs, MlpPolicy) at 5M
steps on the replica, one seed. If it clears the replica dodging bar with
diverse actions, budget was the wall and it ports to a Godot 5M run. If it
reproduces M2.1's diverse sub-baseline plateau at 5x budget, budget is
definitively refuted and the wall is the exploration/credit structure
(critic blind to death-timing from the 3-hazard obs), redirecting the next
lever off budget entirely.


## Result: clean isolation (M2.1 recipe, 5M) CLEARS the replica bar

Evidence: `runs\sd_fast\sd_fast_m21_s0_5M_summary.json`, per-seed dump via
`tools\sd_fast_eval_dump.py`, train log `runs\sd_fast\m21_s0_5M.log`
(all gitignored under runs\).

One from-scratch PPO seed, M2.1 recipe verbatim (read from
`tools\m2_state_ppo_train.py`): gamma 0.999, gae 0.95, n_steps 512, batch 512,
n_epochs 10, clip 0.2, ent_coef 0.01, lr 3e-4, vf_coef default 0.5, 8 envs,
MlpPolicy [64,64], VecNormalize(norm_obs+norm_reward, gamma 0.999, clip 10/10).
Reward "none". 5,000,000 steps. Eval greedy over held-out seeds 5000-5029, obs
normalized through the saved vecnormalize stats (training=False).

Held-out distribution (30 seeds), sorted:
195, 227, 374, 378, 436, 466, 527, 527, 528, 648, 706, 735, 826, 1006, 1052,
1066, 1215, 1532, 1545, 1606, 1787, 1800 x9.

- mean 1119.4, median 1059.0, IQM 1148.7, std 593.3, min 195, max 1800.
- actions L 0.367 / S 0.222 / R 0.411. diversity_ok TRUE.
- 30% of episodes hit the 1800 cap (perfect survival); 40% still die below
  best-constant 746 (policy is bimodal, real dodging but not yet robust).
- explained_variance healthy through training (0.85 early, bouncing 0.1-0.85
  later as normalized-return variance shrinks; value_loss small, not the M2
  EV~0 defect). No constant-action collapse (contrast the MinAtar-recipe run).

Verdict: budget was the wall. IQM 1148.7 vs M2.1's IQM 418 (same recipe, 1M,
Godot) is ~2.7x and clears the 930.27 bar on mean, median, AND IQM. The M2.1
recipe was sound and starved at 1M by Godot's 60 steps/s throughput. At 5M on
the fidelity-validated replica the identical recipe learns diverse dodging.
First from-scratch clear of Signal Dodge in project history. Confidence HIGH
on the replica clear (disk summary + reproducible reload-eval).

Caveats: (1) one training seed; seeds 1 and 2 launched to confirm reproducibility
(`runs\sd_fast\m21_s{1,2}_5M.log`, sentinel `m21_confirm.sentinel`). (2) The
replica is not the eval of record. The 930.27 bar is Godot; the replica is
~10-15% more collision-forgiving (safe direction for dodging transfer, but
survival lengths do not map 1:1). The eval of record is a Godot 5M run of this
recipe.

## Next

1. Confirm replica reproducibility across seeds 1 and 2 (in flight). Record the
   3-seed IQM spread here.
2. Port the M2.1 recipe to a Godot 5M run for the eval of record vs 930.27.
   Cost note: Godot ~60 steps/s single-env means 5M is long (hours to ~a day
   depending on 8-worker aggregate throughput), and Phase M saw Godot worker
   crashes, so the Godot run needs an SB3 CheckpointCallback + resume so a
   mid-run crash does not lose the whole run. Build that before launching.


## Seed reproducibility: the clear is NOT robust (correction to the section above)

The "CLEARS" section above reports seed 0 only and overstated the verdict.
Confirmation seeds were run. Evidence: `sd_fast_m21_s{0,1}_5M_summary.json`.

| seed | mean | LSR actions            | diversity_ok | beats 746 | EV    | verdict |
|------|-----:|------------------------|--------------|-----------|-------|---------|
| 0    | 1119 | 0.367 / 0.222 / 0.411  | true         | yes       | ~0.85 | CLEAR (IQM 1148.7) |
| 1    |  598 | 0.863 / 0.064 / 0.073  | true*        | no        | 0.885 | FAIL (near constant-left) |
| 2    |  888 | 0.05 / 0.41 / 0.55     | true         | yes (thin) | 0.907 | DIVERSE-SUB-BAR (IQM 750.1) |

*seed 1 max-frac 0.863 passes the <0.97 gate but is functionally
constant-left; the gate is too loose to catch an 86%-pinned policy.

Corrected verdict, confidence MEDIUM: budget lifts the CEILING but does not
buy RELIABILITY. Seed 0 is the first from-scratch policy in project history to
clear the Signal Dodge dodging bar (on the replica), so a from-scratch clear is
now demonstrably reachable at 5M, which was never true at 1M. But seed 1
collapsed to the same constant-left attractor Phase G/K/M hit, with an equally
healthy critic (EV 0.885), so 5M does not make from-scratch reliable seed to
seed. This matches the standing project finding: imitation clears reliably
(BC 1737, PPO-finetune 1710); from-scratch reliability is the open problem, and
5x budget did not close it.

Note the critic is healthy in BOTH the clear and the failure (EV 0.85-0.89), so
the wall is not critic capacity or credit assignment. It is exploration: which
basin the policy falls into is seed-luck, and the constant-left basin is a deep
local optimum the entropy bonus (0.01) does not reliably escape.

## Next (revised)

The lever is reliability, not ceiling or budget. Two structurally different
routes, both off the budget axis:

1. Exploration pressure against the constant-action basin. Higher/scheduled
   ent_coef, or an intrinsic-exploration method (RND / NoisyNets, both already
   in-repo: `noisy_qrdqn.py`, `dyn_qrdqn.py`), or a tightened diversity gate as
   an early-stop/restart signal. found-art before building: RND (Burda 2018)
   and NoisyNets (Fortunato 2017) are the standard escapes; check the existing
   in-repo QR-DQN variants first.
2. Accept imitation as the standing solution (BC/PPO-finetune already clear
   reliably at ~1710-1737) and treat from-scratch reliability as a separate
   research thread rather than the mission-critical path. This is a Jeff-facing
   scope call, not a technical fork.

Do NOT launch the Godot 5M eval-of-record yet: porting a seed-lucky recipe that
clears 1-of-2 to a ~day-long Godot run is premature. Establish replica
reliability (or a restart-on-collapse protocol) first, then port a recipe that
clears reproducibly.


## 3-seed spread confirmed + lever redirect (supersedes the two "Next" blocks above)

Seed 2 finished (`m21_confirm.sentinel` = CHAIN_DONE). All three seeds
re-evaluated by reload-eval, greedy, held-out seeds 5000-5029, obs normalized
through each run's saved VecNormalize stats. Reproducible via
`tools\sd_fast_iqm_spread.py`. Evidence: the three
`sd_fast_m21_s{0,1,2}_5M_summary.json` + the reload-eval below.

| seed | mean | median | IQM | max-frac | L/S/R | cap% | final EV | clears 930.27 |
|------|-----:|-------:|----:|---------:|-------|-----:|---------:|:--|
| 0 | 1119.4 | 1059.0 | 1148.7 | 0.41 | 0.37/0.22/0.41 | 30% | 0.18 | YES (all 3) |
| 1 |  598.0 |  482.0 |  482.1 | 0.86 | 0.86/0.06/0.07 |  3% | 0.885 | no (collapse) |
| 2 |  887.7 |  692.0 |  750.1 | 0.55 | 0.05/0.41/0.55 | 13% | 0.907 | no (sub-bar) |

IQM spread 482.1 to 1148.7, range 666.6. **1/3 seeds clear 930.27 by IQM.**
Confidence HIGH (reload-eval, disk artifacts, reproducible tool).

Two corrections to the prior section, tool output over the handoff narrative:

1. EV does not track clearing. The handoff/table above claimed the critic was
   "healthy in BOTH the clear and the failure (EV 0.85-0.89)." Final logged EV
   per the summary JSONs: seed0 (the CLEAR) = **0.18**, seed1 = 0.885, seed2 =
   0.907. The clearing seed has the LOWEST final EV; the two sub-bar seeds have
   the highest. EV bounces under VecNormalize as normalized-return variance
   shrinks, so no seed's critic is grossly broken, but the specific "0.85-0.89
   in both" claim is not what disk shows and the implied "healthy critic =>
   clear" reasoning is inverted here.

2. Basin collapse is now a MINORITY failure mode, not THE failure mode. 2 of 3
   seeds (0 and 2) are genuinely action-diverse; only seed 1 collapsed to
   constant-left. Seed 2 is the new datum: diverse dodging (L/S/R 0.05/0.41/0.55,
   13% of episodes at the 1800 cap, real avoidance) that plateaus below the bar.
   So the recipe has two distinct failure modes: (a) seed-1 basin collapse, and
   (b) seed-2 competent-but-mediocre dodging. "Exploration pressure against the
   constant-left basin" addresses only (a).

## found-art on the exploration lever: ADOPT the prior negative, do NOT re-run it

Search run: in-repo `src\sight_agent\rl\noisy_qrdqn.py` header + K5 evidence
docs + `docs\phase-l-offline-rl-findings.md`. NoisyNet (Fortunato 2017) as a
temporally-coherent exploration-mechanism replacement was already BUILT and
reliability-tested on this exact env: **K5.8, N=10 seeds, IQM 729.2, 1/10 above
bar, best seed 980.7, half degenerate** (`runs\phase_k\k5_8_reliability_report.json`).
Phase L's cross-method conclusion is explicit: online from-scratch QR-DQN /
NoisyNet / aux-head and offline conservative CQL all fail reliability and
collapse toward a single action; the decisive variable is the QUALITY OF THE
ACTION SIGNAL, not the exploration or conservative knobs. Imitation clears every
time (BC 1737.3, PPO-finetune 1710.5).

Verdict: exploration pressure (higher/scheduled ent_coef, NoisyNet, RND) is the
class of intervention that already returned 1/10 on this env. Pushing PPO's
ent_coef harder is "retry the exploration knob harder," which the contract
forbids after a method fails. It also targets the minority failure mode (seed-1
collapse), not the seed-2 mediocre-dodging plateau. So exploration is NOT the
next lever.

## Next lever (decided): dense per-step reward, targeting signal quality

The one lever the whole project has pointed at and never tested on Signal Dodge:
reward geometry. Current reward "none" is +1/step survival, 0 on collision. That
gives identical per-step credit to a policy skimming a hazard and one loafing in
empty space, so once a policy is "good enough to survive a while" the gradient
toward BETTER dodging is weak. This is exactly the Phase-L "action-signal
quality" axis, and it is the diagnosed root cause of the from-scratch wall
(reward that makes constant-action a valid optimum). It is structurally
different from budget and from exploration, satisfying the lateral-audit rule.

found-art (corrected, fuller than the NoisyNet section): a
`threat_weighted_clearance` shaping ALREADY exists in-repo for the GODOT path
(`src\sight_agent\rl\reward_shaping.py`, Phase G/H5, alpha 0.05, lookahead 270,
safe_lateral 180). It was FALSIFIED at Phase G, but the K4.1 mechanistic
evidence (`docs\k4-1-...`) shows WHY: the per-step clearance bonus averaged
0.0306, far below the 0.2834 top1-top2 logit margin, so alpha=0.05 was simply
too small to move the argmax. That falsification also ran at only 10k steps, a
budget this session proved was itself the wall. So Phase G is not a verdict on
shaping at 5M with a real coefficient.

Verdict: ADAPT, not BUILD. The replica shaping I added is potential-based
(Ng et al. 1999), which is a principled improvement over the existing raw-bonus
form: PBS is provably policy-invariant, so unlike a raw clearance bonus it
CANNOT be gamed by a clearance-maximizing-but-not-surviving policy, which is the
exact "shaping satisfiable by a constant action" risk the K5.5 doc raised.
Coefficient 0.5 (phi in [0,1]) sits above the 0.28 margin floor. Testing at 5M,
not 10k. This is a genuinely different experiment, not a rerun of Phase G.

Design: `reward_mode="shaped"` on `SignalDodgeFast` keeps +1/step survival and
adds `shape_coef * (gamma*Phi(s') - Phi(s))`, Phi = imminence-weighted
horizontal clearance to the nearest hazard above the player. `reward_mode="none"`
stays byte-identical (dynamics verified identical across 20 seeds,
`tools\sd_fast_shaped_check.py`). Experiment: matched-seed none vs shaped at 5M
on the replica, compare CLEAR-RATE, not single-seed ceiling.

Do NOT launch the Godot 5M eval-of-record: still 1/3 on the replica.
