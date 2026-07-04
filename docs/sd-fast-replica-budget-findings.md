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
| 2    |  ?   | in flight (~0.5M/5M)    | ?            | ?         | ?     | pending |

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
