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
