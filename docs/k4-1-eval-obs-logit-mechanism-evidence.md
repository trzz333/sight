# K4.1 Eval-Observation Panel-Logit Mechanism Evidence

**Date:** 2026-05-18 / 2026-05-19 UTC
**Phase:** K (K4.1 eval-observation panel-logit diagnostic on existing K3.5c checkpoints)
**Classification:** **K4.1-B boundary-equivalent drift**

---

## Verdict

**K4.1-B boundary-equivalent drift.** On the entire eval observation
distribution actually reached by the K3.5c policy across seeds
1000-1009 (8457 total per-step observations spanning episode lengths
243 to 1800, 9 collisions and 1 timeout), K3.5c 2048 and K3.5c 10000
produce identical deterministic argmax actions on every single step.
Raw logits drift between the two checkpoints with L_inf max 0.236
across the 8457 obs, L1 max 0.543, and a non-trivial margin gap (mean
top1-top2 margin 0.287 for 2048 vs 0.374 for 10000), but the drift
never crosses a decision boundary on any reached eval observation.

This generalizes K4.0's K4-D finding from a narrow scripted panel to
the full reached eval distribution.

**The mechanism for the K3.5c bit-identical eval is now mechanistically
established:** both checkpoints output action "left" at 100% of all
reached eval observations. Trajectories coincide step by step because
at every reached state both policies emit the same action, so the
next observation is also identical, so the same action is emitted
again, ad infinitum until the env terminates. The K3.5 reward-scaling
intervention did not escape the constant-action attractor; it moved
weights enough to change logit magnitudes by ~0.2 across 7952
additional training timesteps (8 to 156 PPO updates) but cannot
change the policy's externally-visible behavior.

---

## What was run

`tools/k4_1_eval_obs_logit_probe.py` (new) runs two passes through the
production Godot eval env factory and probes both checkpoints on each
captured observation.

**Pass 1** (actor = K3.5c 10000, cross-probe = K3.5c 2048): Open one
Godot env via the standard `sight_agent.rl.factories.make_env(
env_id, n_envs=1, seed=base_seed, mode="eval", run_dir=..., **godot_extra)`
pattern (same as `tools/h5_stochastic_eval.py`). Loop seeds 1000-1009.
For each seed: `env.seed(s)`, `env.reset()`, then deterministic
argmax rollout driven by K3.5c 10000's policy. Per step, capture:

- raw obs (uint8 (1,84,84))
- SHA-256 of the contiguous obs bytes
- K3.5c 10000 forward pass: logits L/S/R, probs L/S/R, argmax,
  top1-top2 margin (self-probe)
- K3.5c 2048 forward pass on the SAME obs tensor: same fields
  (cross-probe)
- The action taken (= K3.5c 10000's argmax) that steps the env

**Pass 2** (actor = K3.5c 2048, no cross-probe): Open a second Godot
env, same loop, deterministic argmax rollout driven by K3.5c 2048's
policy. Capture per-step obs SHA-256 and action only. Used to verify
trajectory parity against Pass 1.

**Trajectory parity check** (K4.1-E gate): for each seed, compare
Pass 1 vs Pass 2 on episode length, collision/timeout flags,
per-step obs SHA-256, per-step action. All four match on every step
of every seed.

**Logit aggregate**: argmax match fraction, logit L1 / L_inf
distributions across all 8457 Pass 1 rows, margin distribution per
checkpoint, per-seed first-argmax-diff step (none on any seed).

**Classification mapping**: K4.1-E first (reproducibility); then
K4.1-A vs K4.1-B (logit drift threshold 0.1); then K4.1-C vs K4.1-D
(only reached if argmax differs anywhere).

---

## Per-seed Pass 1 results (actor = K3.5c 10000)

| Seed  | Length | Terminal | Cross-probe diffs | First diff step | Wall time |
|-------|--------|----------|-------------------|-----------------|-----------|
| 1000  | 1383   | collision| 0                 | None            | 38.6 s    |
| 1001  | 483    | collision| 0                 | None            | 13.5 s    |
| 1002  | 1293   | collision| 0                 | None            | 36.0 s    |
| 1003  | 603    | collision| 0                 | None            | 16.8 s    |
| 1004  | 1443   | collision| 0                 | None            | 40.0 s    |
| 1005  | 363    | collision| 0                 | None            | 10.1 s    |
| 1006  | 573    | collision| 0                 | None            | 15.9 s    |
| 1007  | 273    | collision| 0                 | None            | 7.6 s     |
| 1008  | 1800   | timeout  | 0                 | None            | 50.0 s    |
| 1009  | 243    | collision| 0                 | None            | 6.8 s     |

**Total Pass 1 steps captured:** 8457. **Total Pass 1 wall time:** 235.3 s.

---

## Per-seed trajectory parity (Pass 1 vs Pass 2)

| Seed  | Pass 1 len | Pass 2 len | length_eq | termination_eq | obs match | action match | full obs | full action |
|-------|------------|------------|-----------|----------------|-----------|--------------|----------|-------------|
| 1000  | 1383       | 1383       | True      | True           | 1383/1383 | 1383/1383    | True     | True        |
| 1001  | 483        | 483        | True      | True           | 483/483   | 483/483      | True     | True        |
| 1002  | 1293       | 1293       | True      | True           | 1293/1293 | 1293/1293    | True     | True        |
| 1003  | 603        | 603        | True      | True           | 603/603   | 603/603      | True     | True        |
| 1004  | 1443       | 1443       | True      | True           | 1443/1443 | 1443/1443    | True     | True        |
| 1005  | 363        | 363        | True      | True           | 363/363   | 363/363      | True     | True        |
| 1006  | 573        | 573        | True      | True           | 573/573   | 573/573      | True     | True        |
| 1007  | 273        | 273        | True      | True           | 273/273   | 273/273      | True     | True        |
| 1008  | 1800       | 1800       | True      | True           | 1800/1800 | 1800/1800    | True     | True        |
| 1009  | 243        | 243        | True      | True           | 243/243   | 243/243      | True     | True        |

**All 10 seeds reproduce bit-identical eval at both the obs-hash and
action-trace level.** K4.1-E is cleared.

---

## Logit aggregate over 8457 Pass 1 captures

| Statistic                                | K3.5c 10000 | K3.5c 2048  | Comparison           |
|------------------------------------------|-------------|-------------|----------------------|
| Argmax = "left" count / 8457             | 8457        | 8457        | 100% identical       |
| Argmax = "stay" count / 8457             | 0           | 0           | 100% identical       |
| Argmax = "right" count / 8457            | 0           | 0           | 100% identical       |
| Mean top1-top2 margin                    | 0.3744      | 0.2865      | margin grew ~30%     |
| Min top1-top2 margin                     | 0.3476      | 0.2834      | always > 0.28        |
| Max top1-top2 margin                     | 0.4127      | 0.3023      |                      |
| Inter-model L_inf logit delta mean       | -           | -           | 0.2217               |
| Inter-model L_inf logit delta max        | -           | -           | 0.2364               |
| Inter-model L1 logit delta mean          | -           | -           | 0.4943               |
| Inter-model L1 logit delta max           | -           | -           | 0.5427               |

- Argmax match fraction: **1.000000 over 8457 / 8457 rows**.
- Per-seed first divergence step: **None on all 10 seeds**.
- The smallest "left vs second-best" margin observed on the eval
  distribution is 0.2834 (K3.5c 2048 lower-bound), and the largest
  inter-model logit shift is 0.2364. Margin minus drift = 0.047. The
  decision boundary survives the per-checkpoint drift with a
  consistent ~5% margin buffer at the worst point on the reached
  distribution.

---

## Why this is K4.1-B and not K4.1-A

K4.1-A would require both 100% argmax match AND no material logit
drift. The data shows 100% argmax match with L_inf max 0.236 (clearly
above the 0.1 material-drift threshold). The training is doing
something between updates 8 and 156: action_net row norms grew from
{L=0.086, S=0.045, R=0.083} at 2048 to {L=0.112, S=0.116, R=0.104}
at 10000 (K4.0 per-model summary), the mean top1-top2 margin grew
from 0.287 to 0.374, and per-step logits shift by L_inf 0.22 on
average. The training is sharpening the "left" preference, not
maintaining it. Yet none of this sharpening crosses a decision
boundary on the eval distribution.

K4.1-A vs K4.1-B is a meaningful distinction for the next intervention:

- K4.1-A would mean the training is a no-op on the action surface
  reached during eval; the gradient direction is orthogonal to the
  decision boundary entirely.
- K4.1-B means the gradient IS doing work, in the direction normal
  to the decision boundary (sharpening), but the eval distribution
  does not contain any obs whose logits sit close enough to the
  decision boundary to flip under that sharpening.

Either way, the immediate falsification of the original K3.5c "weight
diff = behavior diff" hypothesis is complete: weight differences are
real, logit differences are real, behavior differences on eval are
absolutely zero.

---

## Cross-cutting finding: the K3.5c policy is degenerate constant-left on its reachable eval distribution

Of the 8457 reached eval observations across 10 seeds, K3.5c 2048 and
K3.5c 10000 both output action "left" 100% of the time. Episodes
terminate by collision in 9 / 10 seeds and by timeout (length 1800)
in 1 / 10 seeds.

This matches the wall_hugging_into_collision classification used by
the Phase J stochastic-eval ablation tool. The K3.5c training regime
(`reward_scaling: divide_by_30`, `ent_coef=0.01`, `n_steps=2048`,
`batch_size=64`, `n_epochs=4`, CnnPolicy with NatureCNN) does not
move the policy off the constant-left attractor under 10000 timesteps
even with the corrected reward scale.

This is a substantive finding about the K3.5 training regime
independent of the K4 mechanism question. It implies that any future
experiment intending to demonstrate non-constant behavior on this env
must change something other than reward scaling, entropy coefficient,
or training duration within the constant-action attractor's basin.

---

## Implication for K3.5c reproducibility claim

The K3.5c "bit-identical per-seed eval despite distinct SHA-256
checkpoints" claim is mechanistically explained, not refuted. The
explanation is:

1. The K3.5c policy is degenerate-constant on its reachable eval
   distribution under deterministic-argmax eval.
2. The first PPO update or two pushed action_net into a corner of
   policy space where "left" beats "stay" by margin >= 0.28 on every
   obs the eval distribution will ever reach.
3. Subsequent training updates (through update 156) push logits
   further into that corner (margin grows from 0.287 to 0.374) but
   cannot cross the "left vs stay" boundary on any reached eval obs.
4. Therefore deterministic-argmax eval is invariant to which post-
   update-1-ish checkpoint is used, as long as the checkpoint is
   still inside the constant-left basin.

The bit-identical eval was real and is now explained. It is not
evidence of a bug in checkpoint loading or eval reproducibility.

---

## Files added or changed this session

Code:
- `tools/k4_1_eval_obs_logit_probe.py` (new, ~605 lines)

Runs:
- `runs/phase_k/k4_1_eval_obs_logit_probe.json` (~85 KB)
- `runs/phase_k/k4_1_eval_obs_logit_probe_rows.csv` (~2.7 MB, 8458 rows incl. header)
- `runs/phase_k/godot_k4_1_pass1_actor_10000/` (Pass 1 godot artifacts)
- `runs/phase_k/godot_k4_1_pass2_actor_2048/` (Pass 2 godot artifacts)

Docs:
- `docs/k4-1-eval-obs-logit-mechanism-evidence.md` (this file)
- `docs/sight-handoff.md` (refresh in chore commit)

No changes to `src/`, `games/`, `tests/`, configs, or the K3.5c
checkpoints.

---

## Reproducibility

```cmd
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe
set PYTHONPATH=C:\Projects\Sight\src
set PYTHONUNBUFFERED=1

python -u tools\k4_1_eval_obs_logit_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seeds 1000-1009 ^
  --max-steps 1800 ^
  --out-dir runs\phase_k ^
  --label k4_1_eval_obs_logit_probe ^
  --k3-5c-2048 runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_2048\model.zip ^
  --k3-5c-10000 runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_10000\model.zip
```

End-to-end wall time on StrongerJr: ~470 s. Background-launch pattern
identical to K4.0: write a `.ps1` wrapper using `Start-Process -Wait
-RedirectStandardOutput -RedirectStandardError -NoNewWindow -PassThru`
and a done-sentinel; launch via cmd
`powershell -NoProfile -Command "Start-Process powershell -ArgumentList ... -NoNewWindow -PassThru"`.

---

## What K4.1 does NOT resolve, and what should be next

K4.1 establishes the deterministic-argmax eval is invariant because
the K3.5c policy is constant-left on its reachable eval distribution.
This is the load-bearing K4 answer for the "bit-identical eval"
anomaly. It does NOT answer:

1. **Why the policy collapses to constant-left in the first place.**
   K4.0 ruled out feature-extractor uniformity (latent_pi diversity
   grows with training). The action head is the seat of the pin.
   Candidates: (a) early-update gradient direction lottery from PPO's
   first batch under seed 0, (b) hazard-spawn asymmetry in
   GodotSignalDodgeEnv biasing early-episode advantages toward "left",
   (c) interaction between the H5 entropy coefficient and the small
   action_net rows. Distinguishing among these requires varying the
   seed (cheapest: re-train at seeds 1, 2, 3 with the same K3.5c
   recipe and observe the dominant argmax direction).

2. **Whether any reachable eval obs has top1-top2 margin small enough
   that a different training regime could flip the argmax.** Per the
   data, the smallest margin observed is 0.2834. A regime that
   produces logit shifts of magnitude >= 0.28 on the eval
   distribution would push some obs across the boundary and
   demonstrate behavior change. K3.5c training between 2048 and 10000
   timesteps produced shifts of L_inf max 0.236, narrowly below this
   bar.

3. **Whether the constant-left attractor is escapable from a fresh
   seed with the same recipe.** This is the K-phase exit question.
   The cheapest experiment: H5 entropy config with seeds {1, 2, 3, 4,
   5} for 10000 timesteps each, then K4.1-style eval-obs probe on
   each, and check whether the argmax distribution is still {left:
   100%} on every seed.

Recommendations for GPT scoping K5 or K-phase-exit:

- **K5.0 fresh-seed sweep:** Train H5 entropy config (10000 timesteps,
  K3.5 reward scaling /30) at seeds 1, 2, 3, 4, 5. Run K4.1-style
  probe on each. Verify whether the constant-action attractor is
  hit on every seed and whether the direction (left/stay/right) is
  the same across seeds. Answers question (3) directly.

- **K5.1 wider regime change:** If K5.0 confirms a consistent
  attractor across seeds, that is the new Phase K landing. The
  research question shifts from "why are eval results bit-identical"
  (now answered) to "what training intervention escapes the
  constant-action attractor on this env." Reward shaping,
  architecture change, and curriculum become candidates again.
