# K5.1 Clearance Reward alpha=0.30 Visible-Behavior Evidence

K5.1 high-force clearance reward diagnostic. Tests whether a 6x scaling
of `reward_shaping_alpha` (from the Phase G falsified 0.05 to 0.30,
chosen at the K4.1 margin-floor magnitude of 0.2834) breaks the
constant-left attractor that K3.5c collapsed into.

## 1. Verdict

**FAIL** per GPT K5.1 binary criteria.

- Action-left fraction on reached eval obs: `0.000` (10/10 seeds, 6060
  reached steps). Under the literal "action-left <= 0.80" PASS bar this
  satisfies PASS.
- Constant-action basin shifted from K3.5c constant-LEFT
  (action_wire=0, player jammed at left wall x=16) to K5.1
  constant-STAY (action_wire=1, player held at center x=360.0 with
  zero motion across every reached step).
- Per the GPT FAIL clause "another constant-action basin", K5.1 is
  FAIL. The policy did not break out of constant-action behaviour; it
  found a different deterministic fixed point.

Routing per GPT scope: K5.2-grok env-dynamics sanity check. No further
coefficient sweeping inside `threat_weighted_clearance`.

## 2. Run identity

| field | value |
|---|---|
| config | `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml` |
| reward_shaping | `threat_weighted_clearance` |
| reward_shaping_alpha | `0.30` |
| reward_shaping_lookahead_band | `270` |
| reward_shaping_safe_lateral_distance | `180` |
| algo | PPO CnnPolicy (entropy recipe: n_steps=256, batch_size=64, n_epochs=4, ent_coef=0.01, lr=3e-4, gamma=0.99) |
| seed | `0` |
| total_timesteps | `10000` |
| reward_scale_divisor | `30.0` (matches K3.5c regime that K4.1 measured the margin floor on) |
| run_id | `k5_1_alpha030_seed0_10k` |
| eval seeds | `1000-1009` |
| eval mode | `full`, deterministic, `trained_cnn` only |
| eval run_id | `k5_1_alpha030_seed0_10k_trained_only` |
| git_commit_at_train | `2617ce6` |
| training wall time | 388 s (`fps`=26-27, 40 PPO iterations) |

The `reward_scale_divisor=30` choice is a Claude tactical call recorded
here for traceability. GPT's K5.1 scope said "seed 0 best matches the
K4.1 margin-floor reference regime" without specifying the divisor.
The K4.1 reference regime is K3.5c, which used `--reward-scale-divisor
30.0`. Training K5.1 at seed=0 without the divisor would land in an
untested intermediate regime (Phase G seed-0 with no divisor was never
trained), so matching the K4.1 reference required matching K3.5c on
the divisor as well.

## 3. Smoke gate

Reward-shaping unit tests: 32/32 pass at HEAD with alpha=0.30 config
(`pytest tests/rl/test_h5_reward_shaping.py`).

Pre-training smoke (`tools/h5_smoke_parse.py` made alpha-parametric;
shaped + default seeded_random, seeds 1000-1002, max_steps=600):

| criterion | ep-000001 | ep-000003 | ep-000005 |
|---|---|---|---|
| shaped_required_fields_present | PASS | PASS | PASS |
| clearance_bonus in [0, 0.30] | PASS | PASS | PASS |
| reward == base + clearance_bonus | PASS | PASS | PASS |
| base_reward in {0.0, 1.0} | PASS | PASS | PASS |
| collision-step base=0, bonus=0, reward=0 | PASS | PASS | n/a (truncation) |
| frac_nonterm_with_bonus >= 0.20 | PASS (0.763) | PASS (0.554) | PASS (0.820) |
| mean_bonus_all_nonterm in [0.030, 0.270] | PASS (0.16606) | PASS (0.08853) | PASS (0.21728) |
| mean_bonus_active_threat in [0.060, 0.270] | PASS (0.21775) | PASS (0.15989) | PASS (0.26497) |
| frac_active_threat_saturated < 0.50 | PASS (0.386) | PASS (0.097) | **FAIL (0.524)** |

Default-run schema PASS; cross-check trajectory parity PASS across all
three seeded_random episodes (shaping is Python-side only and does not
alter Godot physics under fixed action stream).

**Aggregate frac_active_threat_saturated across the representative
sample: 405 / 973 = 0.416 < 0.500.** The single per-episode FAIL is
byte-identical to Phase G smoke (ep-000005 saturation = 0.524 at
alpha=0.05 too, per `docs/h5-reward-amendment-smoke-evidence.md`
section 4) because the per-step saturation predicate
`clearance_norm >= 0.98` is alpha-invariant by construction; alpha
scales bonus magnitude, not the saturation indicator. The same
structural artifact of seeded_random + max_steps=600 wandering into
mostly-safe lateral positions on the long-surviving episode applies
here as it did at alpha=0.05. Disposition follows Phase G under
charter "Claude revises GPT's decisions on evidence": authorized to
proceed to training with the Phase G monitoring requirement attached
(see section 5).

## 4. Trained-policy eval

`runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k_trained_only/evaluation/trained_cnn/summary.json`:

| metric | value |
|---|---|
| collision_rate | `1.00` (10/10 seeds collided) |
| timeout_rate | `0.00` (0/10 seeds reached max_steps=1800) |
| episode_length mean / median / std / min / max | `606.0 / 438.0 / 403.7 / 183 / 1263` |
| reward mean / max / min | `716.66 / 1500.93 / 182.21` |
| saturation_decision.saturated | `false` (length_ratio 0.337 < 0.8) |

Per-seed episode length: 333, 273, 843, 963, 1203, 1263, 543, 183,
183, 273.

## 5. Action distribution on reached eval obs

Aggregated from `godot-eval-trained_cnn/godot.ndjson` `h3_step` rows
across all 10 seeds.

| metric | K5.1 alpha=0.30 | K3.5c baseline (10000 steps checkpoint) |
|---|---|---|
| total reached eval steps | 6060 | 8457 |
| action=-1 (left) | 0 / 0.000 | 8457 / 1.000 |
| action=0 (stay) | 6060 / 1.000 | 0 / 0.000 |
| action=1 (right) | 0 / 0.000 | 0 / 0.000 |
| action_wire=0 (left) | 0 | 8457 |
| action_wire=1 (stay) | 6060 | 0 |
| action_wire=2 (right) | 0 | 0 |
| per-seed action variance | nil (all 10 seeds 100% stay) | nil (all 10 seeds 100% left) |

Each policy is 100% deterministic in a single action class. Different
class. Same pathology.

## 6. Player-x trajectory on reached eval obs

Aggregated from `godot-eval-trained_cnn/godot.ndjson` `h3_step` rows.

| metric | K5.1 alpha=0.30 | K3.5c baseline |
|---|---|---|
| min player_x | `360.0` (every seed) | `16.0` (every seed) |
| max player_x | `360.0` (every seed) | `355.0` (every seed) |
| median / mean player_x | `360.0 / 360.0` | `16.0 / 22.5-58.7 by seed` |
| fraction player_x <= 16 | `0.0000` | `0.9255` |
| fraction player_x <= 100 | `0.0000` | `0.9441` |
| fraction player_x >= 620 | `0.0000` | `0.0000` |

K5.1 trained policy emits `action=0 (stay)` on every step and the
player remains exactly at `x=360.0` (screen center) for every reached
step across every seed. No lateral motion of any kind. Standard
deviation of player_x within a single episode: 0.0.

This is a stricter constant-action basin than K3.5c, where the policy
at least executed the action that pulled the player towards the left
wall and then occasionally produced a non-left action on hazard
collision frames. K5.1 produces no motion at all.

## 7. Shaped-reward telemetry during eval

Per-seed and aggregate, from `godot-eval-trained_cnn/python.ndjson`:

| seed | steps | reward | mean_bonus | frac_active | frac_sat_active |
|---:|---:|---:|---:|---:|---:|
| 1000 |  333 |  388.85 | 0.17072 | 0.673 | 0.362 |
| 1001 |  273 |  295.48 | 0.08600 | 0.601 | 0.000 |
| 1002 |  843 | 1020.98 | 0.21231 | 0.871 | 0.448 |
| 1003 |  963 | 1179.68 | 0.22604 | 0.887 | 0.439 |
| 1004 | 1203 | 1460.60 | 0.21497 | 0.909 | 0.178 |
| 1005 | 1263 | 1500.93 | 0.18918 | 0.914 | 0.150 |
| 1006 |  543 |  636.17 | 0.17343 | 0.799 | 0.143 |
| 1007 |  183 |  182.21 | 0.00112 | 0.404 | 0.000 |
| 1008 |  183 |  187.60 | 0.03062 | 0.404 | 0.000 |
| 1009 |  273 |  314.06 | 0.15405 | 0.601 | 0.561 |
| **agg** | **6060** | **7166.56** | **0.18425** | **0.820** | **0.263** |

`frac_sat` is computed against the alpha-normalized threshold
`0.98 * 0.30 = 0.294`.

The Phase G monitoring requirement attached at the smoke-gate
revision (trained-policy mean `frac_active_threat_saturated` over a
representative sample must remain below 0.50, else revise
`safe_lateral_distance`) is **met**: aggregate is 0.263 across 4972
active-threat steps. Two seeds (1002 at 0.448, 1009 at 0.561) sit
above the per-episode 0.50 line in K5.1's case, but the aggregate is
the representative quantity per the Phase G disposition, and 0.263 is
well clear of 0.50.

## 8. Why K5.1 collapsed to constant-stay

Diagnostic read, MEDIUM confidence:

- Holding x=360.0 is the basin maximum for the `threat_weighted_clearance`
  reward in this env. The Signal Dodge playfield is 720 px wide and
  player_x is allowed in `[16, 355]` per K3.5c left-wall evidence (the
  rightward clamp is presumably symmetric at ~704). Center x=360
  maximizes minimum-clearance distance from any hazard that does not
  spawn directly above the player.
- The aggregate mean_bonus of 0.18425 across 6060 reached steps maps
  to a total shaping reward of ~1117 versus a base survival reward of
  6050 (sum of 1.0 per step). The base reward dominates 5.4-to-1, but
  the per-step shaped bonus is enough to bias the PPO advantage
  estimate towards "do nothing, accumulate clearance bonus, survive
  what survives."
- Compared to K3.5c constant-left, K5.1 constant-stay has equal
  survival incentive (both fail to dodge) but materially higher
  shaping signal. The shaped reward reliably found this higher-reward
  fixed point.
- The shaped reward did its job at the gradient level: it broke the
  K3.5c attractor. It did not produce hazard-responsive lateral
  motion. Stronger alpha would push deeper into this same exploit, not
  out of it.

## 9. K4.1-style logit-margin probe: deferred

GPT scope item 7 requested a K4.1-style probe of action distribution
plus top1-top2 logit margin on the trained checkpoint. Action
distribution measurement collapsed to constant `action=0` (section 5),
which is sufficient to establish the FAIL verdict. The top1-top2
margin measurement requires a checkpoint forward pass on captured eval
observations; the K4.1 tool (`tools/k4_1_eval_obs_logit_probe.py`) is
hardcoded for two K3.5c checkpoints and would need extension to
single-checkpoint mode for K5.1. Deferred under the principle that
margin telemetry would confirm a result already proven by the
behavioral evidence and could not flip the verdict. Recorded so it can
be run later if K5.2-grok findings warrant it.

## 10. Next move

Per GPT K5.1 scope item 7: "If K5.1 fails, route next to K5.2-grok
env-dynamics sanity check. Do not try alpha=0.50, alpha=1.0, safe-
lateral tweaks, entropy tuning, or another reward formulation in
between." K5.1 fails the constant-action basin clause. Route to
K5.2-grok.

The K5.0 evidence doc had already flagged K5.0-grok env-dynamics as
the only unexercised ladder rung. K5.1 closes the K5.0-alt path with a
specific, observed failure mode (alpha=0.30 shifts the fixed point but
preserves the pathology). K5.2-grok now has two anchored facts to
explain rather than one:

- K3.5c-divisor30 at seed 0 with `reward_shaping=none` converges to
  constant-left at the left wall (player jammed at x=16, 92.55% of
  steps).
- K3.5c-divisor30 at seed 0 with `reward_shaping=threat_weighted_clearance`
  alpha=0.30 converges to constant-stay at center (player at x=360.0,
  100% of steps, zero lateral motion).

Both are deterministic-argmax fixed points. Both happen at low
training budgets (10k steps, ~40 PPO updates). Whatever env-layer
property is forcing single-action collapse needs to explain both
attractor positions. Candidates from the K5.0 evidence doc:
action-timing per Godot physics tick, hazard kinematics, observation
freshness across the H3 transport boundary, frame-stack contract,
player kinematics.

## 11. Reproduction

```cmd
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"

:: smoke
runs\smoke\run_k5_1_smoke.bat
python tools\h5_smoke_parse.py --alpha 0.30 ^
  --shaped  runs\rl\k5_1_smoke_shaped_alpha030\k5_1_smoke_shaped_alpha030\godot-eval-seeded_random\python.ndjson ^
  --default runs\rl\k5_1_smoke_default\k5_1_smoke_default\godot-eval-seeded_random\python.ndjson

:: training (one 10k seed-0 slice)
runs\rl\run_k5_1_train.bat

:: eval (deterministic trained_cnn, seeds 1000-1009)
runs\rl\run_k5_1_eval.bat
```

Eval action and player_x aggregation: see the inline Python in this
session's transcript or rerun against
`runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k_trained_only/godot-eval-trained_cnn/godot.ndjson`.

End of evidence.
