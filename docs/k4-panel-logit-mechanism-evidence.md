# K4.0 Panel-Logit Mechanism Diagnostic Evidence

**Date:** 2026-05-18
**Phase:** K (K4.0 panel-logit diagnostic on existing K3.5c checkpoints)
**Commit:** this evidence + tool + handoff refresh
**Classification:** **K4-D + K4-C** with a K4-E-adjacent panel-coverage caveat

---

## Verdict

Cross-model (between K3.5c 2048, K3.5c 10000, fresh seed-0 init):

**K4-D eval-invisible logit drift.** K3.5c 2048 and 10000 panel argmax
match on all 32 panel rows (fraction = 1.000) but raw logits drift
materially between checkpoints (Linf max = 0.2038 >= 0.1). Training
continues to move logits from update 8 through update 156 but never
crosses a deterministic decision boundary on the panel. Fresh seed-0
init differs from both K3.5c checkpoints on every panel row
(0/32 argmax match).

Within-model on K3.5c 10000:

**K4-C action-head decision boundary pinned.** latent_pi dim_std_mean
= 0.0135 (above the diversity threshold 0.01); logit range across
panel rows = 0.0388; mean top1-top2 margin = 0.376; a single argmax
("left") dominates all 32 panel rows (num_det_actions = 1). The action
head produces row-to-row logit variation, but the variation is small
relative to the "left vs stay" gap, so no panel row flips to a
different action.

**K4-E-adjacent caveat (panel coverage):** the 32 nominal panel rows
collapse to ~5-8 effectively-distinct logit signatures. The four
panel_seeds produce IDENTICAL logits across all scripted prefixes
except "initial" (and the "initial" rows differ only at the 0.001
logit scale). The panel does not span the eval rollout observation
distribution. The K4-D / K4-C verdict is therefore confined to "the
action head pins 'left' on a narrow panel"; it does not by itself
explain why two checkpoints with distinct SHA-256 hashes produce
bit-identical per-seed eval results. That layer of the K3.5c anomaly
remains open.

---

## What was run

`tools/k4_panel_logit_probe.py` (new) reuses the fixed
observation-conditioning panel machinery from
`tools/h5_training_entropy_probe.py` (specifically
`build_fixed_observation_panel`, `snapshot_policy_state`, and
`snapshot_action_net`) to probe three policy states on the same panel:

- `fresh_seed0_init`: fresh CnnPolicy under the H5 entropy config at
  seed 0 with zero training steps. Constructed via a stub
  `DummyVecEnv` matching the H5 pixel observation space (uint8
  `(1, 84, 84)`, action space `Discrete(3)`); SB3 seeds the RNG
  inside `_setup_model` after the env is attached, so fresh init
  weights are identical to what the production trainer produces at
  update 0 regardless of env source.
- `k3_5c_2048`: loaded via `PPO.load` from
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_2048/model.zip`.
  SHA-256 `E14D1A12...`.
- `k3_5c_10000`: loaded via `PPO.load` from
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_10000/model.zip`.
  SHA-256 `5664A12E...`.

Panel construction: 4 panel_seeds (730001, 730002, 730003, 730004)
crossed with 8 scripted prefixes (`initial`, `stay_15`, `stay_30`,
`left_15`, `right_15`, `left_15_right_15`, `right_15_left_15`,
`zigzag_30`) = 32 panel rows. Built via one Godot subprocess that is
closed before the SB3 work starts.

Outputs:

- `runs/phase_k/k4_panel_logit_probe.json` (120 KB) - full per-model
  state plus pairwise comparisons plus classification.
- `runs/phase_k/k4_panel_logit_probe_panel_rows.csv` (22 KB) - per-row
  raw logits, softmax probs, deterministic argmax, top1-top2 margin
  for every (model, panel_row) pair.

Wall time: ~70 seconds end-to-end (panel build ~60 s, three forward
passes ~5 s).

---

## Per-model panel summary

| Model            | argmax 100% | mean logit L | mean logit S | mean logit R | mean margin | mean logit range across rows | logit_std L / S / R | cnn_features dim_std_mean | latent_pi dim_std_mean | action_net row norms L / S / R | value pred mean / std |
|------------------|-------------|--------------|--------------|--------------|-------------|------------------------------|---------------------|---------------------------|------------------------|---------------------------------|-----------------------|
| fresh_seed0_init | stay        | 0.000645     | 0.001617     | 0.001166     | 0.000451    | 0.000221                     | 0.000088 / 0.000054 / 0.000054 | 0.008071                  | 0.008071               | 0.0100 / 0.0100 / 0.0100        | -0.074 / 0.011        |
| K3.5c 2048       | left        | 0.425418     | 0.126142     | -0.389215    | 0.299276    | 0.023804                     | 0.010467 / 0.004090 / 0.009492 | 0.005299                  | 0.005299               | 0.0856 / 0.0455 / 0.0825        | 2.229 / 0.073         |
| K3.5c 10000      | left        | 0.384993     | 0.009469     | -0.212723    | 0.375525    | 0.038753                     | 0.014946 / 0.010952 / 0.014659 | 0.013485                  | 0.013485               | 0.1117 / 0.1160 / 0.1038        | 3.164 / 0.225         |

Notes:

- `raw_obs` dim_std_mean is 0.726 on all three models (same panel
  observations). The CNN compresses obs diversity by ~50x into
  features of dim_std_mean ~0.005 to 0.013.
- Fresh init action_net weights are at SB3's default near-zero init
  scale (row norms 0.01, biases 0). All three logits are near zero
  and the argmax is decided by numerical noise at the 0.0004 margin
  scale.
- K3.5c training pushes row norms up roughly 5-10x by update 156. At
  update 156 the three row norms are within 10% of each other
  (0.104 to 0.116), so the "left" preference is not a row-norm
  asymmetry; it is the direction of the action_net.weight rows
  aligning with the latent_pi distribution produced by the rest of
  the network.

---

## Pairwise comparison (row-aligned)

| Pair                                | argmax_match / n_rows | argmax_match_fraction | logit_l1_mean | logit_l1_max | logit_linf_mean | logit_linf_max |
|-------------------------------------|-----------------------|-----------------------|---------------|--------------|------------------|-----------------|
| fresh_seed0_init vs K3.5c 2048      | 0 / 32                | 0.000                 | 0.940         | 0.954        | 0.425            | 0.431           |
| fresh_seed0_init vs K3.5c 10000     | 0 / 32                | 0.000                 | 0.611         | 0.632        | 0.384            | 0.394           |
| K3.5c 2048 vs K3.5c 10000           | 32 / 32               | 1.000                 | 0.334         | 0.380        | 0.176            | 0.204           |

Fresh → K3.5c 2048 has a larger logit displacement (linf_max 0.43)
than K3.5c 2048 → K3.5c 10000 (linf_max 0.20), which is consistent
with first-8-updates-do-most-of-the-work but the next 148 updates
continue moving logits at non-negligible scale.

---

## Within-regime invariance mechanism findings

The K3.5c anomaly being decomposed: two checkpoints with distinct
model SHA-256 (E14D1A12... vs 5664A12E...) produce bit-identical
per-seed external eval results (10/10 length match, 10/10 termination
match across seeds 1000-1009). K4.0 asks what happens to the action
logits on a fixed panel under this regime.

On the panel:

1. The action head pins "left" 100% on all 32 panel rows at both
   2048 and 10000 ts. The "left vs stay" mean gap is 0.30 at 2048
   and 0.38 at 10000.
2. Per-row logit variation is small: range across rows is 0.024 at
   2048 and 0.039 at 10000. The action_net does produce different
   logits on different panel obs, but the variation never exceeds
   the "left vs stay" gap.
3. Between checkpoints, logits drift by linf_max 0.20 per element
   (per-row, per-action). The L1 of the per-row 3-vector delta
   averages 0.33. The drift is real but does not cross any decision
   boundary on the panel.
4. latent_pi diversity grows from 0.005 at 2048 to 0.013 at 10000.
   The feature pipeline is not collapsing; if anything, it gains
   diversity with more training. K4-A (feature extractor uniformity)
   is therefore falsified on K3.5c 10000.

The mechanism on the panel is K4-C: a robust action-head decision
boundary that has settled on "left" and is being further refined
without ever destabilizing. K4-D adds that the refinement is
eval-invisible on the panel - logits move under it, but the panel
argmax does not.

---

## Panel coverage caveat (K4-E-adjacent)

The full 32-row CSV shows that across the 4 panel_seeds the logits
are identical (to numerical precision) for every prefix EXCEPT
"initial", and the "initial" rows differ across panel_seeds only at
the 0.001 logit scale. Effectively the panel exposes ~5-8 distinct
logit signatures, not 32.

This is a property of the panel construction (real Godot rollouts
under scripted action prefixes from a few reset seeds); the scripted
prefixes converge the visible observation to a similar state across
panel_seeds within 15-30 steps.

Implication: K4.0 cannot rule out that the eval rollout obs
distribution contains observations on which K3.5c 2048 and K3.5c 10000
produce different argmaxes. The bit-identical 10-seed eval result is
NOT directly explained by "every eval obs maps to 'left' on both
checkpoints, so trajectories must match"; that hypothesis requires a
panel that actually spans the eval obs distribution.

K4.0 verdict refinement: the deterministic-argmax wedge within the
K3.5 training regime is operationally a tight action-head pin on a
narrow panel. The connection between that pin and the bit-identical
external eval (10 different seeds producing 10 different trajectories
of length 243-1800, but identical between 2048 and 10000) is not
established by this probe.

---

## Why "left"?

Tentative, not pinned down by this probe. Three non-exclusive
candidates:

- **Early-trajectory accident**: the first PPO updates' rollout
  advantages happen to gradient-shape the action_net's "left" row to
  align with the dominant latent_pi direction. Reward scaling at /30
  preserves the relative drift ratio of returns, so once "left" is
  ahead at update 1 the gradient direction tends to reinforce it.
- **Spawn-configuration asymmetry**: GodotSignalDodgeEnv's hazard
  spawn distribution may be slightly asymmetric on the early frames,
  making "left" a marginally better survival action in early
  rollouts. A right-bias under different env seeds would falsify
  this.
- **Argmax-init lottery**: fresh init has near-zero logits; the
  margin at fresh init is 0.0004. The first non-zero gradient update
  effectively picks a winner. The training rng state at seed 0 +
  PPO's first batch could be producing a left-favoring gradient.

None of these is testable from the K4.0 artifact alone.

---

## Files added or changed this session

Code:

- `tools/k4_panel_logit_probe.py` (new, ~740 lines)

Runs:

- `runs/phase_k/k4_panel_logit_probe.json`
- `runs/phase_k/k4_panel_logit_probe_panel_rows.csv`
- `runs/phase_k/godot_k4_panel_logit_probe_fixed_panel/` (panel build
  artifacts)

Docs:

- `docs/k4-panel-logit-mechanism-evidence.md` (this file)
- `docs/sight-handoff.md` (refresh in chore commit)

No changes to `src/`, `games/`, `tests/`, or the K3.5c checkpoints.

---

## Reproducibility

```cmd
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe
set PYTHONPATH=C:\Projects\Sight\src
set PYTHONUNBUFFERED=1

python -u tools\k4_panel_logit_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 0 ^
  --out-dir runs\phase_k ^
  --label k4_panel_logit_probe ^
  --k3-5c-2048 runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_2048\model.zip ^
  --k3-5c-10000 runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_10000\model.zip
```

Background-launch pattern used this session (validated): write a
PowerShell launcher .ps1 that uses `Start-Process -Wait
-RedirectStandardOutput -RedirectStandardError -NoNewWindow -PassThru`
for the Python child, then writes a done-sentinel with the exit code.
Launch the .ps1 from cmd via
`powershell -NoProfile -Command "Start-Process powershell ... -NoNewWindow -PassThru"`.
This survives the 4-minute MCP `interact_with_process` ceiling and
captures stdout / stderr to disk.

---

## Recommendations for K4.1+

K4.0 confirmed the action-head pin and the eval-invisible logit drift
on a narrow panel. Two layers remain for K4.x to address.

1. **K4.1 - eval-obs panel.** Replace the scripted-prefix panel with
   real obs captured from the actual trained-only eval rollouts
   (seeds 1000-1009 against K3.5c 2048 and K3.5c 10000). Rerun the
   logit probe on those obs. Tests whether the action-head still
   pins one action 100% on the actual eval distribution, or whether
   the eval distribution includes obs that produce different
   argmaxes between K3.5c 2048 and K3.5c 10000 (which would explain
   the external eval bit-identical result by forcing the policy
   trajectories to coincide step by step). This is the load-bearing
   K4 follow-up.

2. **K4.2 - argmax flip timing.** Instrument the production trainer
   (or rerun the K0-K3 training-time probe under the K3.5 reward
   scale) to record per-update panel argmax distribution. Establishes
   when fresh "stay" flips to "left" and whether the flip is
   monotonic. Cheaper than K4.1 but explains less.

3. If K4.1 fires K4-E (panel and eval argmax surfaces diverge), the
   within-regime invariance mechanism shifts from "action-head fixed
   point" to "two checkpoints producing the same step-by-step action
   trace on eval despite different weights" - which is the original
   Phase K anomaly signature applied inside the K3.5 regime, and the
   K4 question becomes a comparison of action sequences along the
   eval rollouts themselves rather than logits on a held-out panel.
