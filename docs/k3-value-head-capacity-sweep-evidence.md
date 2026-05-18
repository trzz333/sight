# K3 Value-Head Capacity Sweep Evidence

## Context

K3 instrumentation (commit `97a2642`) added a 32-item fixed observation-conditioning panel (4 reset seeds x 8 scripted action prefixes), the value-head CLI overrides `--policy-net-arch-pi` / `--policy-net-arch-vf`, and the gates `observation_conditioning_min_bar` (`top_argmax_fraction < 0.95 AND num_det_actions >= 2 AND max_explained_variance > 0.0`) and `observation_conditioning_better_bar` (`top < 0.80 AND all three actions used AND max EV > 0`). Smoke at 2048 ts seed 3 pi=[64] vf=[128] passed.

This document records the first real capacity sweep: two 10000-ts runs at seed 3, entropy recipe `signal_dodge_ppo_h5_pixel_entropy.yaml`, pi=[64], vf=[128] then vf=[256]. Ran serially, not parallel.

## Results

| Field | vf128 | vf256 |
|---|---|---|
| Label | `value_head_capacity_seed3_pi64_vf128_fixed_panel` | `value_head_capacity_seed3_pi64_vf256_fixed_panel` |
| Elapsed | 312.2 s | 313.1 s |
| n_updates | 40 | 40 |
| Probe verdict | K-C | K-C |
| `observation_conditioning_min_bar` | **false** | **false** |
| `observation_conditioning_better_bar` | **false** | **false** |
| `final_fixed_panel_policy_state.top_argmax_action` | stay | stay |
| `final_fixed_panel_policy_state.top_argmax_fraction` | 1.0 | 1.0 |
| `final_fixed_panel_policy_state.num_det_actions` | 1 | 1 |
| `max(per_update_digest[].explained_variance)` | 0.002202 | 0.000207 |
| `min(per_update_digest[].explained_variance)` | -1.19e-07 | -1.62e-04 |
| `last.fixed_panel_constant_action_attractor` | **true** | **true** |

## Per-run inner snapshot, final update, post-update

### vf128 (`post_update.fixed_panel_policy_state`)

- n = 32
- entropy across panel: mean 0.91802, min 0.91802, max 0.91803 (spread < 2e-5)
- margin across panel: mean 0.31573, min 0.31572, max 0.31573
- argmax_fractions: left 0.0, stay 1.0, right 0.0
- det_argmax_counts: left 0, stay 32, right 0
- mean_probs: left 0.385, stay 0.528, right 0.088
- mean_logits: left 0.443, stay 0.759, right -1.038
- **logit_std across the 32 panel items: left 4.84e-06, stay 7.48e-06, right 1.21e-05**
- prob_ranges: each action's prob varies by < 5e-6 across all 32 panel items

### vf256 (`post_update.fixed_panel_policy_state`)

- n = 32
- entropy across panel: mean 1.05062, min 1.05061, max 1.05062 (spread < 2e-7)
- margin across panel: mean 0.59463, min 0.59463, max 0.59463
- argmax_fractions: left 0.0, stay 1.0, right 0.0
- det_argmax_counts: left 0, stay 32, right 0
- mean_probs: left 0.267, stay 0.483, right 0.250
- mean_logits: left -0.225, stay 0.370, right -0.287
- **logit_std across the 32 panel items: left 5.08e-07, stay 3.75e-07, right 1.72e-07**
- prob_ranges: each action's prob varies by < 4e-7 across all 32 panel items

## Classification per GPT K3 contract

- **Candidate mechanism win** (`min_bar` true): not met on either run.
- **Stronger mechanism win** (`better_bar` true): not met on either run.
- **Weak improvement** (`max EV > 0 AND final det argmax constant`): technically met on both (max EV positive, 0.0022 and 0.0002), but the EV magnitudes are within sample noise of zero (min EV is negative on both, max EV is order of magnitude of fluctuation), so the "improvement" is statistical noise not signal.
- **Failure** (`fixed_panel_constant_action_attractor = true` on final update OR `top_argmax_fraction >= 0.95` OR `num_det_actions < 2`): all three sub-criteria met on both runs.

**Verdict: Failure on both. Constant-action attractor (stay) on the fixed observation-conditioning panel for vf=[128] and vf=[256] alike. No deployment eval.**

## Meta finding: the policy network is observation-blind at 10k ts

The fixed panel contains 32 distinct observations produced by 4 different reset seeds combined with 8 distinct scripted action prefixes. These observations are genuinely different pixel states. Yet the network's post-update logits vary across these 32 inputs by **O(1e-5) on vf128 and O(1e-7) on vf256**. Probabilities vary by O(1e-6) or smaller. This is not a value-head capacity problem. The CNN feature extractor + policy head is producing essentially constant outputs regardless of input.

Counter-intuitively, vf=[256] is *worse* than vf=[128] on cross-panel variance by roughly two orders of magnitude. Increasing value-head capacity did not buy observation-sensitivity, and may have made the policy head converge to a flatter constant faster via shared-feature feedback.

The value-head capacity hypothesis is falsified by this slice. The next investigation target is upstream of the value head: CNN feature extractor behavior on these observations, or the optimization dynamics that collapse the policy to constant `stay` within the first ~10 updates.

## Schema reconciliation against GPT's contract

GPT's contract specified four flat keys on per_update_digest rows: `fixed_panel_constant_action_attractor`, `fixed_panel_det_argmax_counts_post`, `fixed_panel_logit_std_post`, `fixed_panel_prob_ranges_post`. The actual K3 instrumentation schema has only `fixed_panel_constant_action_attractor` flat at the digest-row level; the other three live inside the digest row at `post_update.fixed_panel_policy_state.det_argmax_counts`, `.logit_std`, and `.prob_ranges` respectively (and analogously at `pre_update.fixed_panel_policy_state.*`). All requested values are present; the names are not flat. No instrumentation change is required, but the contract template should be updated to address the nested path so post-run extractors do not look for non-existent flat keys.

Summary `final_fixed_panel_policy_state` is the equivalent of the last row's `post_update.fixed_panel_policy_state`, so its `top_argmax_action` / `top_argmax_fraction` / `num_det_actions` are accessible at the summary top level.

## Self-correction

The first-pass extraction report named max EV as 0.0000 across all 40 updates on both runs, derived from the log-tail PPO digest line (which prints with `%.4f` precision and rounds 0.0022 to 0.0000). The correct max EVs are 0.0022 (vf128) and 0.00021 (vf256), both still effectively zero given the negative min-EV floor, but technically positive. This does not change the classification.

## Reproduction

```bat
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
cd /d C:\Projects\Sight
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 3 --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label value_head_capacity_seed3_pi64_vf128_fixed_panel ^
  --policy-net-arch-pi 64 --policy-net-arch-vf 128
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 3 --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label value_head_capacity_seed3_pi64_vf256_fixed_panel ^
  --policy-net-arch-pi 64 --policy-net-arch-vf 256
```

Artifacts (runs/ is gitignored):

- `runs\phase_k\value_head_capacity_seed3_pi64_vf128_fixed_panel.{ndjson,summary.json}`
- `runs\phase_k\value_head_capacity_seed3_pi64_vf256_fixed_panel.{ndjson,summary.json}`

Bat-with-sentinel templates at `C:\Users\maste\AppData\Local\Temp\sight_k3_vf128\run_vf128.bat` and `C:\Users\maste\AppData\Local\Temp\sight_k3_vf256\run_vf256.bat`.
