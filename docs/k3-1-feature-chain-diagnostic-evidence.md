# K3.1 Feature-Chain Diagnostic Evidence

## Context

K3 closed with both pi=[64]/vf=[128] and pi=[64]/vf=[256] failing the K3 contract as constant-action attractors on the 32-item fixed observation-conditioning panel. The value-head capacity hypothesis was falsified. The K3 evidence showed cross-panel logit_std O(1e-5) on vf128 and O(1e-7) on vf256, which is policy observation-blindness, but the locus of variance collapse inside the actor-critic network was not yet identified.

K3.1 adds a feature-chain diagnostic to `snapshot_policy_state` in `tools/h5_training_entropy_probe.py`. The diagnostic captures per-dim std across the 32 panel items at six points along the forward pass:

  raw_obs (flat) -> cnn_features (NatureCNN output) ->
  latent_pi (mlp_extractor pi branch) / latent_vf (mlp_extractor vf branch) ->
  action_logits (action_net output) / value_predictions (value_net output)

For each tensor stage the snapshot reports `total_dims`, `dim_std_mean`, `dim_std_min`, `dim_std_max`, `dim_std_median`, and `n_dims_above_eps` (count of dims with per-dim std > 1e-6). For `value_predictions` (scalar per panel item) it reports `n`, `mean`, `std`, `min`, `max`.

Smoke at seed 3, 2048 ts, pi=[64], vf=[128] passed: new field populated correctly, schema stable, runtime added ~1s over the K3 smoke baseline.

## Run

  Label: `k3_1_seed3_pi64_vf128_feature_chain_10k`
  Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`
  Seed: 3
  Timesteps: 10000 (40 PPO updates at n_steps=256)
  Arch: pi=[64], vf=[128], shared CNN feature extractor (NatureCNN, 512-dim output)
  Elapsed: similar to K3 vf128 (~5.2 minutes)

## Final-update post-update fixed-panel feature-chain snapshot

| Stage | total_dims | live dims (std > 1e-6) | dim_std_mean | dim_std_max |
|---|---|---|---|---|
| raw_obs | 7056 | 96 | 7.26e-01 | 7.75e+01 |
| cnn_features | 512 | 129 | 4.64e-03 | 4.56e-02 |
| latent_pi | 64 | 40 | 3.42e-05 | 3.03e-04 |
| latent_vf | 128 | **0** | 5.57e-10 | 2.58e-08 |
| action_logits | 3 | 3 | 8.15e-06 | 1.21e-05 |
| value_predictions | 32 (scalars) | std = 0.0 exactly, all panel items predict 33.9500 |

## Initial state (update 0 pre-update) for contrast

| Stage | total_dims | live dims | dim_std_mean | dim_std_max |
|---|---|---|---|---|
| raw_obs | 7056 | 96 | 7.26e-01 | 7.75e+01 |
| cnn_features | 512 | 258 | 6.71e-03 | 4.00e-02 |
| latent_pi | 64 | **64 (all)** | 1.29e-02 | 3.04e-02 |
| latent_vf | 128 | **128 (all)** | 1.24e-02 | 3.25e-02 |
| action_logits | 3 | 3 | 1.42e-04 | 1.66e-04 |
| value_predictions | 32 | mean -0.0690, std 7.65e-03, range [-0.0741, -0.0504] |

At initialization the policy network is fully observation-conditioning. All 64 latent_pi dims and all 128 latent_vf dims show nonzero variance across the 32 panel items. value_predictions have a small but real range of about 0.024.

## Collapse trajectory

Selected post-update snapshots showing the latent_vf collapse timeline:

| Update | cnn_live | latent_pi live | latent_vf live | latent_vf max_std | value_std |
|---|---|---|---|---|---|
| 1 | 140 | 57 | 70 | 1.00e-02 | 6.11e-04 |
| 2 | 129 | 45 | 2 | 7.26e-03 | 1.05e-04 |
| **3** | 129 | 41 | **0** | 2.80e-07 | **0.0** |
| 10 | 129 | 40 | 0 | 1.58e-07 | 0.0 |
| 20 | 129 | 40 | 0 | 7.44e-08 | 0.0 |
| 30 | 129 | 40 | 0 | 3.57e-08 | 0.0 |
| 40 | 129 | 40 | 0 | 2.58e-08 | 0.0 |

By update 3 the value-head latent representation has collapsed to zero cross-panel variance. It never recovers. value_predictions are bit-identical across all 32 panel observations from update 3 onward.

cnn_features drop from 258 live dims at init to 129 by update 2, then stabilize. The CNN itself does not collapse further. max_std on cnn_features stays at 0.046 throughout.

latent_pi keeps 40 of 64 dims live throughout, with max_std oscillating between 3e-4 and 1.5e-2. The policy MLP retains observation-conditioning but at a small magnitude.

action_net (Linear 64 -> 3) compresses latent_pi to action_logits with max_std O(1e-5), which is what produces the cross-panel logit_std O(1e-5) seen in K3.

## Mechanism

The locus of variance collapse is the value MLP head, specifically the part of `mlp_extractor` that produces `latent_vf`. Within 3 PPO updates that path is trained to output an effectively constant 128-dim vector regardless of input. value_net then maps that constant to the constant value 33.95.

A plausible driver: Signal Dodge returns are dominated by episode length plus a small survival shaping term. If the return distribution across the rollout buffer is nearly identical regardless of observation (because the early policy is near-uniform and survival is short and uniform), then the L2 value loss is minimized by predicting a single scalar everywhere. The shared `mlp_extractor` has separate pi and vf MLP sub-modules; the value-MLP submodule is the only path that can be flattened to satisfy the value loss, so PPO drives it to a degenerate constant. Meanwhile the policy loss has weaker gradient pressure (clip range, advantage near zero once values are constant), so latent_pi keeps a small amount of variance but action_net's small init absorbs nearly all of it.

The previously hypothesized "unshared actor/critic feature extractors" (GPT's branch b) would give value its own CNN, but cnn_features are NOT collapsing in the shared regime. Unsharing the CNN does not address the value MLP head collapse, which happens downstream of the CNN regardless of sharing. The mechanism is inside `mlp_extractor.value_net` and is driven by the return distribution and the value loss, not by gradient interference between pi and vf at the CNN.

## What this rules in and rules out

Ruled out by this evidence:
- "Pixel preprocessing is broken" - raw_obs has 96 live pixel dims with max std 77.5, and cnn_features stably produces 129 live dims with max std 0.046. The CNN sees diverse input and outputs diverse features. GPT's branch (a) audit of pixel pipeline is not needed.
- "Value-head capacity is the bottleneck" - K3 already falsified this. K3.1 confirms by showing the latent_vf collapse is qualitative (variance vanishes), not quantitative (more dims would not unflatten it; vf256 was worse than vf128 in K3).
- "Shared CNN squashes pi-relevant features under value loss pressure" - cnn_features have stable variance across all 40 updates. The CNN is not being flattened.

Ruled in:
- The value MLP submodule of `mlp_extractor` is the variance killer.
- The return distribution is the upstream driver. If returns are nearly constant across observations, the L2-optimal value function is a constant, and the optimizer correctly finds it.

## Recommended next direction (for GPT)

Two diagnostics that would discriminate the next-best fix without running another arch sweep:

1. Returns-distribution probe. Inspect `rollout_buffer.returns` cross-observation variance and `rollout_buffer.advantages` cross-observation variance per PPO update. If returns std is near zero at update 0 and stays near zero, the value head is correctly degenerate and the problem is the reward function (or episode-length distribution under the initial random policy). If returns have real variance but the value loss still drives latent_vf to zero, the optimizer or value loss weighting is at fault.

2. Action-net init gain. The policy collapses despite latent_pi keeping ~40 live dims. SB3's default `ortho_init` uses gain=0.01 for the action head. Cross-observation latent_pi variance of ~0.01 multiplied by an action_net with weights at gain ~0.01 produces logit cross-observation variance of ~1e-4, very close to the action_logits max_std at init (1.66e-4) and the failure mode at end (1.2e-5). The action_net may be initialized so small that it cannot transmit observation-conditioning into the logits even when latent_pi has it. Test: bump the action_net init gain (or remove the small-gain init) and rerun. This is a one-line policy_kwargs change.

Direction 1 is the cleaner diagnostic - it isolates the upstream driver and is independent of architecture choices. Direction 2 is a cheap targeted experiment. Either is more informative than a fresh capacity sweep or an unshared-CNN refactor.

## Schema delta

`snapshot_policy_state` now returns `feature_chain_diversity` with these keys:

  share_features_extractor: bool
  raw_obs / cnn_features / latent_pi / latent_vf / action_logits: per-stage dict
    {total_dims, dim_std_mean, dim_std_min, dim_std_max, dim_std_median, n_dims_above_eps, eps}
  value_predictions: {n, mean, std, min, max}

This field is now present on every `pre_update.fixed_panel_policy_state` and `post_update.fixed_panel_policy_state` block in NDJSON digest rows, and on `initial_fixed_panel_policy_state` and `final_fixed_panel_policy_state` in the summary.

The function assumes `share_features_extractor=True` (default for SB3 ActorCriticCnnPolicy). If a future run sets it False, the existing extract_features-then-mlp_extractor chain in this function will need adapting. The flag is reported in the diagnostic for audit.

## Reproduction

```bat
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
cd /d C:\Projects\Sight
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 3 --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label k3_1_seed3_pi64_vf128_feature_chain_10k ^
  --policy-net-arch-pi 64 --policy-net-arch-vf 128
```

Artifacts (runs/ is gitignored):

- `runs\phase_k\k3_1_seed3_pi64_vf128_feature_chain_10k.{ndjson,summary.json}`
- `runs\phase_k\k3_1_smoke_seed3_pi64_vf128_feature_chain.{ndjson,summary.json}`

Bat-with-sentinel templates at `C:\Users\maste\AppData\Local\Temp\sight_k3_1_smoke\run_smoke.bat` and `C:\Users\maste\AppData\Local\Temp\sight_k3_1_10k\run.bat`.
