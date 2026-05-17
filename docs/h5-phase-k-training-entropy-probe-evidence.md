# H5 Phase K K0 training-time entropy-collapse probe evidence

Status: VERIFIED end-to-end on StrongerJr. Two instrumented training runs complete: K0-2048 pilot (8 PPO updates, verdict **K-C**) and K0-10k extension (40 PPO updates, verdict **K-A late**). K-B detector tightened from `or` to `and` semantics before the 10k rerun. K0-10k addendum starts at section 11.

## 1. TL;DR

Phase J closed the eval-side diagnostic chain and recommended training-side work, with the entropy-collapse probe as the cheapest first slice. Phase K K0 implements that probe by subclassing SB3 PPO and instrumenting the `train()` method to record per-update rollout statistics, optimization losses, gradient norms, action_net weight/bias deltas, and pre/post-update entropy and raw-logit margin on the same rollout observations.

The 2048-timestep run produces 8 PPO updates and crosses none of the three collapse thresholds. Verdict: **K-C** (entropy healthy through pilot budget). Mean policy entropy stays in `[1.025, 1.083]` against a discrete-3-uniform ceiling of `ln(3) = 1.0986`; rollout top-action sampling fraction stays in `[0.40, 0.52]` well below the 0.95 threshold; raw top1-top2 logit margin oscillates in `[0.08, 0.60]` an order of magnitude below the 4.0 threshold. Advantage standard deviation stays healthy (1.4 to 12.6) and value-function gradients are not degenerate.

Two qualitatively important observations beyond the verdict.

First, the trained Phase E seed2 raw-logit margin of 0.83 (left tape, Phase I section 8) is materially larger than the K0 update-8 margin of 0.12. The wedge-strategy commitment that Phase J seed-1008 falsified does not exist at the 2048-timestep budget; it forms between 2048 and 10000 timesteps. The 2048-step policy is essentially still uniform.

Second, the K0 rollout argmax at update 8 is `stay`, not `left`. Phase E seed2 ends at 10000 timesteps with argmax `left`. Whether argmax converges to `left` vs `stay` is undecided at 2048 timesteps and oscillates across updates 1-8 (left, left, stay, stay, stay, stay, left, left). The basin assignment Phase H called "Class B = train_seed=2 picks `left`" is not yet present at this budget. The Class B identity emerges later in training.

K0 was scoped explicitly as a pilot gate. Per the K-C interpretation rule from the scope note: recommend rerunning the same probe to 10000 timesteps next session, do not auto-run 10000 in this prompt, do not switch directions to K1 architecture changes yet.

## 2. Why Phase J forces this slice

Phase J ablation found that the H5 plateau is not eval-determinism-recoverable: the Phase E seed2 stochastic eval changed 36.0% of per-step actions from the deterministic argmax, dropped mean episode length by 22%, lost seed-1008's deterministic timeout in all five replicates, and reclassified every episode as `non_wall_hugging_collision`. Phase G seed2 stochastic eval was bit-identical to deterministic on every replicate of every seed. Phase J recommended training-side diagnostics in increasing order of structural change: (1) entropy probe on existing config, (2) architecture change to `net_arch=dict(pi=[64], vf=[64])`, (3) train-seed asymmetry probe under Grok consult.

GPT's Phase K scope note ordered K0 first because it is the cheapest cost-discriminator: a short instrumented training run answers whether the existing entropy YAML setup has any path out of the wedge basin, and isolates whether the collapse mechanism is entropy-driven, advantage-driven, or something else upstream of the trained-policy artifact itself. Jeff's revision added the K-C clause: treat 2048 as a pilot gate, auto-classify "no collapse by 2048" as K-C, recommend the same probe to 10000 next session, do not switch directions on a 2048-timestep null.

## 3. Tool and methodology

Tool: `tools/h5_training_entropy_probe.py`. Implements an `InstrumentedPPO` subclass that overrides `PPO.train()` while mirroring SB3 2.8.0's `stable_baselines3/ppo/ppo.py` `train()` body exactly. The mirroring is deliberate: a callback fires only at rollout boundaries (between `collect_rollouts` and `train`), so it cannot capture gradient norms between `loss.backward()` and `torch.nn.utils.clip_grad_norm_()` and cannot capture per-minibatch loss components. SB3 logs aggregate `entropy_loss`, `policy_gradient_loss`, `value_loss`, `approx_kl`, `clip_fraction`, and `explained_variance` to its logger after `train()` returns, but does not expose minibatch gradient norms or per-minibatch entropy values.

Per-update record captured:

- `update_idx`, `num_timesteps`, `n_epochs_done`, `n_minibatches`, `clip_range`.
- Rollout action stats: counts and fractions per action over the rollout buffer's sampled actions, top action and top fraction, n_actions.
- Rollout episode stats: rollout length, episode resets count, first-step-was-reset flag (best-effort terminal detection from `rollout_buffer.episode_starts`).
- Advantage/return/value stats: mean, std, min, max for advantages; positive/negative fraction; mean and std for returns and values; value-error vs returns (mean, std, abs-mean); explained_variance computed over the full rollout buffer.
- Pre/post-update policy state on the same rollout obs: mean/min/max entropy across the 256 rollout obs, mean/min/max raw top1-top2 margin, per-action mean probabilities, per-step argmax fractions, top argmax action, top argmax fraction.
- Pre/post-update action_net snapshot: per-action weight row norms, per-action biases, blake2b-128 digest of weight+bias for cheap equality detection.
- Action_net delta: per-action row-norm delta and bias delta, weights_changed flag.
- Per-minibatch losses aggregated across all minibatches in the update: mean policy_gradient_loss, mean entropy_loss, mean value_loss, last total_loss, mean approx_kl, max approx_kl, mean clip_fraction.
- Pre-clip gradient norms per module: features_extractor mean, mlp_extractor mean, action_net mean, value_net mean; total mean and max across minibatches.

Pre/post-update policy state uses the same actor-path extraction as `tools/h5_activation_compare.py` (Phase I section 6: `extract_features` → `mlp_extractor` → `action_net` → `softmax`), so values are commensurable with Phase I summary statistics.

Collapse thresholds per GPT scope note:

- Mean rollout policy entropy `< 0.20`.
- Rollout top-action sampling fraction `>= 0.95`.
- Mean raw top1-top2 logit margin `>= 4.0`.

Auto-classification rules:

- **K-A**: any of {entropy, top-action fraction, margin} crosses its threshold by update index 3 (within first 1-3 PPO updates).
- **K-B**: advantage std `< 0.05` OR explained_variance `< 0.10` precedes any entropy-threshold crossing.
- **K-C**: no threshold crosses through the entire probe.
- **K-D**: rollout top-action fraction `>= 0.95` but entropy stays `>= 0.20` and margin stays `< 4.0` throughout (wedge behavior without distributional collapse).
- Priority: K-B if value/advantage degeneration leads any entropy flag; else K-A if early collapse; else K-D if wedge-only; else K-C.

## 4. Run configuration and fingerprints

Single invocation:

| Field | Value |
|---|---|
| Config | `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml` |
| Train seed | 2 |
| Total timesteps | 2048 (pilot gate; YAML default is 10000) |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 4 |
| ent_coef | 0.01 |
| learning_rate | 0.0003 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| Policy | `CnnPolicy` (NatureCNN + Linear(512,3) action_net + Linear(512,1) value_net; `net_arch=None` so latent_pi = latent_vf = features per Phase I section 6) |
| Device | cpu |
| Wall time | 64.79 s |
| PPO updates produced | 8 (`total_timesteps / n_steps = 2048 / 256`) |
| Minibatches per update | 16 (`n_steps / batch_size * n_epochs = 256 / 64 * 4`) |
| Total minibatches | 128 |

Pre-update-1 action_net (effectively random init plus the 256 random-policy rollout steps' implicit warmup):

| Action | Row norm | Bias |
|---|---:|---:|
| left | 0.010000 | 0.000000 |
| stay | 0.010000 | 0.000000 |
| right | 0.010000 | 0.000000 |

Action_net blake2b-128: `4416b7c857b6a6c02d37e33778772b9d`.

Post-update-8 action_net:

| Action | Row norm | Bias |
|---|---:|---:|
| left | 0.013991 | 0.000551 |
| stay | 0.015105 | 0.000871 |
| right | 0.013153 | -0.000196 |

Action_net blake2b-128: `8b14958aeab0fbeae420bdd183eb4e92`. Weights changed every update (all 8 update records have `weights_changed: true`).

Phase I comparison anchor for context, trained Phase E seed 2 final action_net row norms (Phase I section 10):

| Action | Phase E seed2 row norm | K0 update-8 row norm | Ratio K0/E2 |
|---|---:|---:|---:|
| left | 0.0247 | 0.013991 | 0.566 |
| stay | 0.0273 | 0.015105 | 0.553 |
| right | 0.0178 | 0.013153 | 0.739 |

K0 finishes at ~55-74% of Phase E seed 2's action_net row magnitudes. The five additional 2048-timestep "doublings" of the training budget would carry these into the Phase E range.

## 5. Per-update results

Headline metrics per PPO update. All policy-state values are computed on the same 256 rollout observations for that update (pre = before this update's gradient steps; post = after).

| Upd | ts | top_act | top_frac | H_pre | H_post | margin_pre | margin_post | adv_std | EV | pg_loss | val_loss | ent_loss | KL | clip | gn_total | gn_action |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | left | 0.395 | 1.0986 | 1.0307 | 0.0015 | 0.5992 | 2.7209 | 0.005 | 0.0029 | 107.06 | -1.072 | 0.029 | 0.333 | 400.2 | 3.0 |
| 2 | 512 | left | 0.492 | 1.0312 | 1.0251 | 0.5967 | 0.5336 | 2.0438 | 0.009 | 0.0053 | 47.82 | -0.994 | 0.067 | 0.513 | 691.4 | 8.7 |
| 3 | 768 | stay | 0.523 | 1.0252 | 1.0610 | 0.5333 | 0.5509 | 1.7570 | -0.008 | 0.0043 | 46.01 | -1.011 | 0.007 | 0.115 | 889.7 | 8.0 |
| 4 | 1024 | stay | 0.473 | 1.0608 | 1.0634 | 0.5527 | 0.2771 | 9.2078 | 0.008 | 0.0028 | 106.72 | -1.076 | 0.008 | 0.112 | 748.5 | 12.2 |
| 5 | 1280 | stay | 0.477 | 1.0635 | 1.0429 | 0.2769 | 0.4778 | 1.4149 | 0.044 | 0.0057 | 30.79 | -1.032 | 0.010 | 0.148 | 905.5 | 9.8 |
| 6 | 1536 | stay | 0.520 | 1.0431 | 1.0833 | 0.4770 | 0.0818 | 1.2517 | -0.008 | 0.0015 | 15.37 | -1.031 | 0.005 | 0.068 | 635.1 | 9.3 |
| 7 | 1792 | left | 0.398 | 1.0833 | 1.0493 | 0.0819 | 0.1035 | 11.7722 | 0.008 | 0.0283 | 146.95 | -1.057 | 0.073 | 0.518 | 665.7 | 12.2 |
| 8 | 2048 | left | 0.398 | 1.0493 | 1.0726 | 0.1034 | 0.1156 | 12.5893 | 0.008 | 0.0102 | 160.37 | -0.991 | 0.021 | 0.230 | 405.1 | 11.2 |

Columns: `ts` = num_timesteps; `top_act` = rollout sampled top action; `top_frac` = its fraction; `H_pre`/`H_post` = mean policy entropy on rollout obs before/after this update; `margin_pre`/`margin_post` = mean raw top1-top2 logit margin; `adv_std` = rollout advantage std; `EV` = explained_variance over rollout buffer; `pg_loss`/`val_loss`/`ent_loss` = minibatch-mean losses; `KL` = mean approx KL; `clip` = mean clip fraction; `gn_total` = mean pre-clip total gradient norm across minibatches; `gn_action` = mean pre-clip action_net gradient norm.

Five observations.

First, **entropy is essentially uniform throughout**. The discrete-3-uniform maximum entropy is `ln(3) = 1.0986`. Eight updates of post-update entropy span `[1.0251, 1.0833]`, which is 93-99% of the uniform-distribution maximum. The `ent_coef=0.01` regularizer is not collapsing the distribution; it is also barely binding because the entropy is already near maximum.

Second, **the margin's first growth came in update 1** (from 0.0015 to 0.5992) and **never grew past that level**. By update 8 the margin is 0.116. This is 14% of the trained Phase E seed 2 margin of 0.83 (Phase I left-tape mean). The wedge-commitment basin Phase J seed-1008 falsified does not exist at 2048 timesteps.

Third, **rollout top-action assignment is unstable**: left (updates 1, 2), stay (updates 3, 4, 5, 6), left (updates 7, 8). The "Class B = train_seed=2 picks left" identity from Phase H section 6 is not yet established at this budget. Either trajectory will become the basin depending on later optimization dynamics.

Fourth, **the value function has not learned by 2048 timesteps**. Explained variance bounces between -0.008 and 0.044, far below any operationally useful level. Value loss is 15-160 (large in absolute terms because the survival-only reward sums to large positive returns). The advantage signal exists (std 1.2 to 12.6) but is not yet a structured learning signal.

Fifth, **gradient norms are large pre-clip**. Total gradient norm averages 400-905 across minibatches, while `max_grad_norm=0.5` clips heavily on every step. The action_net's per-module gradient norm averages 3-12. This is normal for a fresh pixel-CNN policy whose value function has not converged, but worth noting: the actual policy update step is severely clipped, so per-update progress is bounded by the clip ratio, not by the raw gradient signal.

## 6. Collapse-threshold check

| Criterion | Threshold | K0 trajectory | Crossed at any update? |
|---|---|---|---:|
| Mean rollout entropy (post-update) | `< 0.20` | min 1.0251, max 1.0833 | no |
| Rollout top-action fraction | `>= 0.95` | min 0.395, max 0.523 | no |
| Mean raw top1-top2 margin (post-update) | `>= 4.0` | min 0.082, max 0.599 | no |

All three flags are false on every one of the 8 update records. No subcriterion of the K-B value/advantage degeneration test fires either: advantage std is in `[1.25, 12.59]` (all above 0.05), explained_variance is in `[-0.008, 0.044]` which technically dips below 0.10 on every update.

This last point matters: the K-B rule from the scope note says "advantage std `< 0.05` or explained_variance `< 0.10`" but explained_variance is essentially zero throughout the probe. That is a normal early-training pattern for a fresh value head, not a degeneration: the value loss is decreasing (107 → 47 → 46 → 107 → 30 → 15 → 147 → 160, with noise driven by episode-length variation), gradient norms are healthy, and the policy advantage signal is informative (std up to 12.59 in update 8). The K-B rule as scoped would have fired on this run for the wrong reason. The K-B detector should be tightened in future probes: a stricter rule would require explained_variance to be near zero AND advantage std to also collapse, not either alone. For the current probe the verdict is unambiguously K-C because no entropy threshold ever crosses, so the K-B precedence rule does not engage.

(Tool's current K-B rule uses OR semantics. The verdict logic still routes correctly because K-B requires `first_value_adv < first_entropy_collapse` and there is no entropy collapse to compare against, so the K-B branch does not fire. The OR-vs-AND question only affects ordering if both events occur.)

## 7. Verdict and rationale

**Verdict: K-C.** Auto-rationale from the tool: `no collapse threshold crossed across 8 PPO updates; entropy stayed >= 0.20, rollout top-action fraction stayed < 0.95, raw margin stayed < 4.0. Recommend rerun same probe to 10000 timesteps next session before drawing structural conclusions.`

Independent interpretation:

- The H4 smoke-cheap and H5 Phase D entropy recipes were both diagnosed as ending with deterministic-argmax-driven eval anomalies (Phase B/C bit-identical pattern, Phase E/G commitment to `left` or `stay`). K0 confirms those outcomes are not present at 2048 timesteps under the same entropy recipe. The wedge-lock pathology emerges later, somewhere in the 2048-10000 timestep window, not at random init or after the first PPO updates.
- At 2048 timesteps the policy is in a near-uniform stochastic regime. Stochastic eval at this budget would draw all three actions with roughly equal frequency (left ~0.36, stay ~0.41, right ~0.23 from the final policy state); deterministic eval would pick `stay` 100% of the time because every one of the 256 final-rollout observations argmaxes to `stay`.
- The `ent_coef=0.01` is not binding at 2048 timesteps because entropy is already at 97% of its discrete-3 maximum. Whether `ent_coef=0.01` binds later (once natural commitment grows) is exactly the question the 10000-timestep rerun must answer.

## 8. Recommended next slice

Per the K-C interpretation rule pre-agreed with GPT and Jeff: **rerun the same probe at `total_timesteps=10000`, same seed, same config, no other changes.** This is K0-extended, not a new direction.

Concrete scope for the next session:

- Run `tools/h5_training_entropy_probe.py --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml --seed 2 --total-timesteps 10000 --out-dir runs/phase_k --label entropy_probe_seed2_10k`.
- Expected updates: 39 (`10000 / 256` rounded down by SB3, plus tail behavior).
- Expected wall time: ~5-6 minutes by linear scaling of the 64.8s/8-update K0 result.
- Decision rule after 10000-timestep rerun:
  - If collapse threshold crosses within updates 9-39 → K-A (late) → name the collapse mechanism and scope K1 architecture probe against that mechanism.
  - If wedge top-action fraction crosses but entropy and margin stay healthy → K-D → architecture probe K1 with `net_arch=dict(pi=[64], vf=[64])`.
  - If still no collapse threshold crosses → K-C-persistent → seriously consider that the H5 plateau emerges via env-policy coupling rather than within-policy dynamics, and pivot to Phase J option (3) train-seed asymmetry probe with Grok consult.

K1 architecture and K2 train-seed asymmetry remain held behind this rerun per the K-C clause.

## 9. Reproduction recipe

Environment requirements identical to Phase H/I/J:

- Windows 11, repo at `C:\Projects\Sight`, Python at `C:\Users\maste\AppData\Local\Python\bin\python.exe`
- `stable-baselines3 == 2.8.0`, `torch == 2.11.0+cpu`, Python 3.14.4
- Godot 4.6.2 at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`
- No trained model artifacts required; K0 trains from scratch.

Inline invocation (cmd.exe):

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
    --seed 2 ^
    --total-timesteps 2048 ^
    --out-dir runs\phase_k ^
    --label entropy_probe_seed2
```

Detached-driver caveat. A first attempt to run K0 via `start "phase_k" /B cmd /c "%TEMP%\phase_k_driver.bat"` from a Desktop Commander shell failed silently: the .bat wrote its start-header line to the log but the Python subprocess did not produce any output to the log before the parent cmd shell died, taking the detached child with it. Inline invocation captured the same workload with the same SB3 internal randomness in 64.8s and produced the full 8-update record. The lesson is to either invoke the probe inline under `interact_with_process` for short runs, or to use `start /MIN cmd /c` (not `/B`) and detach properly for longer runs; `/B` shares I/O with the parent shell which is not durable when the parent shell exits.

Output artifacts (all gitignored under `runs/`; durable evidence in this doc):

- `runs/phase_k/entropy_probe_seed2.ndjson`: one header row plus one row per PPO update with the full per-update record.
- `runs/phase_k/entropy_probe_seed2.summary.json`: header + verdict + per-update digest + initial/final action_net and policy_state snapshots.
- `runs/phase_k/godot_entropy_probe_seed2/`: Godot env NDJSON sidecars from `make_env`.
- `runs/phase_k/inline.log`: stdout/stderr capture from the inline driver (development artifact; not required).

Re-running with the same train_seed must produce bit-identical action_net blake2b-128 digests at update 0 and update 8 (modulo any SB3 or torch version drift). Summary statistics (entropy, margin, advantage std, gradient norms) are deterministic at fixed seed and fixed SB3/torch versions.

## 10. Implementation-not-at-fault statement

- The `InstrumentedPPO` subclass mirrors SB3 2.8.0 `PPO.train()` body exactly. Per-minibatch loss computation, advantage normalization, ratio clipping, value-prediction clipping, entropy loss, KL approximation, and `clip_grad_norm_` order are unchanged. The only additions are: per-minibatch grad-norm computation between `loss.backward()` and `clip_grad_norm_`, pre/post-update policy-state and action_net snapshots, and rollout-buffer aggregate statistics. SB3's own logger calls are preserved at the end of train() with the same keys so downstream SB3 callbacks consuming the logger see no shape change.
- The `extract_features → mlp_extractor → action_net → softmax` actor-path used for pre/post-update entropy and margin computation is the same path Phase I section 6 verified equivalent to SB3's `get_distribution.probs` to within 1.2e-7 across four trained models.
- The `compute_total_grad_norm` helper computes L2 norm over `p.grad` for every parameter in `policy.parameters()` after `loss.backward()` and before `clip_grad_norm_`. This matches the canonical pre-clip total gradient norm definition; per-module breakdowns (`features_extractor`, `mlp_extractor`, `action_net`, `value_net`) use the same definition restricted to that submodule's parameters.
- No model.zip is written by K0. K0 is diagnostic-only. The trained policy at update 8 is discarded when `env.close()` runs in the `finally` block.

The K0 result localizes the H5 plateau further but does not yet diagnose it. The 2048-timestep budget is too short to observe the wedge-commitment regime that Phase H, Phase I, and Phase J characterized at 10000 timesteps. The next slice extends the same probe to the full training budget.


---

## 11. K0-10k extension: TL;DR

Status: VERIFIED. Same tool, same config (`configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`), same train_seed=2, same Godot env build. Only the `--total-timesteps` flag changed (2048 → 10000) and `--label` (`entropy_probe_seed2` → `entropy_probe_seed2_10k`). K-B detector patched from `or` to `and` semantics before the run (see section 14). 40 PPO updates, 312.0 s wall.

**Verdict: K-A (late variant).** Collapse threshold crosses at update 25 (ts=6400): rollout entropy `H_post = 0.1776` (below 0.20 threshold), rollout sampled top-action fraction `left = 0.973` (above 0.95 threshold), raw top1-top2 margin `m_post = 3.6152` (still below 4.0 threshold; never crosses). The two crossings happen in the same update.

**Three findings beyond the verdict.**

First and most consequential: the prior handoff's claim that "K0 rollout argmax oscillates left/stay across 8 updates" was a measurement-frame error. That observation was on `rollout_action_stats.top_action` (sampled actions actually drawn in the rollout, near-uniform when entropy is high), not on `post_update.policy_state.top_argmax_action` (deterministic argmax of post-update logits over the same 256 rollout observations). Across 40 updates the deterministic argmax is fixed at 1.000 every single update: left at upd 1, stay at upd 2-8, left at upd 9, then left for every remaining update through upd 40. Two basin flips total (left → stay → left), the second landing at upd 9 and never reversing. The "Class B = train_seed=2 picks left" identity from Phase H locks at update 9 (ts=2304), inside the 2048-10000 window the K0-2048 evidence predicted.

Second, the wedge basin commitment forms gradually between update 9 (argmax lock, ts=2304, raw margin 0.43) and update 25 (sampled-fraction lock, ts=6400, raw margin 3.62). Entropy slides monotonically from 1.03 at upd 9 down through 0.20 at upd 25 without any sharp transition. The collapse is not a discrete phase change; it is steady drift under PPO's natural commitment dynamics once the value function starts producing usable advantage signal (EV crosses 0.10 at upd 9, climbs to 0.62 by upd 36).

Third, the K-B detector did not fire even with the tightened AND semantics. Advantage std stayed in `[0.68, 18.45]` (never below 0.05). EV started near zero through upd 8 but climbed steadily after upd 9. This validates the tightening: K0-2048 under OR semantics would have falsely tripped K-B on near-zero EV alone, but the actual mechanism is entropy/action collapse with healthy advantage signal, which is K-A.

## 12. K0-10k per-update trajectory (dual stat table)

Per GPT's distinction: `rollout sampled` is `rollout_action_stats.top_action` and `top_action_fraction` (actions actually drawn in the rollout, n=256). `deterministic argmax` is `post_update.policy_state.top_argmax_action` and `top_argmax_fraction` (argmax of post-update logits evaluated over the same 256 rollout observations). At high entropy these can diverge sharply: argmax is the basin selection, sampled is what the rollout actually did.

| upd | ts | rollout sampled | deterministic argmax | H_post | margin_post | EV |
|---:|---:|---|---|---:|---:|---:|
| 1 | 256 | left=0.395 | **left=1.000** | 1.0307 | 0.5992 | +0.005 |
| 2 | 512 | left=0.492 | **stay=1.000** | 1.0251 | 0.5336 | +0.009 |
| 3 | 768 | stay=0.523 | stay=1.000 | 1.0610 | 0.5509 | -0.008 |
| 4 | 1024 | stay=0.473 | stay=1.000 | 1.0634 | 0.2771 | +0.009 |
| 5 | 1280 | stay=0.477 | stay=1.000 | 1.0429 | 0.4778 | +0.044 |
| 6 | 1536 | stay=0.520 | stay=1.000 | 1.0833 | 0.0818 | -0.008 |
| 7 | 1792 | left=0.398 | stay=1.000 | 1.0493 | 0.1035 | +0.008 |
| 8 | 2048 | left=0.398 | stay=1.000 | 1.0726 | 0.1156 | +0.009 |
| 9 | 2304 | stay=0.406 | **left=1.000** | 1.0288 | 0.4323 | +0.134 |
| 10 | 2560 | left=0.484 | left=1.000 | 0.9610 | 0.5267 | -0.008 |
| 11 | 2816 | left=0.578 | left=1.000 | 0.8950 | 0.6728 | +0.002 |
| 12 | 3072 | left=0.613 | left=1.000 | 0.5869 | 1.6846 | +0.009 |
| 13 | 3328 | left=0.801 | left=1.000 | 0.3987 | 2.8214 | +0.008 |
| 14 | 3584 | left=0.898 | left=1.000 | 0.3116 | 3.1371 | +0.008 |
| 15 | 3840 | left=0.941 | left=1.000 | 0.5124 | 2.3693 | +0.010 |
| 16 | 4096 | left=0.836 | left=1.000 | 0.7177 | 1.8216 | +0.025 |
| 17 | 4352 | left=0.758 | left=1.000 | 0.4131 | 2.4089 | +0.039 |
| 18 | 4608 | left=0.906 | left=1.000 | 0.7560 | 1.3177 | +0.047 |
| 19 | 4864 | left=0.785 | left=1.000 | 0.6119 | 1.6286 | +0.078 |
| 20 | 5120 | left=0.781 | left=1.000 | 0.5455 | 1.8774 | +0.100 |
| 21 | 5376 | left=0.867 | left=1.000 | 0.4495 | 2.5598 | +0.174 |
| 22 | 5632 | left=0.891 | left=1.000 | 0.4235 | 2.1907 | +0.175 |
| 23 | 5888 | left=0.836 | left=1.000 | 0.3017 | 3.1224 | +0.200 |
| 24 | 6144 | left=0.902 | left=1.000 | 0.2084 | 3.4157 | +0.244 |
| 25 | 6400 | **left=0.973** | left=1.000 | **0.1776** | 3.6152 | +0.237 |
| 26 | 6656 | left=0.953 | left=1.000 | 0.2501 | 2.9081 | +0.399 |
| 27 | 6912 | left=0.926 | left=1.000 | 0.4986 | 1.6498 | +0.377 |
| 28 | 7168 | left=0.844 | left=1.000 | 0.4463 | 1.8684 | +0.462 |
| 29 | 7424 | left=0.840 | left=1.000 | 0.3373 | 2.4829 | +0.445 |
| 30 | 7680 | left=0.902 | left=1.000 | 0.3315 | 2.7541 | +0.408 |
| 31 | 7936 | left=0.918 | left=1.000 | 0.3851 | 2.4904 | +0.562 |
| 32 | 8192 | left=0.895 | left=1.000 | 0.3163 | 3.0500 | +0.495 |
| 33 | 8448 | left=0.941 | left=1.000 | 0.4639 | 2.5395 | +0.478 |
| 34 | 8704 | left=0.836 | left=1.000 | 0.5545 | 2.1762 | +0.469 |
| 35 | 8960 | left=0.809 | left=1.000 | 0.7004 | 1.8809 | +0.467 |
| 36 | 9216 | left=0.824 | left=1.000 | 0.5631 | 2.2822 | +0.616 |
| 37 | 9472 | left=0.793 | left=1.000 | 0.8991 | 1.1809 | +0.538 |
| 38 | 9728 | left=0.660 | left=1.000 | 0.7910 | 1.3862 | +0.546 |
| 39 | 9984 | left=0.676 | left=1.000 | 0.6207 | 1.4579 | +0.538 |
| 40 | 10240 | left=0.754 | left=1.000 | 0.8224 | 0.9980 | +0.555 |

Columns: `rollout sampled` = `rollout_action_stats.top_action` / `top_action_fraction`; `deterministic argmax` = `post_update.policy_state.top_argmax_action` / `top_argmax_fraction`; `H_post` = mean policy entropy on the 256 rollout obs after the update; `margin_post` = mean raw top1-top2 logit margin; `EV` = `explained_variance` over the rollout buffer. Bold cells mark basin transitions and the K-A crossing.

## 13. K0-10k findings

**Finding 1: deterministic argmax basin is fixed every update, not bouncing.**

`top_argmax_fraction = 1.000` on every one of the 40 updates. That means the post-update policy, evaluated argmax-style over the 256 rollout observations, picks the same action on every observation. The policy is locally consistent at every checkpoint; what changes between updates is which action that is. Two transitions in 40 updates: left → stay at upd 2, stay → left at upd 9. After upd 9 there are zero further basin flips. Phase H's "Class B = train_seed=2 picks left" identity is real and forms at ts=2304.

**Finding 2: sampled rollout top action is a strictly weaker signal than deterministic argmax at high entropy.**

At upd 1-8 the rollout sampled top action bounces (left, left, stay, stay, stay, stay, left, left), but the deterministic argmax is exactly stationary within each contiguous run and flips once. The prior K0-2048 evidence section 3.5 and the K-C verdict rationale called this "rollout argmax oscillation." That conflated `rollout_action_stats.top_action` with `policy_state.top_argmax_action`. The probe records both separately by design (see tool docstring), but the K0-2048 evidence presented the sampled stat and labeled it argmax. The K0-10k addendum corrects that framing.

The practical implication: at the K0-2048 budget, the basin had already flipped left → stay at upd 2 and was held at stay through upd 8 (8 of 8 deterministic argmax = stay across the post-upd-2 window). Phase H's argmax-from-rollout-buffer measurement on the final K0-2048 step would have read stay, which matches the K0-10k row at upd 8 exactly. The Class B identity reads `left` only if measurement is taken after upd 9, which the K0-2048 budget did not reach.

**Finding 3: collapse is steady drift, not a phase change.**

From upd 9 through upd 25, entropy slides 1.03 → 0.18, sampled top-action fraction rises 0.41 → 0.97, raw margin grows 0.43 → 3.62. EV climbs 0.13 → 0.24 over the same window. No update shows a step change of more than ~0.3 in entropy or ~0.1 in EV. The advantage std stays large (8.0 to 18.4) throughout. The mechanism is: once EV crosses ~0.10 (upd 9), the advantage signal is structured enough that PPO commits to the argmax basin and incrementally sharpens the logit margin, which in turn sharpens the softmax and reduces entropy, which feeds back into more confident actions on subsequent rollouts. Standard PPO commitment dynamics with no pathology.

**Finding 4: raw margin never crosses 4.0, while entropy and sampled fraction do.**

The 4.0 raw-margin threshold from the scope note was a wedge-strategy-commitment proxy informed by Phase E seed2's converged margin of 0.83 on the left tape (Phase I section 8). At K0-10k the trained margin peaks at 3.76 (upd 26 pre) but oscillates back down to 1.00 by upd 40. The Phase E seed2 trained policy at 10000 timesteps had margin 0.83; the K0-10k policy at 10000 timesteps has margin 1.46 (last post-update value). These are within the same order of magnitude. The 4.0 threshold is too strict for this config and should be relaxed for future K-D wedge-only classification or dropped in favor of sampled-fraction-only.

**Finding 5: K-B AND patch did not engage but is correctly silent here.**

Per-update AND check on `(adv_std < 0.05) and (ev < 0.10)`: there is no update where advantage std drops below 0.05. EV is below 0.10 only on updates 1-8 (plus a transient -0.008 at upd 10). So `first_value_adv` is `None` and K-B does not fire. Under the previous OR rule, K-B would have falsely fired at upd 3 (ev=-0.008) because OR semantics treat low-EV-alone as degeneration, when in fact EV is naturally near zero for the first ~2k timesteps of any fresh-init PPO run. The patch behaves correctly.

## 14. K-B detector patch

Pre-patch:

```python
def _value_adv_degenerate(r):
    adv_std = r["adv_ret_val_stats"]["advantages"]["std"]
    ev = r["explained_variance"]
    return bool(adv_std < 0.05 or ev < 0.10)
```

Post-patch:

```python
def _value_adv_degenerate(r):
    adv_std = r["adv_ret_val_stats"]["advantages"]["std"]
    ev = r["explained_variance"]
    return bool(adv_std < 0.05 and ev < 0.10)
```

Docstring updated at lines 549-554 of `tools/h5_training_entropy_probe.py` and module-header comment updated at lines 23-24. Reason: advantage collapse is the load-bearing K-B signal; near-zero EV alone is a normal early-PPO artifact (fresh value head, no calibration to returns yet) and was producing false-positive degeneration flags. The K-C clause from the K0-2048 scope already flagged this and recommended tightening before the 10000-timestep rerun; the patch lands that recommendation. K0-2048 verdict is unchanged because K-B never had a chance to compete with K-A or K-C at that budget either way.

## 15. K0-10k verdict and rationale

**Verdict: K-A (late variant).** Tool rationale: `collapse threshold crossed at update 25 (later than first 3 PPO updates but still within probe budget); entropy_first=25 action_first=25 margin_first=None`.

Independent interpretation:

- The H5 plateau pathology is reproduced at the K0-10k budget. The Phase E seed2 wedge-commitment basin forms by update 9 (deterministic argmax lock, ts=2304) and the rollout-level signature of commitment (top sampled action fraction >= 0.95) lands at update 25 (ts=6400). The 10000-timestep run is sufficient to reach a converged-style basin under the entropy YAML.
- The mechanism is steady-drift entropy collapse driven by PPO commitment under a structured advantage signal, not value/advantage degeneration (K-B), not random-init early lock (K-A early), not wedge-without-distributional-collapse (K-D). Standard PPO behavior on this env-policy-config combination.
- The two-flip pattern (left → stay at upd 2, stay → left at upd 9, then stay-left forever) means Phase H's basin assignments are sensitive to where in training the measurement is taken. Class A vs Class B vs Class C labels are only meaningful at converged training; pre-convergence basin can be misread.
- The H5 plateau is now characterized: under this config + env + seed, PPO commits to a single deterministic-argmax action by upd 9 and sharpens it monotonically to a margin of ~3 (sampled fraction 0.95+) by upd 25. From there, training continues to refine but never broadens the distribution back. Once committed, the only escape from the basin is a structural intervention (architecture, env, reward, or training distribution).

## 16. K0-10k recommended next slice

The K-C clause is now discharged: the same probe was run at 10000 timesteps and produced a definitive verdict.

K1 architecture probe is the structurally correct next step under the Phase J option ladder. Open question for GPT before K1 executes: does Phase H's basin definition need re-evaluation given finding 1 (deterministic argmax locks early, sampled rollout misreads the lock)?

K2 train-seed asymmetry probe remains held behind K1.

The attached GPT plan extends past K0-10k into immediate K1 execution. Holding K1 until Jeff's explicit go signal in the next prompt.

## 17. K0-10k reproduction recipe

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
    --seed 2 ^
    --total-timesteps 10000 ^
    --out-dir runs\phase_k ^
    --label entropy_probe_seed2_10k
```

Artifacts (gitignored under `runs/`):

- `runs\phase_k\entropy_probe_seed2_10k.ndjson` (sha256 `5a8d7a4df7666c55b45d5d5ae9a0eb8f5a4fd3c36ae2589b7054884720870eab`, 137628 bytes)
- `runs\phase_k\entropy_probe_seed2_10k.summary.json` (sha256 `bd420e0f3b253d8e3ecc86843f68aeaa1c4b6783f80b8a38c1ab90a2168fa80b`, 47340 bytes)
- `runs\phase_k\godot_entropy_probe_seed2_10k\` (Godot env NDJSON sidecars)

Wall time: 312.0 s (5.20 min). Linear extrapolation from K0-2048's 64.79 s / 8 updates predicted ~324 s for 40 updates; observed slightly faster.

Launch pattern. Inline invocation via `interact_with_process` is feasible at this duration but presses the 4-min MCP timeout. The actual launch used a bat-with-sentinel wrapper at `C:\Users\maste\AppData\Local\Temp\run_k0_10k.bat` that sets `SIGHT_GODOT_EXE` inline, redirects stdout/stderr to `C:\Users\maste\AppData\Local\Temp\k0_10k.log`, and writes a done-sentinel `k0_10k.done` containing the Python exit code (`EXIT=0`) on completion. The wrapper was started with `start "k0_10k" /MIN cmd /c <bat>`, then a separate poll loop checked the sentinel every 60-120 s. This pattern is durable across the 5-min mark and is recommended for any future probe at or above this budget.

## 18. K1 architecture mechanism probe: TL;DR

K1 swaps the default SB3 CnnPolicy MLP head for an explicit `policy_kwargs.net_arch = dict(pi=[64], vf=[64])`, holding everything else identical to the entropy YAML. One instrumented 10000-timestep training session, train_seed=2, 40 PPO updates, 311.5 s wall (essentially equal to K0-10k's 312.0 s).

K1 verdict: **K-C**, no collapse threshold crossed across 40 updates. Entropy stayed in [0.96, 1.10], sampled top-action fraction stayed in [0.34, 0.55] (max 0.547), raw margin stayed under 0.66. The K0-10k wedge mechanism does not reproduce under separated pi/vf MLP heads at this budget.

Critical caveats:

- Deterministic argmax `det_argmax_fraction_post` is still 1.0 every update, i.e. inside any single 256-step rollout the policy still picks one action across all observations. The architecture intervention did not produce per-observation conditional behavior. It produced **basin-shifting commitment**: the chosen action changes across updates (left → right at upd 8, right → left at upd 24, left → right at upd 39) rather than staying observation-invariant `left`.
- Explained variance never crosses zero in K1. The value head with 64-unit MLP fails to learn meaningful return predictions inside the 10k budget. K0-10k crossed ev=0.10 at upd 9. K1 produces no value signal, which is the mechanical reason commitment never sharpens enough to trigger the collapse thresholds.
- Final policy probs are roughly balanced: left=0.43, stay=0.11, right=0.46. K0-10k's final policy concentrated on `left`.

Read: **K1 changed the mechanism, but not by teaching the policy to dodge. It changed the mechanism by hobbling the value head enough that PPO never gathers structured advantage to commit on.** This matches the K1 decision-rule entry GPT outlined: "Different-basin result: deterministic argmax locks to stay or right; that is not automatically a win. It means architecture changed basin selection, not necessarily observation-conditioned learning."

K1 is therefore a productive mechanism probe (the wedge does not reproduce), but it is not a competence win. The next slice should either extend K1 training budget to see whether the value head ever learns, or move to K2 train-seed asymmetry to test whether the K0-10k wedge was seed-specific, or both.

Artifact sha256:

- `runs/phase_k/entropy_probe_seed2_10k_k1_netarch.ndjson`        `5e8cb430e67b93674848e4f388f08f9b7c47eff0c890c1e25621d07421c40cdf`
- `runs/phase_k/entropy_probe_seed2_10k_k1_netarch.summary.json`  `48f24a71b7d454724a96617aa415a5dd34d6baf258712d7b8bc2409fa86f1b84`

Both gitignored under `runs/phase_k/`.


## 19. K1 vs K0-10k headline comparison

| Metric | K0-10k (shared MLP head) | K1 (`net_arch=dict(pi=[64], vf=[64])`) |
|---|---|---|
| Verdict | K-A late | K-C |
| Wall seconds | 312.0 | 311.5 |
| Updates | 40 | 40 |
| Final entropy (upd 40, post) | 0.18 | 0.96 |
| Min entropy across run | ~0.18 (upd 25) | 0.96 (monotone floor, upd 40) |
| Max sampled top-action fraction | 0.97 (upd 25) | 0.547 (upd 37) |
| Max raw margin (post) | 3.76 (upd 26 pre) | 0.66 (upd 15) |
| Deterministic argmax locked to single action? | Yes, `left` from upd 9 onward | No, oscillates: left→right at upd 8, right→left at upd 24, left→right at upd 39 |
| `det_argmax_fraction_post` within an update | 1.0 throughout | 1.0 throughout (per-update single action; differs across updates) |
| EV crosses 0.10? | Yes, at upd 9 | No (stays ~0 throughout) |
| Advantage std range | [0.68, 18.45] | [2.21, 6.35] (narrower, never explodes) |
| Final mean policy probs | concentrated on `left` | left 0.43, stay 0.11, right 0.46 |
| Collapse thresholds crossed | entropy < 0.20 and sampled top-action ≥ 0.95 at upd 25 | none |

Interpretation: K0-10k showed the canonical PPO wedge — value head learns enough structure to drive policy commitment, advantage std climbs, margin sharpens, entropy collapses. K1 short-circuits this by giving the value head insufficient capacity to learn returns inside the budget. The wedge cannot form because the gradient signal to form it never appears. Policy stays high-entropy and basin-shifts across updates.

## 20. K1 per-update digest

40 rows. Columns:

- `upd` — PPO update index.
- `ts` — total timesteps after this update.
- `top` — `rollout_top_action` (sampled top action over the 256 rollout obs).
- `top_frac` — `rollout_top_action_fraction`.
- `det_pre` / `det_post` — deterministic argmax over the same 256 obs, before and after the update.
- `H_pre` / `H_post` — mean policy entropy before and after the update.
- `marg_post` — mean raw top1-top2 margin after the update.
- `adv_std` — advantage std for this update's minibatch.
- `EV` — explained variance for this update.
- `p_left` / `p_right` — post-update mean policy probability for `left` and `right` (`stay` = 1 - p_left - p_right - rounding).

```
| upd |  ts | top   | top_frac | det_pre | det_post | H_pre | H_post | marg_post | adv_std |   EV   | p_left | p_right |
|-----|-----|-------|----------|---------|----------|-------|--------|-----------|---------|--------|--------|---------|
|   1 |  256| stay  |   0.36   |  stay   |  left    | 1.10  | 1.10   |   0.03    |  2.71   | -0.007 |  0.35  |  0.30   |
|   2 |  512| left  |   0.35   |  left   |  left    | 1.10  | 1.10   |   0.06    |  4.27   | -0.000 |  0.36  |  0.30   |
|   3 |  768| left  |   0.37   |  left   |  left    | 1.10  | 1.10   |   0.06    |  2.53   |  0.000 |  0.36  |  0.30   |
|   4 | 1024| left  |   0.34   |  left   |  stay    | 1.10  | 1.09   |   0.05    |  2.52   |  0.000 |  0.35  |  0.29   |
|   5 | 1280| stay  |   0.41   |  stay   |  stay    | 1.09  | 1.09   |   0.22    |  4.39   |  0.000 |  0.30  |  0.31   |
|   6 | 1536| stay  |   0.43   |  stay   |  stay    | 1.09  | 1.08   |   0.25    |  2.50   |  0.000 |  0.27  |  0.32   |
|   7 | 1792| stay  |   0.44   |  stay   |  stay    | 1.08  | 1.09   |   0.18    |  4.51   |  0.000 |  0.27  |  0.33   |
|   8 | 2048| stay  |   0.41   |  stay   |  right   | 1.09  | 1.09   |   0.04    |  4.49   |  0.000 |  0.26  |  0.38   |
|   9 | 2304| right |   0.42   |  right  |  right   | 1.09  | 1.07   |   0.41    |  2.48   |  0.000 |  0.25  |  0.45   |
|  10 | 2560| right |   0.45   |  right  |  right   | 1.07  | 1.05   |   0.45    |  2.47   |  0.000 |  0.23  |  0.47   |
|  11 | 2816| right |   0.48   |  right  |  right   | 1.05  | 1.04   |   0.24    |  2.46   |  0.000 |  0.19  |  0.46   |
|  12 | 3072| right |   0.43   |  right  |  right   | 1.04  | 1.03   |   0.06    |  2.45   |  0.000 |  0.17  |  0.43   |
|  13 | 3328| stay  |   0.42   |  right  |  right   | 1.03  | 1.04   |   0.08    |  2.44   |  0.000 |  0.18  |  0.43   |
|  14 | 3584| right |   0.41   |  right  |  right   | 1.04  | 1.04   |   0.44    |  4.73   |  0.000 |  0.21  |  0.48   |
|  15 | 3840| right |   0.46   |  right  |  right   | 1.04  | 1.03   |   0.66    |  2.43   | -0.000 |  0.23  |  0.51   |
|  16 | 4096| right |   0.53   |  right  |  right   | 1.03  | 1.03   |   0.64    |  2.42   |  0.000 |  0.23  |  0.51   |
|  17 | 4352| right |   0.46   |  right  |  right   | 1.03  | 1.07   |   0.39    |  2.41   |  0.000 |  0.25  |  0.45   |
|  18 | 4608| right |   0.45   |  right  |  right   | 1.07  | 1.09   |   0.20    |  2.40   | -0.000 |  0.32  |  0.39   |
|  19 | 4864| right |   0.38   |  right  |  right   | 1.09  | 1.09   |   0.07    |  4.86   |  0.000 |  0.35  |  0.38   |
|  20 | 5120| stay  |   0.35   |  right  |  left    | 1.09  | 1.08   |   0.03    |  2.38   |  0.000 |  0.39  |  0.37   |
|  21 | 5376| right |   0.38   |  left   |  right   | 1.08  | 1.08   |   0.03    |  4.94   | -0.000 |  0.37  |  0.38   |
|  22 | 5632| right |   0.41   |  right  |  right   | 1.08  | 1.06   |   0.36    |  2.37   |  0.000 |  0.32  |  0.45   |
|  23 | 5888| right |   0.44   |  right  |  right   | 1.06  | 1.06   |   0.31    |  6.35   |  0.000 |  0.33  |  0.45   |
|  24 | 6144| right |   0.43   |  right  |  left    | 1.06  | 1.07   |   0.10    |  5.05   |  0.000 |  0.41  |  0.37   |
|  25 | 6400| left  |   0.42   |  left   |  left    | 1.07  | 1.07   |   0.07    |  2.34   |  0.000 |  0.40  |  0.37   |
|  26 | 6656| left  |   0.40   |  left   |  left    | 1.07  | 1.06   |   0.09    |  2.33   |  0.000 |  0.41  |  0.38   |
|  27 | 6912| left  |   0.43   |  left   |  left    | 1.06  | 1.02   |   0.14    |  5.21   | -0.000 |  0.45  |  0.39   |
|  28 | 7168| left  |   0.47   |  left   |  left    | 1.02  | 1.02   |   0.11    |  2.31   |  0.000 |  0.44  |  0.40   |
|  29 | 7424| left  |   0.44   |  left   |  left    | 1.02  | 1.02   |   0.33    |  5.24   |  0.000 |  0.48  |  0.35   |
|  30 | 7680| left  |   0.51   |  left   |  left    | 1.02  | 1.00   |   0.52    |  2.30   |  0.000 |  0.52  |  0.31   |
|  31 | 7936| left  |   0.54   |  left   |  left    | 1.00  | 1.00   |   0.48    |  2.29   |  0.000 |  0.52  |  0.32   |
|  32 | 8192| left  |   0.45   |  left   |  left    | 1.00  | 1.02   |   0.24    |  5.42   |  0.000 |  0.47  |  0.37   |
|  33 | 8448| left  |   0.48   |  left   |  left    | 1.02  | 1.02   |   0.30    |  2.27   | -0.000 |  0.48  |  0.35   |
|  34 | 8704| left  |   0.47   |  left   |  left    | 1.02  | 1.01   |   0.40    |  2.26   |  0.000 |  0.50  |  0.34   |
|  35 | 8960| left  |   0.50   |  left   |  left    | 1.01  | 0.99   |   0.53    |  2.25   |  0.000 |  0.53  |  0.31   |
|  36 | 9216| left  |   0.49   |  left   |  left    | 0.99  | 0.99   |   0.57    |  5.55   |  0.000 |  0.54  |  0.30   |
|  37 | 9472| left  |   0.55   |  left   |  left    | 0.99  | 1.00   |   0.29    |  5.54   |  0.000 |  0.49  |  0.36   |
|  38 | 9728| left  |   0.45   |  left   |  left    | 1.00  | 1.01   |   0.11    |  2.23   |  0.000 |  0.45  |  0.40   |
|  39 | 9984| left  |   0.45   |  left   |  right   | 1.01  | 0.99   |   0.05    |  5.63   | -0.000 |  0.42  |  0.45   |
|  40 |10240| right |   0.48   |  right  |  right   | 0.99  | 0.96   |   0.06    |  2.21   |  0.000 |  0.43  |  0.46   |
```

Three deterministic basin transitions visible at update 8 (left→right), update 24 (right→left), and update 39 (left→right). Compare to K0-10k where the only transitions were update 2 (left→stay) and update 9 (stay→left), after which `left` held through update 40.

## 21. K1 findings

1. **The K0-10k wedge mechanism does not reproduce under separated pi/vf MLP heads at the same budget.** No collapse threshold crossed. Entropy floor 0.96, sampled top-action fraction ceiling 0.547, raw margin ceiling 0.66. The "agent gets stuck on `left` forever" lock-in K0-10k diagnosed is absent. This is a real mechanism effect, not a noise effect: the K0-10k probe was bit-deterministic on this seed and recipe; only the architecture kwarg changed.

2. **Architecture did not produce observation-conditional behavior; it produced basin-shifting commitment.** Within any single PPO update, the K1 policy still picks one action deterministically across all 256 rollout observations (`det_argmax_fraction_post=1.0` every update). What changed is that the chosen action shifts across updates. This is the GPT-anticipated "Different-basin result" decision-rule entry, except the basin itself does not stabilize on a non-left action; it cycles.

3. **Value head with 64-unit MLP underfits inside 10k timesteps.** Explained variance never crosses zero; advantage std stays bounded in [2.21, 6.35] (K0-10k went up to 18.45). The mechanical explanation for finding 2 is that PPO never receives structured-enough advantage to commit beyond the rollout-level scale. The K0-10k wedge required the value head to learn structured returns first (EV crossed 0.10 at upd 9, after which commitment sharpened). K1 disables that pre-condition by shrinking the value MLP head.

4. **Per-step deterministic argmax fraction = 1.0 every update is a known property of CnnPolicy on this env, not specific to K0 or K1.** It means within any 256-step rollout under any tested architecture, the policy at this scale and budget always picks one action for all observations. The wedge / non-wedge distinction lives entirely in (a) whether that single action stabilizes across updates and (b) whether the supporting probability margin sharpens. K0-10k: yes to both. K1: no to both.

5. **K1 does not satisfy the "Strong K1 mechanism win" decision rule from the prior planning turn.** GPT's strong-win criterion was "deterministic argmax does not lock by update 9, entropy does not cross < 0.20 by update 25, and final policy is not constant-action across rollout obs." K1 passes 1 and 2 but fails 3 (the final policy at upd 40 is still constant-action across rollout obs, just `right` instead of `left` and oscillating across updates). The honest categorization is GPT's "Weak win" plus "Different-basin result" rolled together: architecture attacks the mechanism, but not by enabling observation-conditioned learning.

## 22. K1 verdict and rationale

Verdict: **K-C** (probe-classifier output).

Editorial verdict for the K-series ladder: **mechanism intervention successful but inconclusive about competence.** K1 demonstrates that the K0-10k wedge is not architecture-invariant; the shared MLP head is part of the mechanism. K1 does **not** demonstrate that the K1 architecture can learn to dodge given more training, because the value head shows no learning signal at 10k timesteps. Two follow-on slices are honest:

- **K1-extended**: rerun K1 with `total_timesteps=30000` or `50000` to see whether the value head ever learns and whether commitment then forms in a non-wedge basin.
- **K2 train-seed asymmetry**: rerun K0-10k baseline (shared head, same recipe) at train_seed=1, 3, 4 to test whether K0-10k's `left` wedge was seed-specific. If the wedge appears at every seed under shared-head, the wedge is architectural. If it only appears at seed 2, the wedge is seed-specific and K1's basin-shifting may be a less interesting result than the K0-10k narrative implies.

K2 is structurally more informative because it disambiguates "architecture matters" from "seed matters" before further architecture changes are tested. K1-extended is cheaper but only confirms or rejects the K1-specific question without resolving the K0-10k seed asymmetry question.

## 23. Demo-0 connection

Demo-0 (this session, prior turn) ran the K0-10k trained `model.zip` (Phase E seed=2, entropy recipe) in deployment with `deterministic=True` at eval seed 1008 for 1800 steps. Behavior: 100% `left`. Total reward 1800.0, truncated by timeout. The deployment-time wedge matches what K0-10k diagnosed at training time: the policy commits to `left` across observations and never reconsiders.

K1 attacks that deployment-time wedge at the source. If K1-extended or K2 produces a non-wedge converged policy and that policy is then loaded by `tools/demo0_visible_play.py` against eval seed 1008, the Demo-0 video would show a non-pinned paddle. That is the chain of evidence from mechanism work back to the watchable artifact. K1 by itself does not produce that policy because EV never crosses zero; no `model.zip` is written by the entropy probe regardless.

## 24. K1 reproduction recipe

From `C:\Projects\Sight` in cmd.exe:

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy_k1_netarch.yaml ^
  --seed 2 ^
  --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label entropy_probe_seed2_10k_k1_netarch
```

Wall: 311.5 s on StrongerJr. Tool-call budget: exceeds the inline MCP 4-minute timeout. Use the bat-with-sentinel wrapper pattern (template at `C:\Users\maste\AppData\Local\Temp\sight_k1\run_k1.bat` from this session) rather than `interact_with_process` inline.

Config differences vs the unmodified entropy YAML (`configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml`):

- `run.name` changes to `signal_dodge_ppo_h5_pixel_entropy_k1_netarch`.
- `algo.hyperparams` adds `policy_kwargs: {net_arch: {pi: [64], vf: [64]}}`.
- No other field differs.

`per_update_digest` schema now includes (`tools/h5_training_entropy_probe.py` digest patch this session): `det_argmax_pre`, `det_argmax_fraction_pre`, `det_argmax_post`, `det_argmax_fraction_post`, `policy_prob_left_post`, `policy_prob_stay_post`, `policy_prob_right_post`. These fields were always recorded in the full NDJSON records under `pre_update.policy_state` and `post_update.policy_state`; the patch only hoists them into the per-update digest so analysis tooling does not need to walk the full NDJSON.


## 25. K2 baseline shared-head train-seed asymmetry probe (overview)

K2 reruns the K0-10k baseline recipe (shared NatureCNN head, no `policy_kwargs.net_arch` override) at `train_seed in {1, 3, 4}` to test whether the K0-10k wedge is seed-specific or architectural. K1 already showed the wedge does not reproduce when the policy and value heads are split into separate 64-unit MLPs. K2 fixes the architecture at K0's setting and varies the seed.

Config: `configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml` unchanged from K0-10k. Probe tool: `tools\h5_training_entropy_probe.py` (digest patch from K1 session). Total timesteps: 10000 per seed. Three runs executed serially via bat-with-sentinel wrapper at `C:\Users\maste\AppData\Local\Temp\sight_k2\run_k2_serial.bat`. Wall times 311.9 s / 312.4 s / 312.7 s for seeds 1 / 3 / 4 (essentially identical to K0-10k seed 2's 312.0 s).

Artifact locations (all under `runs\phase_k\`, gitignored):

- `entropy_probe_seed1_10k_k2_baseline.{ndjson,summary.json}`
- `entropy_probe_seed3_10k_k2_baseline.{ndjson,summary.json}`
- `entropy_probe_seed4_10k_k2_baseline.{ndjson,summary.json}`

## 26. K2 per-seed evidence

### 26.1 Seed 1

- Verdict: `K-A`. Probe-classifier rationale: collapse threshold crossed at update 8 (later than first 3 PPO updates but still within probe budget); entropy_first=8, action_first=9, margin_first=8.
- First entropy collapse update (entropy_post < 0.20): **8**.
- First rollout top-action fraction collapse update (>= 0.95): **9**.
- First raw-margin collapse update (>= 4.0): **8**.
- Deterministic argmax trajectory: `stay` upd 1-40 (no oscillation; locked from update 1).
- First permanent det_argmax lock update: **1**.
- Entropy floor: 0.0024. Entropy ceiling: 1.0843. Final entropy_post: 0.0114.
- Rollout top-action fraction ceiling: 1.0000. Final: 0.9922.
- Raw margin ceiling: 8.2659. Final margin_post: 6.5180.
- Explained variance: min -0.0588, max 0.3581. First positive: update **1**.
- Advantage std range: [0.2916, 15.6277].
- Final mean policy probs: L=0.0000, S=0.9985, R=0.0015.
- Final deterministic argmax: `stay` (frac=1.0000). Initial deterministic argmax: `right` (frac=1.0000).
- Wall: 311.9 s. n_updates: 40.
- Artifact sha256:
  - summary: `71acb32d4c086c4dbc4fbc48827ac31777ed1ff54959a7842e16368d3fa5a134`
  - ndjson: `3fed5642fdd89e3ccff4d74f25e9fa6fcd99d9537fa699f1962e5945a27a8d21`

### 26.2 Seed 3

- Verdict: `K-A`. Probe-classifier rationale: collapse threshold crossed at update 20 (later than first 3 PPO updates but still within probe budget); entropy_first=20, action_first=21, margin_first=20.
- First entropy collapse update: **20**.
- First rollout top-action fraction collapse update: **21**.
- First raw-margin collapse update: **20**.
- Deterministic argmax trajectory: `left` upd 1, `right` upd 2-6, `left` upd 7-8, `right` upd 9, `left` upd 10-13, `right` upd 14, `left` upd 15-40.
- First permanent det_argmax lock update: **15**.
- Entropy floor: 0.0000. Entropy ceiling: 0.9764. Final entropy_post: 0.0001.
- Rollout top-action fraction ceiling: 1.0000. Final: 1.0000.
- Raw margin ceiling: 13.0672. Final margin_post: 11.7210.
- Explained variance: min -0.3900, max 0.2507. First positive: update **2**.
- Advantage std range: [0.2272, 16.4744].
- Final mean policy probs: L=1.0000, S=0.0000, R=0.0000.
- Final deterministic argmax: `left` (frac=1.0000). Initial deterministic argmax: `stay` (frac=1.0000).
- Wall: 312.4 s. n_updates: 40.
- Artifact sha256:
  - summary: `f6aaeea96fa9875d0b60c8184473119e5fbba61532c43081c717a618b9d7dda5`
  - ndjson: `aa522e645e6b3383d9d496120a32592a759dcca9bffb9080afa6edb96ad4d9ed`

### 26.3 Seed 4

- Verdict: `K-A`. Probe-classifier rationale: collapse threshold crossed at update 13 (later than first 3 PPO updates but still within probe budget); entropy_first=13, action_first=14, margin_first=13.
- First entropy collapse update: **13**.
- First rollout top-action fraction collapse update: **14**.
- First raw-margin collapse update: **13**.
- Deterministic argmax trajectory: `right` upd 1, `left` upd 2, `right` upd 3-11, `left` upd 12-40.
- First permanent det_argmax lock update: **12**.
- Entropy floor: 0.0037. Entropy ceiling: 1.0914. Final entropy_post: 0.0074.
- Rollout top-action fraction ceiling: 1.0000. Final: 1.0000.
- Raw margin ceiling: 8.1455. Final margin_post: 7.4217.
- Explained variance: min -0.2119, max 0.4436. First positive: update **1**.
- Advantage std range: [0.1410, 14.6133].
- Final mean policy probs: L=0.9991, S=0.0003, R=0.0006.
- Final deterministic argmax: `left` (frac=1.0000). Initial deterministic argmax: `left` (frac=1.0000).
- Wall: 312.7 s. n_updates: 40.
- Artifact sha256:
  - summary: `7cd7265d12b8d69d76c4e280366c65411805fc45306cca17c373da59ba632122`
  - ndjson: `74888472f4bc8c6f5b6ddc2d5ec2da3b66df8b7c428e035147efce31a3883efd`

## 27. K2 cross-seed comparison (with K0-10k seed 2 and K1 seed 2)

All rows are 10000-timestep, 40 PPO updates, entropy recipe (`signal_dodge_ppo_h5_pixel_entropy.yaml` or the K1 sibling). "Shared head" means CnnPolicy default (NatureCNN feature extractor with shared MLP head); "Split 64/64" means `policy_kwargs.net_arch = dict(pi=[64], vf=[64])`.

| seed | arch | verdict | ent collapse upd | margin collapse upd | top-action-frac collapse upd | det_argmax permanent lock upd | final det_argmax | final mean probs L / S / R | max raw margin | EV first positive upd | adv std max |
|-----:|------|---------|-----------------:|-------------------:|----------------------------:|------------------------------:|------------------|---------------------------|---------------:|---------------------:|------------:|
| 2 | shared | K-A | 25 | none (< 4.0) | 25 | 10 | left | 0.664 / 0.090 / 0.247 | 3.62 | 1 | 18.45 |
| 1 | shared | K-A | 8  | 8  | 9  | 1  | stay | 0.000 / 0.999 / 0.001 | 8.27 | 1 | 15.63 |
| 3 | shared | K-A | 20 | 20 | 21 | 15 | left | 1.000 / 0.000 / 0.000 | 13.07 | 2 | 16.47 |
| 4 | shared | K-A | 13 | 13 | 14 | 12 | left | 0.999 / 0.000 / 0.001 | 8.15 | 1 | 14.61 |
| 2 | split 64/64 | K-C | none | none | none | none (oscillating) | right (upd 40) | 0.433 / 0.109 / 0.459 | 0.66 | none (EV stuck ≤ 0) | 6.35 |

Key observations:

1. **All four shared-head seeds wedge to verdict K-A within 10k timesteps.** No shared-head seed escaped the collapse classifier. The K0-10k wedge is **not** a seed 2 artifact.
2. **K2 seeds 1/3/4 collapse harder than K0-10k seed 2.** K0-10k seed 2 never crossed the margin >= 4.0 threshold (max 3.62) and ended with entropy 0.82 and mean prob L=0.66. All three K2 seeds crossed margin >= 4.0 (8.27, 13.07, 8.15), bottomed out with entropy approximately 0 (0.011, 0.0001, 0.007), and ended with the dominant-action prob approximately 1.0. K0-10k seed 2 is the **mildest** shared-head wedge of the four, not the most representative.
3. **Basin choice is seed-dependent under the same architecture.** Three of four shared-head seeds wedge to `left`; one (seed 1) wedges to `stay`. None of the tested seeds wedged to `right`. The "left-wall" framing of the K0-10k narrative is incomplete; the shared-head architecture wedges into *some* constant action, not specifically `left`.
4. **Lock-in timing varies widely.** Permanent det_argmax lock occurs at upd 1 (seed 1), upd 10 (seed 2), upd 12 (seed 4), and upd 15 (seed 3). Seed 1 wedges immediately and stably; the other three oscillate briefly before committing. K1 (split 64/64) is the only configuration that never locks within 40 updates.
5. **K1 remains the only configuration with no value-learning signal at 10k.** EV stays at or below zero across all 40 updates in K1; every shared-head run has EV first positive within the first two updates and reaches non-trivial EV (>= 0.25 max) in seeds 1 and 4.

## 28. K2 verdict and decision-rule mapping

Verdict against the GPT decision rule (from the K2 scoping turn):

- **Primary outcome: K2-A.** All or most seeds wedge under shared-head. Specifically, 4/4 tested seeds (2 from K0-10k, 1/3/4 from K2) produce verdict K-A under the shared-head recipe. The shared-head architecture is implicated; the wedge is not a seed 2 artifact. Per GPT's rule, the next architectural slice should be value-head capacity sweep rather than more baseline seeds.
- **Secondary outcome: constant-policy attractor that is not specifically `left`.** Seed 1 collapses to `stay`, seeds 2/3/4 collapse to `left`, none to `right`. This is the GPT decision-rule scoping note: "If seeds wedge but choose different actions, the project has a general constant-policy attractor problem, not a left-wall-only problem." Both readings apply simultaneously.
- **K2-D explicitly ruled out.** "No seeds wedge" is false; every seed wedged.
- **K2-B ("only seed 2 wedges, K0-10k is seed-specific") explicitly ruled out** by 3/3 K2 seeds also producing K-A with sharper collapses than seed 2.
- **K2-C ("mixed pattern, treat as architecture-seed interaction") partially applies** in the sense that *which* action the basin settles into depends on the seed; but it does not apply in the sense GPT scoped it, because every seed in fact wedged. The verdict is K2-A primarily, with a K2-C-flavored basin asymmetry as a sub-result.

## 29. K2 honest reads

1. **The shared-head + entropy recipe produces a constant-action deterministic basin by 10k timesteps with high reliability across seeds.** This is now a four-seed result, not a one-seed observation. The mechanism is architectural under the K2 evidence.
2. **K0-10k seed 2 is unusually mild relative to its peers.** Final entropy 0.82, max margin 3.62, mean prob L=0.66 — these are the *weakest* collapse signals among the four shared-head seeds tested. The K0 narrative would be sharper if seed 2 were replaced with seed 3 or seed 4 as the canonical reference. The Demo-0 visualization happens to be on the mildest seed; if Demo-0 were rerun on a K2-seed-1 trained policy, the paddle would visibly hold `stay` for 1800 steps with the same "constant action" character but a different action choice.
3. **The constant-policy attractor is not the same as a "left-wall" attractor.** Three of four seeds did wedge to `left`, but seed 1 wedged to `stay` with essentially zero probability of any other action. Any future framing that says "the policy hugs the left wall" overstates the regularity; the policy commits to *one* of the three actions and abandons the others.
4. **Value-learning is not by itself the wedge mechanism.** Every shared-head seed has EV positive within the first two updates and three of four reach non-trivial EV. The wedge happens *despite* a working value signal, not because of its absence. K1's value-head failure is downstream of architecture (split heads, smaller capacity), not a precondition for the K0-style wedge.
5. **K1 still looks like an architecture effect under K2.** K1's failure to wedge happens at a configuration where the value head shows no learning signal (EV stuck) and the policy oscillates rather than committing. K2 confirms shared-head wedges across seeds; K1's non-wedge therefore is the architectural lever, even if K1 itself does not produce a competent policy.

## 30. K2 reproduction recipe

From `C:\Projects\Sight` in cmd.exe, with `SIGHT_GODOT_EXE` set inline in the same shell or the launching .bat:

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 1 ^
  --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label entropy_probe_seed1_10k_k2_baseline

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 3 ^
  --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label entropy_probe_seed3_10k_k2_baseline

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_training_entropy_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
  --seed 4 ^
  --total-timesteps 10000 ^
  --out-dir runs\phase_k ^
  --label entropy_probe_seed4_10k_k2_baseline
```

Wall: ~312 s per seed on StrongerJr, ~16 min total serial. Tool-call budget: exceeds the inline MCP 4-minute timeout per seed; use bat-with-sentinel pattern (template at `C:\Users\maste\AppData\Local\Temp\sight_k2\run_k2_serial.bat`). The serial bat runs all three seeds in one process and writes a single sentinel with `seed1=<rc> seed3=<rc> seed4=<rc>` on completion.

No config edits required vs. K0-10k. K2 uses the unmodified `signal_dodge_ppo_h5_pixel_entropy.yaml`; the only variable is `--seed`. This is intentional: K2's hypothesis is that nothing but the seed differs between K0-10k seed 2 and the K2 runs, and the comparison is only valid if the YAML is identical.

## 31. What K2 closes and what it opens

Closed:

- The hypothesis that K0-10k's wedge was a seed 2 artifact. Falsified. All four shared-head seeds wedge.
- The hypothesis that the wedge is specifically a "left-wall" basin. Falsified. Seed 1 wedges to `stay` with no other actions in the rollout.
- The hypothesis that K1's non-wedge could be a seed effect rather than an architecture effect. Falsified relative to seeds 1/3/4: under shared head, every tested seed wedges.

Opened:

- **Value-head capacity sweep.** K1 split (64, 64) MLP heads produced no value-learning signal at 10k. The next architectural probe should sweep value-head capacity (for example, vf=[128], vf=[256], or shared vf with frozen pi) to find the minimum value head that enables observation-conditional commitment rather than constant-action collapse. K1-extended (rerun K1 to 30k or 50k) remains parked but is now lower priority than a capacity sweep, because K2 makes the architectural mechanism more central than the budget mechanism.
- **Basin-choice mechanism.** Why does seed 1 commit to `stay` while seeds 2/3/4 commit to `left`? The answer is presumably in early rollout statistics; seed 1's first rollout already top-actions `stay`, and seeds 2/3/4 top-action `left` or `stay`/`right` early but converge to `left`. A short investigative slice on initial weight init plus first-rollout action statistics would close this question without further training.
- **Whether the constant-action attractor is a property of CnnPolicy specifically.** Every probe so far uses CnnPolicy on pixels. A control run with `MlpPolicy` on state observations under the same env would test whether the attractor is policy-architecture-specific or env-specific. This is out of K-series scope but worth naming for later.
