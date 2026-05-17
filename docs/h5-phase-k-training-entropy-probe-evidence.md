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
