# K5.6 -> Literal RL: PPO Finetune from BC Warm-Start

Verdict: **PASS**. A PPO-finetuned (literal RL) policy clears the above-baseline
bar in-env. Mean episode length 1710.5 vs bar 930.27 (+780.23), vs best constant
845.7 (+864.8), on held-out seeds 1000-1009. This is the first literal-RL Signal
Dodge policy in this project to beat the constant-action baseline. The BC pivot
(K5.6) cleared it by supervised imitation; this closes it as reinforcement
learning.

## What was done

Warm-started an SB3 PPO `MlpPolicy` actor from `bc_policy.pt` (state mode, 10-dim
obs, 3 actions, NOT pixel/CNN) and PPO-finetuned for 20480 timesteps against the
production `GodotSignalDodgeEnv` (state, headless, reward_shaping "none",
max_steps 1800). Tool: `tools/k5_6_ppo_finetune.py`.

Weight mapping (shape-verified against a live SB3 2.8.0 policy before the run):
- BC `net.0` (Linear 10->64) -> `mlp_extractor.policy_net.0`
- BC `net.2` (Linear 64->64) -> `mlp_extractor.policy_net.2`
- BC `net.4` (Linear 64->3)  -> `action_net`
- value path (`mlp_extractor.value_net.*`, `value_net`) left at SB3 init.

Obs parity: the BC actor was trained on `(obs - mu)/sd`. The train VecEnv is
wrapped in `FixedObsNormalize` (frozen mu/sd from the BC checkpoint), so the
warm-started weights stay in-distribution. On export the finetuned actor is
written back into the BCPolicyNet checkpoint schema with the SAME mu/sd, so the
existing verified harnesses (`k5_6_bc_eval_inenv.py`, `k5_6_bc_render_demo.py`)
run on it unchanged and the number is directly comparable to BC's 1737.3.

PPO hyperparams: lr 1e-4 (conservative, to protect the warm start), n_steps 2048,
batch_size 256, n_epochs 10, gamma 0.99, gae_lambda 0.95, clip 0.2, ent_coef 0.0,
vf_coef 0.5, max_grad_norm 0.5, seed 0, CPU. Wall 345.6s (~52-59 fps).

## Diagnostic: warm-start avoids the K5.1-K5.5 value-head collapse

The sharper question the handoff posed is answered. The value head learned from
scratch WITHOUT collapsing and WITHOUT corrupting the actor.

explained_variance per update (`finetune_metrics.ndjson`): NaN, -0.030, 0.028,
0.082, 0.160, 0.167, 0.036, 0.357, 0.442, **0.560**. value_loss fell 265 -> 192.
Throughout, the actor barely moved: clip_fraction held ~0.0077, policy_gradient
_loss order 1e-3, approx_kl <= 0.017, entropy_loss steady ~-0.04 to -0.05.

Structural reason: `net_arch=dict(pi=[64,64], vf=[64,64])` gives the actor and
critic SEPARATE networks, and the state feature extractor is parameter-free
(Flatten). The random value head's gradient shock flows only through `value_net`,
never through the actor. The K5.1-K5.5 collapses used `CnnPolicy` with a SHARED
CNN trunk, so value collapse dragged perception down (latent_vf -> 0 live dims by
update 3). Here that path does not exist. Warm-start is not the only fix; the
separate-head architecture is what makes the warm-started actor immune.

## In-env eval (the verdict)

`tools/k5_6_bc_eval_inenv.py --ckpt ppo_ft_policy.pt --seeds 1000-1009`, greedy
argmax, max_steps 1800. Held-out seeds (disjoint from BC training seeds
2000-2035), computed identically to the 845.7 / 1737.3 baselines.

| seed | steps | reason    | acts L/S/R      |
|------|-------|-----------|-----------------|
| 1000 | 1800  | timeout   | 407 / 353 /1040 |
| 1001 | 1800  | timeout   | 599 / 680 / 521 |
| 1002 | 1800  | timeout   | 476 / 892 / 432 |
| 1003 | 1800  | timeout   | 513 / 457 / 830 |
| 1004 | 1800  | timeout   | 519 / 530 / 751 |
| 1005 | 1800  | timeout   | 500 / 756 / 544 |
| 1006 | 1278  | collision | 351 / 636 / 291 |
| 1007 | 1800  | timeout   | 494 / 723 / 583 |
| 1008 | 1427  | collision | 331 / 614 / 482 |
| 1009 | 1800  | timeout   | 318 /1040 / 442 |

mean 1710.5, min 1278, max 1800, collision_rate 0.20, timeout_rate 0.80. 8/10
survive to the 1800 cap. Action counts non-degenerate on every seed: real
three-way dodging, NOT the K5.1-K5.5 constant-action collapse.

## Comparison to BC

The finetune PRESERVED the BC policy, it did not improve it: 1710.5 vs BC 1737.3
(within 1.6%), identical 0.20 collision rate. Expected for a low-LR finetune of
an actor already near the K5.2 oracle. The contribution here is not a higher
score; it is that the policy is now PPO-updated (literal RL) and still clears the
bar, and that the value head learned rather than collapsed. If a lower collision
rate is wanted, the lever is DAgger (Ross 2011) on the covariate-shift states,
NOT frame_stack and NOT a CNN change.

## RL clip

`runs/phase_k/k5_6_bc/ppo_ft/demo/seed1009/demo.mp4` (1801 frames, 30 fps, no
encode error). Rendered from the finetuned checkpoint on seed 1009; manifest
`action_counts_LSR` [318, 1040, 442] are bit-identical to the seed-1009 eval
episode, so the clip IS the evaluated episode. Player = cyan box, hazards = red
boxes, true logged game geometry.

## found-art

ADAPT. General problem: initialize an SB3 PPO actor from a pretrained
behaviorally-cloned network, then continue with RL. Searched the SB3 policy
`state_dict` warm-start pattern and the HumanCompatibleAI `imitation` library's
BC->RL handoff. Adopted the weight-copy recipe (policy_net + action_net from BC,
fresh value path) rather than writing a new trainer; reused the repo's `make_env`
factory and both K5.6 harnesses verbatim via checkpoint-schema matching.

## Artifacts (on disk; runs/ is gitignored, not tracked, per project pattern)

- `runs/phase_k/k5_6_bc/ppo_ft_policy.pt` finetuned actor in BCPolicyNet schema
- `runs/phase_k/k5_6_bc/ppo_ft/finetune.log`, `finetune_metrics.ndjson`
- `runs/phase_k/k5_6_bc/ppo_ft/eval_inenv/bc_eval_inenv_report.json`, `eval.log`
- `runs/phase_k/k5_6_bc/ppo_ft/demo/seed1009/demo.mp4` + `manifest.json` + `steps.ndjson`
- `runs/phase_k/k5_6_bc/ppo_ft/ppo_ft_sb3.zip` full SB3 model (for resume)

Tracked in git: `tools/k5_6_ppo_finetune.py`, the `k5_6_bc_render_demo.py` note
fix, and this doc. The mp4 and checkpoint live on disk only, as with the BC demo.

## Self-audit anchors

- Verdict mean 1710.5, per-seed table: `ppo_ft/eval.log` + `bc_eval_inenv_report.json` (read this session).
- EV trajectory and value_loss decline: `finetune_metrics.ndjson` (read this session).
- Clip frame count 1801, action counts bit-identical to seed-1009 eval: `demo/seed1009/manifest.json` (read this session).
- SB3 submodule names / shapes: live probe against SB3 2.8.0 (run this session).

## Honest caveats

- Finetune did not beat BC; it matched it. The win is "literal RL that holds," not a new high score.
- 2/10 seeds still collide (1006, 1008). Collision rate unchanged from BC's 0.20.
- Value EV reached 0.56, not ~1.0; the critic is useful but not perfect at 20480 steps.
- The clip is one seed (1009). The verdict rests on the 10-seed mean, not the clip.
