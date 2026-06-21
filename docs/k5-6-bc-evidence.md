# K5.6 Behavioral Cloning - Evidence

Verdict: **PASS**. The behavioral-cloning policy clears the 930.27
above-baseline bar in-env. First learned Signal Dodge policy in this
project to beat the constant-action baseline.

## Result (load-bearing)

In-env greedy eval, held-out seeds 1000-1009 (disjoint from the BC
training seeds 2000-2035, so genuine generalization, not leakage).
Same `_build_env` kwargs as the K5.2 layer-6 baseline (state mode,
`reward_shaping="none"`, `max_steps=1800`) and the same episode-length
metric, so the number is directly comparable to 845.7 / 1762.8.

- Mean episode length: **1737.3** vs bar **930.27** (+807.03).
- vs best constant (845.7, K5.2 `constant_left`): **+891.6**.
- 8/10 seeds survive to the 1800 cap; 2 collide (1546, 1427).
- Collision rate 0.20, timeout rate 0.80.
- Action counts non-degenerate on every seed: real three-way dodging,
  not the K5.1-K5.5 constant-action collapse.
- BC clone retains the expert: 1737.3 vs the K5.2 oracle's 1762.8, a
  1.4% survival gap under covariate shift.

Per-seed (steps, terminal, action counts L/S/R):

| seed | steps | terminal | L | S | R |
|------|-------|----------|------|------|------|
| 1000 | 1800 | timeout | 481 | 442 | 877 |
| 1001 | 1800 | timeout | 639 | 479 | 682 |
| 1002 | 1546 | collision | 380 | 726 | 440 |
| 1003 | 1800 | timeout | 516 | 647 | 637 |
| 1004 | 1800 | timeout | 580 | 527 | 693 |
| 1005 | 1800 | timeout | 574 | 640 | 586 |
| 1006 | 1800 | timeout | 589 | 661 | 550 |
| 1007 | 1800 | timeout | 443 | 920 | 437 |
| 1008 | 1427 | collision | 379 | 561 | 487 |
| 1009 | 1800 | timeout | 336 | 1013 | 451 |

Local artifact (runs/ is gitignored):
`runs\phase_k\k5_6_bc\eval_inenv\bc_eval_inenv_report.json`.

## Method

PPO collapsed to constant-action across K5.1-K5.5 (value-head collapse,
latent_vf -> 0 live dims by update 3). Per "method fails twice, change
it", the lever changed from RL to supervised imitation: clone the K5.2
`hazard_reactive_oracle` (which clears the bar, oracle mean 1762.8) into
a small 10-dim state MLP via cross-entropy on expert (state, action)
pairs. Supervised training structurally cannot reach the value-head
collapse: there is no value head. Respects NOT frame_stack / NOT CNN.

found-art verdict: **ADAPT** (imitation learning; DAgger, Ross 2011).
Plain PyTorch BC chosen over the `imitation` library for lowest
dependency. Eval harness ADAPTs the K5.2 layer-6 `_run_one_episode`
loop with a torch-greedy action selector so the comparison is
apples-to-apples. Renderer ADAPTs demo0's cv2 VideoWriter pattern,
drawing true game geometry instead of pixel obs.

## Pipeline

1. `tools/k5_6_bc_dataset.py` - oracle rollout to (obs, action) npz.
   Dataset: 36 seeds (2000-2035), 64,800 samples, every episode 1800
   steps (expert survives to cap on all 36), action dist L/S/R
   0.253/0.498/0.249.
2. `tools/k5_6_bc_train.py` - class-weighted 2-hidden-layer MLP (64
   units), 80 epochs. Best val_acc 0.9903, per-class recall ~0.99.
   NOTE: the val split is a random permutation over flattened
   (state, action) pairs, so seed/episode leakage inflates val_acc.
   Val accuracy is NOT the verdict for that reason.
3. `tools/k5_6_bc_eval_inenv.py` - THE verdict. Greedy argmax in-env on
   held-out seeds 1000-1009. Result above.
4. `tools/k5_6_bc_render_demo.py` - faithful top-down mp4 of the policy
   playing (player box + hazard boxes from logged geometry). Demo:
   `runs\phase_k\k5_6_bc\demo\seed1009\demo.mp4` (1801 frames, 30 fps).
   Seed 1009 action counts [336,1013,451] are bit-identical to the eval
   run's seed 1009, confirming the clip IS the evaluated episode.

## Caveats

- Greedy argmax is deterministic: 10 fixed trajectories over 10 seeds,
  computed identically to how 845.7/1762.8 were. Fair comparison.
- The 0.20 collision rate is BC's covariate-shift error compounding; it
  does not pull the mean under the bar. DAgger is the lever if a later
  goal needs the collision rate down.
- Scope: mission says "RL policy"; BC is imitation. To keep the
  deliverable genuinely RL, optional next step is PPO-finetune from the
  BC weights. If finetune re-collapses, the BC checkpoint stands as the
  demo, labeled imitation-learned. Public-sample labeling is Jeff's call.
