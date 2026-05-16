# H5 Behavior Audit Evidence

Pause-and-reframe diagnostic approved by Jeff on 2026-05-15 after the state-observation comparator slice closed as a negative result.

Approach: read per-step `godot.ndjson` traces from the latest two eval runs and classify failure modes without retraining.

Models inspected:

- **state_comparator_seed2** (`MlpPolicy`, 10k timesteps, recipe inherited from Phase D/F): `C:\Projects\Sight\runs\rl\signal_dodge_ppo_h5_state_comparator\h5_eval_state_comparator_seed2_10k_trained_only\godot-eval-trained_cnn\godot.ndjson`

- **phase_e_seed2** (`CnnPolicy/NatureCNN`, 10k timesteps, entropy recipe): `C:\Projects\Sight\runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_eval_phase_e_seed2_entropy_10k_trained_only\godot-eval-trained_cnn\godot.ndjson`


---

## state_comparator_seed2

Total episodes parsed: 10

| seed | ep_idx | length | terminal_reason | classification |
| ---- | ------ | ------ | --------------- | -------------- |
| 1000 | 1 | 1383 | collision | wall_hugging_into_collision |
| 1001 | 3 | 483 | collision | wall_hugging_into_collision |
| 1002 | 5 | 183 | collision | high_frequency_oscillation_into_collision |
| 1003 | 7 | 603 | collision | wall_hugging_into_collision |
| 1004 | 9 | 1443 | collision | wall_hugging_into_collision |
| 1005 | 11 | 363 | collision | wall_hugging_into_collision |
| 1006 | 13 | 573 | collision | wall_hugging_into_collision |
| 1007 | 15 | 273 | collision | wall_hugging_into_collision |
| 1008 | 17 | 1800 | timeout | survived_to_timeout |
| 1009 | 19 | 243 | collision | wall_hugging_into_collision |

**Aggregate label distribution:** {'wall_hugging_into_collision': 8, 'high_frequency_oscillation_into_collision': 1, 'survived_to_timeout': 1}

### Shortest collision (eval seed 1002, episode index 5)

- length=183, reward=0.0, terminal_reason='collision', terminated=True, truncated=False
- action distribution: {'-1': 0.645, '1': 0.295, '0': 0.06} (idle=0.06, reversal_rate=0.59)
- player_x trajectory: mean=165.6, min=40.0, max=360.0, span=320.0, samples_at_terciles=[355.0, 255.0, 40.0, 40.0]
- wall_hug_ratio (frames within 40px of edge): 0.0
- final_x=40.0, death_x=-1.0, collision_frame=182
- **classification: high_frequency_oscillation_into_collision**

### Longest non-timeout collision (eval seed 1004, episode index 9)

- length=1443, reward=0.0, terminal_reason='collision', terminated=True, truncated=False
- action distribution: {'-1': 0.783, '1': 0.217} (idle=0.0, reversal_rate=0.434)
- player_x trajectory: mean=39.3, min=16.0, max=360.0, span=344.0, samples_at_terciles=[355.0, 16.0, 21.0, 16.0]
- wall_hug_ratio (frames within 40px of edge): 0.914
- final_x=16.0, death_x=-1.0, collision_frame=1442
- **classification: wall_hugging_into_collision**

### Survival / timeout (eval seed 1008, episode index 17)

- length=1800, reward=1.0, terminal_reason='timeout', terminated=False, truncated=True
- action distribution: {'-1': 0.807, '1': 0.191, '0': 0.002} (idle=0.002, reversal_rate=0.381)
- player_x trajectory: mean=33.7, min=16.0, max=360.0, span=344.0, samples_at_terciles=[355.0, 16.0, 21.0, 16.0]
- wall_hug_ratio (frames within 40px of edge): 0.933
- final_x=16.0, death_x=None, collision_frame=None
- **classification: survived_to_timeout**


---

## phase_e_seed2

Total episodes parsed: 10

| seed | ep_idx | length | terminal_reason | classification |
| ---- | ------ | ------ | --------------- | -------------- |
| 1000 | 1 | 1383 | collision | wall_hugging_into_collision |
| 1001 | 3 | 483 | collision | wall_hugging_into_collision |
| 1002 | 5 | 1293 | collision | wall_hugging_into_collision |
| 1003 | 7 | 603 | collision | wall_hugging_into_collision |
| 1004 | 9 | 1443 | collision | wall_hugging_into_collision |
| 1005 | 11 | 363 | collision | wall_hugging_into_collision |
| 1006 | 13 | 573 | collision | wall_hugging_into_collision |
| 1007 | 15 | 273 | collision | wall_hugging_into_collision |
| 1008 | 17 | 1800 | timeout | survived_to_timeout |
| 1009 | 19 | 243 | collision | wall_hugging_into_collision |

**Aggregate label distribution:** {'wall_hugging_into_collision': 9, 'survived_to_timeout': 1}

### Shortest collision (eval seed 1009, episode index 19)

- length=243, reward=0.0, terminal_reason='collision', terminated=True, truncated=False
- action distribution: {'-1': 1.0} (idle=0.0, reversal_rate=0.0)
- player_x trajectory: mean=64.0, min=16.0, max=355.0, span=339.0, samples_at_terciles=[355.0, 16.0, 16.0, 16.0]
- wall_hug_ratio (frames within 40px of edge): 0.737
- final_x=16.0, death_x=-1.0, collision_frame=242
- **classification: wall_hugging_into_collision**

### Longest non-timeout collision (eval seed 1004, episode index 9)

- length=1443, reward=0.0, terminal_reason='collision', terminated=True, truncated=False
- action distribution: {'-1': 1.0} (idle=0.0, reversal_rate=0.0)
- player_x trajectory: mean=24.1, min=16.0, max=355.0, span=339.0, samples_at_terciles=[355.0, 16.0, 16.0, 16.0]
- wall_hug_ratio (frames within 40px of edge): 0.956
- final_x=16.0, death_x=-1.0, collision_frame=1442
- **classification: wall_hugging_into_collision**

### Survival / timeout (eval seed 1008, episode index 17)

- length=1800, reward=1.0, terminal_reason='timeout', terminated=False, truncated=True
- action distribution: {'-1': 1.0} (idle=0.0, reversal_rate=0.0)
- player_x trajectory: mean=22.5, min=16.0, max=355.0, span=339.0, samples_at_terciles=[355.0, 16.0, 16.0, 16.0]
- wall_hug_ratio (frames within 40px of edge): 0.964
- final_x=16.0, death_x=None, collision_frame=None
- **classification: survived_to_timeout**


---

## Synthesis

Both models — `MlpPolicy` on state observations and `CnnPolicy/NatureCNN` on pixel observations, each trained for 10k timesteps with the H5 entropy recipe — converge on essentially the same policy: **drive left, oscillate or stay at x=16 (the left wall clamp), survive until a hazard happens to spawn near the left edge**.

The pixel `CnnPolicy` is more degenerate than the state `MlpPolicy`: in all three representative episodes its action distribution is `action=-1` 100% of the time. The state `MlpPolicy` produces ~78–81% `action=-1` with ~19–22% `action=+1` and ~0% `action=0`. Both end up wedged against the left wall for >90% of episode frames.

The dominant determinant of episode length is the eval seed (hazard spawn pattern), not the policy. On seed 1000 both models died at frame 1383. On seed 1004 both models died at frame 1443. On seed 1008 both models survived to timeout at 1800. Wall-hugging policies plus seed-determined hazard patterns produce nearly identical outcomes across radically different encoders and observation channels.

### Failure-mode breakdown across the 20 representative-model episodes

| Failure mode                                  | state_comparator_seed2 | phase_e_seed2 |
| --------------------------------------------- | ---------------------- | ------------- |
| wall_hugging_into_collision                   | 8                      | 9             |
| high_frequency_oscillation_into_collision     | 1                      | 0             |
| survived_to_timeout (still wall-hugging)      | 1                      | 1             |
| **All collision events**                       | **9**                  | **9**         |

The single state-mode oscillation case (seed 1002, 183 frames, fastest death) does not reflect a different strategy. Wall_hug_ratio there is 0.0 because the episode ended at frame 183 before the agent had finished its initial leftward sweep; final_x=40 still shows it was on its way to the wall.

### Why this is the expected fixed point of the current reward

Reward is `+1` per non-terminal step. Terminal collision yields 0. The agent's optimal policy under this objective alone is the policy that maximizes expected number of non-terminal steps. There is no per-step reward signal that distinguishes "moved skillfully to avoid a hazard" from "happened to be standing somewhere safe." Two corollaries follow mechanically:

1. **The left wall is a fixed point of the gradient.** Once the agent finds that `action=-1` is locally stable (movement clamps at x=16, so further `-1` actions don't reduce reward) and that hazards spawning at non-left x positions miss the agent, gradient descent has no reason to leave that policy. Trying `action=0` or `action=+1` from x=16 is high-variance: it may briefly reduce reward by exposing the agent to right-side hazards. Gradient on `+1/step` survival reward therefore *punishes* exploration away from the wall.
2. **The agent is doing approximately the same thing across encoders.** Encoder choice does not break the symmetry because the symmetry is in the reward shape, not the perception channel. The state `MlpPolicy` having ~20% right-action exploration vs the pixel `CnnPolicy`'s 100% left is best read as residual entropy bonus noise (`ent_coef=0.01`) on top of the same underlying drift toward the wall, not as a meaningful difference in behavior.

### What this rules in and rules out

- **Rules in:** the reward function is the proximate driver of the failure mode. `+1/step` survival is being optimized correctly; the bad behavior IS the gradient's solution to the reward.
- **Rules in:** the agent does occasionally survive (1/10 timeouts in each model), so the wall-hugging policy is not pathological from the reward's perspective; it is the *correct* answer to the survival objective.
- **Rules out:** perception / encoder / observation channel as the H5 blocker. Two distinct encoders trained on two distinct observation channels produce indistinguishable behavior. This generalizes the Phase F result.
- **Rules out:** training-budget-only as the H5 fix. The agent has already converged to a local optimum of the current reward; adding timesteps moves it deeper into that optimum, not toward avoidance.
- **Does not rule out:** entropy-coefficient or learning-rate tuning, but only as a way to escape this specific local optimum; the underlying reward gradient still points back to wall-hugging once exploration noise is reduced.

### Recommended next experiment lever

Reward shaping. Specifically a per-step reward signal that distinguishes "near a hazard" from "far from any hazard," so that being pinned against a wall while a hazard is approaching is not as rewarding as moving away from that hazard. Plausible shapes (NOT to be implemented without a charter amendment per H5 plan section 7):

- **Distance bonus:** add a small `+epsilon * min_distance_to_active_hazard` term, normalized.
- **Approach penalty:** subtract a small `delta * d(distance_to_nearest_hazard)/dt` when the agent is moving toward a hazard.
- **Lateral-utility bonus:** reward the agent for being within the lane of an active hazard only while that hazard is still far above; penalize the same position when the hazard is close.

Any of these requires the H5 plan section 7 amendment process. None can be adopted by Claude unilaterally.

### Secondary observation worth flagging

The eval-side `godot.ndjson` contains two `episode_start` events per `controller_reset_received` (20 vs the 10 expected for seeds 1000-1009), producing alternating empty-episode shells in the trace. The audit script filters them by requiring `len(steps) > 0` before assigning eval seeds. This does not affect any prior eval summary — those use `episodes.ndjson` from the Python side, which is one row per `terminated` or `truncated` step. But the Godot-side double emit is a potential cleanup target for `games/signal-dodge/scripts/main.gd` if/when the project does another evidence-instrumentation pass.

### What this does NOT establish

- It does not establish that the same pattern holds at 50k or 100k timesteps. Phase D's 50k single-seed point also showed survival-style behavior, but a behavioral audit of that artifact has not been done.
- It does not establish that ALL pixel models behave the way Phase E seed 2 does. The audit picked one representative pair (the latest state-comparator seed 2 and one Phase E pixel seed) to test the hypothesis economically; broader sampling could revisit Phase F seeds or Phase E seeds 1 and 3.
- It does not establish that the seeded_random or untrained_cnn negative controls do NOT also wall-hug. They might, by chance. A control audit would falsify or confirm whether wall-hugging is a learned strategy vs a baseline tendency. This was not run.

### Verdict for the next session

Pause H5 training sweeps until a reward-amendment proposal is on the table. The encoder, the observation channel, the entropy coefficient, the frame-stack contract, and the training budget have all been varied; none of them is the proximate cause of bad in-game behavior. The reward function is. Continuing to tune within the current reward shape is wheel-spinning.

This audit closes Jeff's pause-and-reframe diagnostic. H5 remains open. Phase G remains NOT triggered. Next experiment lever requires a charter-level reward amendment proposal, not another sweep.
