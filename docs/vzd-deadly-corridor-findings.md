# ViZDoom deadly_corridor: the fail-to-curriculum story (VZD-3)

Run story for the third ViZDoom track. Scenario `VizdoomDeadlyCorridor-v1`:
a corridor with enemies on both sides and body armor at the far end. The agent
must survive gunfire while advancing. cfg skill 5, death penalty 100, no living
reward, WAD distance shaping toward the armor, Discrete(8), 2100-tic timeout.
Untrained skill-3 eval reward ~767 (distance shaping alone, no combat).

This doc is written incrementally as the track runs. Sections marked TBD are
placeholders and carry no result until an eval fills them.

## 1. Flat-reward PPO at skill 5 collapses (VERIFIED, failed)

First attempt was the defend_the_center recipe unchanged: PPO CnnPolicy,
gamma 0.99, flat scenario reward, straight at skill 5.

Result: mean 130.5 / IQM 93.6 over 30 deterministic episodes, with 14 of 30
episodes byte-identical. approx_kl and clip_fraction collapsed to ~0 early in
training and stayed there. The policy converged to a sprint-and-die local
optimum: rush forward along the distance-shaping gradient, take the armor's
worth of progress reward, die to the first enemy pair. Because the behavior is
deterministic and short, the critic's explained variance looked fine while the
policy had stopped improving.

Confidence: HIGH. Artifacts: failure clip `runs/vzd/ppo_deadly_corridor/gameplay_fail_s5.mp4`
(15.3 MB), baseline run dir `runs/vzd/ppo_deadly_corridor/` preserved as evidence.

Diagnosis: this is a reward-landscape trap, not a hyperparameter miss. Distance
shaping plus a large terminal death penalty makes "sprint and die" a wide, easy
basin. Nothing in the flat reward pays the agent to survive the enemies, so it
never learns to fight.

## 2. Found-art: the published recipe is curriculum PLUS shaping (verdict ADAPT)

Search run: "ViZDoom deadly corridor PPO doom_skill curriculum training".
Verdict: ADAPT, not BUILD. The obstacle is a known, solved one.

- Khan 2025 (Computer Animation & Virtual Worlds) reports deadly_corridor
  learnable through skill 5 using a difficulty curriculum (train up from skill 1)
  combined with game-variable reward shaping.
- nicknochnack/DoomReinforcementLearning uses an s1..s5 cfg curriculum plus a
  shaped reward: movement + damage_taken_delta*10 + hitcount_delta*200 +
  ammo_delta*5. Killing enemies is paid for directly.
- The structurally different family behind the same result is intrinsic
  exploration: RND (callumhay/vizdoom_ppo_rnd), ICM (mehdiboubnan). Held in
  reserve if shaping is not enough.

Key risk the prior art flags: skill-3 success can be combat-free (just run to the
vest), while skill 5 requires killing the first pair. So a curriculum ALONE may
transfer poorly at the skill-5 step. Published success uses both levers. Curriculum
is being tried first as the single cheapest change; shaping is pre-registered as
the next change if the curriculum-only transfer stalls, exactly as the prior art
predicts it might.

## 3. Curriculum stage 1: skill 3, flat reward (IN FLIGHT, no verdict yet)

PPO CnnPolicy, gamma 0.99, flat scenario reward, skill 3, 1.5M steps.
tools/vzd_ppo_train.py --env-id VizdoomDeadlyCorridor-v1 --doom-skill 3
--steps 1500000 --out runs/vzd/ppo_deadly_corridor_s3

Live curve health at ~88k steps (~6%): ep_rew_mean climbing (last ~1020, already
above the ~767 untrained smoke), ep_len_mean 44 to 32 as it reaches the armor
faster, approx_kl and clip_fraction both active. One flag: explained_variance
pinned near 0.01, so the critic is not yet fitting the high-magnitude, high-variance
returns. Reward normalization is the pre-registered lever if this persists; not
applied to a running healthy job on a soft flag alone.

Eval bar (pre-registered): IQM decisively above BOTH 93.6 (the skill-5 flat
collapse) AND ~767 (untrained skill-3 smoke), with ep_len_mean well above 14
(the collapse floor). TBD: fill from `runs/vzd/ppo_deadly_corridor_s3/summary.json`
on DONE.

## 4. Curriculum stage 2: resume-finetune at skill 5 (NOT STARTED)

If stage 1 passes: resume the stage-1 weights at skill 5.
--resume runs/vzd/ppo_deadly_corridor_s3/model.zip --doom-skill 5
--out runs/vzd/ppo_deadly_corridor_s5ft (--resume treats --steps as ADDITIONAL;
verify from the log at launch). TBD.

If stage 2 fails: add the Renotte-style game-variable shaping wrapper (HEALTH
already in cfg; verify HITCOUNT/ammo variable names at implementation),
coefficients per the cited notebook. Then RND/ICM if still stuck.

## 5. Results table (fill on eval, do not pre-populate)

| Stage | Method | Score | Verdict |
|---|---|---|---|
| skill-5 flat | PPO CnnPolicy gamma 0.99, no curriculum | mean 130.5 / IQM 93.6, 14/30 identical eps | FAILED, sprint-and-die local optimum |
| skill-3 curriculum s1 | PPO CnnPolicy gamma 0.99, flat reward | TBD | in flight |
| skill-5 resume-finetune | resume s3 weights at skill 5 | TBD | not started |
