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

## 3. Curriculum stage 1: skill 3, flat reward (COMPLETE, FAILED the bar)

PPO CnnPolicy, gamma 0.99, flat scenario reward, skill 3, 1.5M steps.
Completed 2026-07-14 14:41, 13,310s train, 112.7 steps/s.

Result: mean 891.1, **IQM 683.9** over 30 deterministic episodes.
Pre-registered bar was IQM decisively above BOTH 93.6 (skill-5 flat collapse)
AND ~767 (untrained skill-3 smoke). IQM 683.9 is BELOW 767. **FAILED.**
Goalposts not moved: mean (891) clears 767, but the bar was written on IQM and
IQM is what it is judged on.

Distribution is bimodal: 15 of 30 episodes are byte-identical at
664.1885223388672, and 4 of 30 reach ~2280 (armor reached). The modal value
664.1885223388672 is *the same float* the untrained 2k-step smoke produced.
Training did not move the dominant mode; it only added an occasional (13%)
success mode. Confidence HIGH, from `runs/vzd/ppo_deadly_corridor_s3/summary.json`.

Methodology defect found: the ~767 anchor came from a **3-episode** smoke
(`_smoke_corridor.log`, rewards [664.19, 972.32, 664.19]); its "IQM" over 3
episodes is just the mean. Comparing IQM-of-30 to that is not like-for-like.
The verdict does not hinge on it (683.9 ~= the untrained modal 664), but a
30-episode untrained anchor is owed.

## 3a. The real mechanism: entropy collapse from reward scale (both flat runs)

Flat reward did not fail because "the reward landscape lacks an incentive to
fight". It failed because PPO's optimizer broke. From the stage-1 log
(`_parse_fields.py`):

    entropy_loss   -2.07 -> -0.0 by q1 -> -0.00021 at end
    value_loss     1.3e3 -> 6.9e4 -> 5.5e4
    approx_kl      0.0155 -> 0.0 -> 0.0
    clip_fraction  0.277 -> 0.0 -> 0.0

Entropy went from 2.07 (uniform over 8 actions; ln(8)=2.079) to ~0 within the
first quarter of training. The policy became a point mass, so the PPO ratio is
always 1, approx_kl and clip_fraction are 0, and no gradient can move it. The
last ~1.2M of 1.5M steps were wasted compute on a frozen policy. The identical
signature is present in the skill-5 flat run.

Mechanism: the corridor reward is ~1000x defend_the_center's scale, so
value_loss sits at ~5e4. SB3's CnnPolicy **shares the features extractor**
between value and policy heads, so value-fitting gradients (x vf_coef 0.5)
swamp the entropy bonus (ent_coef 0.01 x ~2.0) by orders of magnitude and
saturate the shared trunk. This is the project's own pre-registered
"high kl/clip_fraction at low clip_range is a reward-scale signature,
normalize first" rule, which the early blocks show exactly (kl 0.32/1.19/0.84,
clip_frac 0.77/0.80) before the freeze.

Consequence for the plan: **shaping alone would also have failed.** Adding
hitcount*200 to an optimizer that saturates by 300k steps changes nothing.
Reward normalization is a precondition, not an alternative.

## 4. Stage 2 revised: skill 3 + shaping + reward normalization (IN FLIGHT)

The pre-registered next step was "skill 1 + shaping". Revised on two pieces of
new evidence:

1. Reward normalization is mandatory (section 3a), and was not in that plan.
2. Skill 1 is trivial: a 2k-step **untrained** policy evals ~2280 at skill 1
   (`runs/vzd/_smoke_shape`), because nothing meaningfully opposes walking to
   the vest. A skill-1 eval therefore cannot distinguish "learned to fight"
   from "walked forward", so it would burn ~4h for an uninformative number.

Running instead: skill 3, `--shape-reward --norm-reward`, 1.5M steps, out
`runs/vzd/ppo_deadly_corridor_s3_shaped`. Skill 3 contains real combat and
keeps the eval directly comparable to the flat skill-3 IQM 683.9, making this a
clean A/B on the two fixes.

Eval is deliberately RAW (unshaped, unnormalized) at the training skill so the
number stays comparable to 683.9 and 93.6.

Early health at ~47k steps, against the flat run at the same point:

| metric | flat s3 | shaped+norm s3 |
|---|---|---|
| value_loss | 5.5e4 | **0.32** |
| entropy_loss | -> ~0 by q1 | **-2.08 -> -1.89** (still exploring) |
| approx_kl | 0.0 | **0.001-0.003** |
| clip_fraction | 0.0 | **0.11** |

Confidence HIGH that the optimizer pathology is fixed; MEDIUM on the outcome.
Re-check entropy at 300-400k, the point where the flat run had already frozen.
Note ep_rew_mean is now ~-1000 and not comparable to the flat curve: the shaped
reward charges damage_taken*10, so a full-health death costs about -1000.

## 5. Results table (fill on eval, do not pre-populate)

| Stage | Method | Score (raw scenario, 30-ep deterministic) | Verdict |
|---|---|---|---|
| skill-5 flat | PPO CnnPolicy gamma 0.99, no curriculum | mean 130.5 / IQM 93.6, 14/30 identical eps | FAILED, entropy collapse |
| skill-3 flat | PPO CnnPolicy gamma 0.99, curriculum only | mean 891.1 / IQM 683.9, 15/30 identical eps | FAILED, IQM below the ~767 untrained anchor; entropy collapse |
| skill-3 shaped+norm | + game-var shaping + VecNormalize returns | TBD | in flight |
| skill-5 resume-finetune | resume shaped weights at skill 5 | TBD | not started |


## 6. FOUND-ART on the failure (verdict ADOPT)

Generalized problem, stripped of project vocabulary: *policy entropy collapse in
PPO driven by large-magnitude returns, where a shared actor-critic trunk lets
value-loss gradients dominate the policy.* Searches run this turn: "PPO entropy
collapse large reward scale value loss dominates shared feature extractor";
"PopArt learning values across many orders of magnitude reward scale
normalization RL"; "Phasic Policy Gradient shared network interference between
policy and value function optimization"; "37 implementation details of PPO
reward scaling normalization VecNormalize".

**FOUND-ART: ADOPT** - this is a textbook, named, documented PPO failure mode,
and the fix already shipped (VecNormalize return scaling + clip) is the
canonical published remedy, not a hand-roll. Nothing to build.

Prior art (closest first):

- **Pop-Art, van Hasselt et al., NeurIPS 2016, "Learning values across many
  orders of magnitude"** [VERIFIED this turn]. Adaptive target normalization for
  value learning when return magnitude varies by orders of magnitude. Closest
  theoretical match: it explicitly establishes *an equivalence between
  normalizing targets and scaling gradients in lower layers*, which is exactly
  our mechanism (huge value targets -> huge gradients in the shared CNN trunk).
  Motivated by removing reward clipping in Atari DQN.
- **"The 37 Implementation Details of PPO", ICLR Blog Track 2022 + CleanRL**
  [VERIFIED]. Reward scaling via VecNormalize (rewards divided by the std of a
  rolling discounted sum of returns) then clipped to [-10, 10] is *standard
  documented PPO practice*, not an exotic lever. This is precisely what we
  shipped. Maturity: it is the reference implementation lineage.
- **Phasic Policy Gradient, Cobbe et al. 2020 (OpenAI)** [VERIFIED]. Names the
  shared-trunk problem directly: any method jointly optimizing policy and value
  in one network must weight them, and "there is always a risk that the
  optimization of one objective will interfere with the optimization of the
  other". Supplies two cheaper levers than PPG itself: detach the value gradient
  at the last shared layer during the policy phase, or simply lower vf_coef (the
  Procgen competition winner used 0.25 after finding gradient-stopping unstable).
- **"Revisiting Design Choices in PPO"** [VERIFIED]. Confirms reward scaling is
  crucial to PPO's success, and that a simple constant reward scaling can match
  the more complex return-std scheme.
- **"No Representation, No Trust: Connecting Representation, Collapse, and Trust
  Issues in PPO" (arXiv 2405.00662)** [VERIFIED]. Adjacent, not identical:
  documents PPO feature-rank collapse, growing pre-activation norms, and the
  resulting breakdown of the clipping trust region. Explicitly notes their rank
  collapse is *distinct* from typical entropy collapse (it yields high but
  trivial entropy). Ours is the ordinary entropy collapse, so this is a related
  failure family, not our case. Recorded to avoid overselling the match.

**Correction this forced on our own mechanism claim.** Earlier phrasing said
huge advantages swamp the entropy bonus. That part is wrong: SB3 PPO sets
`normalize_advantage=True` by default and normalizes advantages per minibatch,
so the policy-gradient path is already scale-free. The pathway that actually
carries reward scale into the policy is the **shared trunk via value loss**,
which is exactly what Pop-Art's target-normalization/lower-layer-gradient
equivalence describes. The mechanism stands; the sloppy version of it does not.

Checked footgun, does NOT apply here: CleanRL flags that using a different gamma
in the reward-normalization wrapper than in PPO is technically incorrect. Ours
match (PPO gamma 0.99, SB3 VecNormalize default gamma 0.99).

Gap: the literature settles the optimizer fix. It does not settle the
corridor-specific questions: whether the 200/10/5 shaping coefficients are right,
and whether skill-3 -> skill-5 transfer holds. Those stay empirical.

Recommendation: keep VecNormalize (shipped, running). If entropy collapses
again, escalate in this order, cheapest first: (1) vf_coef 0.5 -> 0.25 (the
PPG/Procgen-winner lever, one line), (2) raise --ent-coef, (3) detach the value
gradient at the last shared layer or set share_features_extractor=False. Do NOT
implement Pop-Art: VecNormalize is the packaged version of the same idea, and
building it would cost roughly a day to reproduce a wrapper we already import.
Effort saved by the search: avoided a from-scratch Pop-Art build, and avoided
the shaping-only path that would have burned another ~4h run on a frozen policy.

## 7. Infra failure: run 1 of the shaped config crashed (2026-07-14 ~23:00)

First shaped+norm launch died at ~47k steps after ~35 minutes with
`vizdoom.vizdoom.ViZDoomUnexpectedExitException: Controlled ViZDoom instance
exited unexpectedly`, followed by `BrokenPipeError [WinError 109]` and `EOFError`
in `SubprocVecEnv.step_wait` as the parent found a dead worker. The Doom engine
subprocess died inside a normal `env.step`, taking one of the 8 workers and
therefore the whole run with it.

Root cause confidence LOW. The flat s3 run survived 4h on the same 8-worker
SubprocVecEnv, so the shaping wrapper is weakly implicated (n=1), but this is
also a known intermittent ViZDoom failure. Not over-theorized on one sample.
Relaunched fresh (pid 21264). If it dies the same way again, that is failure
twice and the method changes: drop `n_envs`, and/or add a supervisor that
auto-resumes from the newest checkpoint, and/or stop `ShapedCorridorReward.step`
from silently swallowing exceptions with a bare `except Exception` (which can
hide the first engine error and is a real defect regardless).
