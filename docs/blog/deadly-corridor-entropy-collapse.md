---
title: "How PPO froze solid on ViZDoom's deadly_corridor, and what fixed it"
layout: default
---

# How PPO froze solid on ViZDoom's deadly_corridor, and what fixed it

*July 2026. Part of [Sight](https://github.com/trzz333/sight), a local-first RL
lab that trains small policies from scratch on one gaming laptop. The full run
log with every number in this post is
[`docs/vzd-deadly-corridor-findings.md`](../vzd-deadly-corridor-findings.md).*

`deadly_corridor` is one of ViZDoom's bundled scenarios: a corridor with three
enemy pairs on the flanks and body armor at the far end. The agent has to
advance under fire. The reward is distance shaping toward the armor plus a
death penalty of 100, at difficulty (doom_skill) 5. My starting point was a
PPO recipe that had already worked on `defend_the_center`: SB3 CnnPolicy from
pixels, grayscale 60x80, frame-skip 4, frame-stack 4, gamma 0.99, running on
an RTX 4080 Laptop GPU at about 110 environment steps per second.

That recipe, pointed straight at skill 5, produced a policy with an
interquartile-mean (IQM) return of 93.6 over 30 deterministic evaluation
episodes. 14 of the 30 episode returns were byte-identical floats. The policy
had converged on sprinting down the corridor into the first enemy pair,
collecting a burst of distance reward, and dying.

## The wrong first theory

My first diagnosis was a reward-landscape story: nothing in the flat scenario
reward pays the agent to survive the enemies, so "sprint and die" is a wide,
easy basin. That story is true as far as it goes, and it suggested the fix the
prior art also suggests (a difficulty curriculum plus reward shaping that pays
for kills). A curriculum-only test at skill 3 came back IQM 683.9 against an
untrained anchor of roughly 767. Training for 1.5M steps had not moved the
dominant behavior at all: 15 of 30 episodes landed on the same float that an
untrained policy produces.

## What the training log actually said

The real mechanism was sitting in the SB3 progress log of both failed runs:

    entropy_loss   -2.07 -> -0.0002   (ln(8) = 2.079 is uniform over 8 actions)
    value_loss      1.3e3 -> 5.5e4
    approx_kl       0.0155 -> 0.0
    clip_fraction   0.277  -> 0.0

Policy entropy collapsed from uniform to a point mass within the first quarter
of training. Once the policy is deterministic, the PPO probability ratio is
always 1, approx_kl and clip_fraction sit at exactly zero, and no surrogate
gradient can move it again. The last 1.2M of 1.5M steps were spent optimizing
a frozen policy.

The cause is reward scale. The corridor's returns are roughly 1000x
`defend_the_center`'s, so the value loss sat near 5e4. SB3's CnnPolicy shares
one CNN trunk between the policy and value heads, which means value-fitting
gradients (weighted by vf_coef 0.5) flow through the same features the policy
depends on, and at that magnitude they swamp everything else, entropy bonus
included. A detail that matters: SB3 normalizes advantages per minibatch by
default, so the policy-gradient path is already scale-free. The route by which
reward scale reaches the policy is the shared trunk via the value loss.

This is a known, named failure. Pop-Art (van Hasselt et al., NeurIPS 2016)
exists precisely because value targets spanning orders of magnitude produce
destructive gradients in lower layers, and it proves an equivalence between
normalizing targets and rescaling those gradients. Phasic Policy Gradient
(Cobbe et al., 2020) names the shared-trunk interference problem directly. The
packaged remedy is return normalization, standard practice per "The 37
Implementation Details of PPO" (ICLR Blog Track 2022): divide rewards by the
standard deviation of a rolling discounted return and clip. In SB3 that is
`VecNormalize(norm_reward=True, clip_reward=10)`, one line. I did not build
anything here; the search for prior art was the work.

One consequence was worth writing down before running anything: reward shaping
alone, the other half of the published recipe, would also have failed. Adding
a kill bonus to an optimizer that saturates its trunk by 300k steps changes
nothing about the saturation. Normalization is a precondition for the shaping
to matter, which is why the two ship together in the runs below.

## The pipeline that worked

Stage 1 trains at skill 3 with two changes: VecNormalize return scaling, and
game-variable reward shaping (per-step deltas on top of the scenario reward:
+200 per hit landed, -10 per point of damage taken, -5 per round of ammo
spent, coefficients from the published deadly_corridor recipes). Evaluation is deliberately
raw (unshaped, unnormalized, at the training skill) so every number stays
comparable to the failed baselines.

The optimizer health flipped immediately. At 47k steps, value loss was 0.32
where the flat run had 5.5e4; at the end of 1.5M steps, entropy sat at -0.08
with approx_kl and clip_fraction still nonzero, explained variance at 0.972,
and episode reward climbing monotonically. The eval: IQM 2279.4, with 30 of 30
episodes reaching the armor. The failed run reached it in 4 of 30.

Stage 2 resumes those weights at skill 5 for another 1.5M steps. Two mechanics
in the resume path did real work:

- Checkpoints pair with their VecNormalize statistics by step count. Resuming
  with fresh statistics would restart the return-std estimate at 1.0 and
  re-inflate the exact scale that caused the collapse.
- `PPO.load` silently restores the checkpoint's entropy coefficient, so a
  `--ent-coef 0.05` flag on the resume command line was a no-op until the
  script reapplied it after loading. Stage 1 ended near-deterministic
  (entropy -0.08), and the finetune needs that entropy re-injected to have any
  exploration budget for the harder skill. This one silent-no-op bug cost a
  run before it was caught.

The skill-5 result: IQM 2279.67 against the flat-PPO collapse at 93.6, and
against this same policy's own skill-5 cold start of roughly 93 before
finetuning. 29 of 30 episodes cleared the corridor.

## Is it actually fighting?

A raw corridor score is ambiguous: distance shaping means a policy could in
principle score well by slipping past enemies. The scenario reward cannot
distinguish that from combat, so I probed the engine directly, reading
KILLCOUNT, HITCOUNT, and DAMAGE_TAKEN through
`unwrapped.game.get_game_variable()` during a separate 30-episode
deterministic eval, from independently written code.

At skill 5 the policy kills 5.8 enemies per episode on average, with a kill in
all 30 episodes and a full 6-of-6 clear in 29. The probe also reproduced the
training script's eval IQM exactly, which re-verifies that number from a
second code path. A tic-level trace killed one of my own wrong hypotheses
along the way: I suspected KILLCOUNT was crediting monster infighting, but
every KILLCOUNT increment lands on the same tic as a HITCOUNT increment and an
ammo decrement. One bullet, one hit, one kill. Two counters I planned to
report, SHOTS_FIRED and derived accuracy, turned out to be contaminated by a
stale ammo baseline at episode reset (episode 2 inherits episode 1's terminal
value, yielding an impossible accuracy of 1.007), so they are omitted
everywhere and only the clean counters are reported.

## Replication

Everything above was one seed, and a deterministic eval on a deterministic map
measures which mode one policy landed in, so more episodes cannot substitute
for more seeds. The full two-stage pipeline re-ran from scratch on two more
seeds. Six passing evals across the three seeds land inside a 3.9-point band
(2276.58 to 2280.44) against bars of 683.9 and 93.6, and every skill-5 combat
probe reports 5.7 to 5.9 kills per episode with 28 or more of 30 episodes
surviving. Seed 2 ran the whole pipeline unattended under Windows Task
Scheduler, including the automatic stage handoff.

## What I took from it

The diagnosis order was backwards at first. I reached for an environment-level
story (the reward landscape) when the optimizer-level evidence (entropy,
value loss, approx_kl) was already in the log and fully determined the fix.
Optimizer health metrics are cheap to read and settle questions that reward
curves cannot; in the flat runs, explained variance looked fine on a policy
that had been frozen for a million steps.

Second, the failure had a name and a literature. The working fix was assembled
entirely from published parts: return normalization from the PPO
implementation-details lineage, difficulty curriculum and kill-paying shaping
from the ViZDoom literature, entropy re-injection on resume from reading SB3's
load semantics. The project keeps a standing rule to search for prior art
before building, and this track is the clearest case yet of the rule paying
for itself, in both directions: it prevented a from-scratch Pop-Art
implementation (VecNormalize is the packaged version of the same idea) and it
prevented a shaping-only run that the entropy analysis says would have burned
four hours on a frozen policy.

Third, verify claims from a second code path. The combat probe exists because
a raw score cannot say what behavior produced it. It ended up doing double
duty, reproducing the eval numbers to four decimals from independent code and
catching two contaminated metrics before they reached any write-up.

Artifacts: [repo](https://github.com/trzz333/sight), with the
[full findings doc](../vzd-deadly-corridor-findings.md), a demo GIF in the
README, and per-run `summary.json` files with all 30 episode returns per eval.
Models are on Hugging Face Hub with replay videos.
