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

## 4. Stage 2 revised: skill 3 + shaping + reward normalization (COMPLETE, PASSED)

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

### 4a. Result (VERIFIED, PASSED the bar)

Completed 2026-07-15 20:08, 1,501,472 steps, 8,445.7s train, 118.4 steps/s,
from `runs/vzd/ppo_deadly_corridor_s3_shaped/summary.json`.

**mean 2279.14 / IQM 2279.43** over 30 deterministic raw episodes, against the
pre-registered bar of IQM decisively above the flat skill-3 IQM 683.9 (same
skill, same raw eval). That is 3.3x the bar. **PASSED**, and it also clears the
~767 untrained skill-3 anchor by 3x, so the section-3 methodology defect in
that anchor does not affect the verdict.

The distribution is the point. Every one of the 30 episodes lands in
2276.2-2281.6, i.e. **30/30 reach the armor**. The flat skill-3 run reached
~2280 in **4/30** and sat at the untrained modal 664.1885 in 15/30. So the two
fixes moved the dominant mode, which is exactly what stage 1 failed to do.
Same tight-cluster signature as the flat run's byte-identical 664.1885, but on
the success mode instead of the failure mode: deterministic policy, deterministic
map, so a repeated float is expected and is not evidence of degeneracy here.

Optimizer health at end, against the flat run's frozen signature:

| metric | flat s3 (failed) | shaped+norm s3 |
|---|---|---|
| entropy_loss | -2.07 -> **-0.0002** (frozen) | -2.08 -> **-0.0802** |
| approx_kl | -> **0.0** | **0.0028** (nonzero) |
| clip_fraction | -> **0.0** | **0.0317** (nonzero) |
| value_loss | 5.5e4 | **0.00996** |
| explained_variance | fine while frozen | **0.972** |
| ep_rew_mean | flat | **-1160 -> 3120** monotone |

Entropy did decay hard (-2.08 to -0.08). That is convergence, not the section-3a
collapse: the flat run froze at *exactly* zero kl/clip_fraction while reward went
nowhere, whereas here kl and clip_fraction stay nonzero and reward climbs
monotonically to 3120 with the critic at 0.972. The policy sharpened onto a
solution that works.

**What this does NOT establish (UNKNOWN, and it matters).**

1. **Fight or run past. RESOLVED 2026-07-18: FIGHT (HIGH).** A 30-episode
   deterministic raw eval at skill 3 of `ppo_deadly_corridor_s3_shaped/model.zip`,
   instrumented with the engine's own counters via
   `unwrapped.game.get_game_variable()` (`tools/vzd_probe_combat.py`,
   `combat_probe.json`), gives **5 kills / 5 hits every one of 30 episodes,
   0 deaths, mean final health 91.4/100**. Not combat-free. The probe reproduced
   `summary.json` to four decimals from independently written code (mean
   2279.1444 vs 2279.1443, IQM 2279.4275 vs 2279.4274), which re-verifies the
   stage-2 eval itself. A tic-level trace (`tools/vzd_trace_combat.py`,
   `_trace_combat.log`) then killed an intermediate wrong hypothesis of mine:
   seeing DAMAGECOUNT ~45 across 5 kills, I guessed KILLCOUNT was crediting
   monster infighting. The trace falsified it: every KILLCOUNT increment lands
   on the *same tic* as a HITCOUNT increment and a -1 on AMMO2. One bullet, one
   hit, one kill, five times. My "20-HP zombieman" premise was simply wrong; a
   single 5-damage pistol hit kills these. Metric caveat: `SHOTS_FIRED`/
   `accuracy` in `combat_probe.json` are contaminated (the AMMO2 baseline read
   at reset() inherits the prior episode's terminal value, giving impossible
   accuracy 1.007 and SHOTS_FIRED=1 on two episodes); KILLCOUNT/HITCOUNT/
   DAMAGE_TAKEN are clean (episode 1 is the process's first, nothing to inherit,
   and reads 5 kills; DAMAGE_TAKEN falls between episodes, which a stale max
   forbids). The trace's per-tic AMMO2 read shows true accuracy 5/5 to 5/6.
2. **Single seed.** n=1. The eval is deterministic on a deterministic map, so
   IQM over 30 episodes measures which mode this one policy landed in, not a
   distribution over training runs. A decisive claim needs multiple seeds, not
   more episodes. Do not over-read one clear.
3. **Low entropy leaves little headroom.** A near-deterministic policy (-0.08)
   has little exploration left to adapt at skill 5. The resume-finetune step may
   need entropy re-injection (raise --ent-coef on resume).

### 4b. Stage 3: skill-5 resume-finetune (COMPLETE, PASSED, combat-verified)

Resume the stage-2 shaped weights at skill 5, 1.5M -> 3.0M steps, with
`--ent-coef 0.05` reapplied after `PPO.load` (the flag was a silent no-op on
resume before fb7ff99: SB3 restores the checkpoint's saved ent_coef, so the
override was ignored while still being written into summary.json). The
re-injection is what section 4a's note 3 predicted would be needed: entropy
re-climbed from -0.08, giving the near-deterministic policy room to re-adapt.
Ran under Task Scheduler (`Sight-VZD3-S5`), 13,922s, clean finish rc=0.

**Reward: mean 2199.67 / IQM 2279.67** over 30 deterministic raw episodes
(`runs/vzd/ppo_deadly_corridor_s5_ft/summary.json`; IQM recomputed from the raw
rewards array matches stored to the penny). Pre-registered bar: decisively
above the skill-5 flat collapse IQM 93.6. This policy's own cold start at
skill 5 was IQM ~93.3, so ~93 -> ~2280 is real transfer, not a warm start.
29/30 episodes reach the armor at ~2280; 1 death (-115.9).

**Combat probe (2026-07-19, resolves the fight-or-run-past caveat at skill 5):
FIGHT, HIGH.** `tools/vzd_probe_combat.py` at skill 5, 30 deterministic
episodes (`combat_probe.json`): **kills_mean 5.83, 30/30 episodes with a kill,
29/30 full clears at 6/6 kills, 6/6 hits**, damage_taken_mean 58.8, final
health mean 41.2, 1 death (the same -115.9 episode as the eval; the probe
reproduces the eval IQM 2279.67 exactly, re-verifying it from independent
code). The modal clear is 6 kills at skill 5 versus 5 at skill 3 (whether
that is one more spawn or one fewer escapee is not established and does not
matter for the verdict). Raw-reward ambiguity closed: the score is combat,
not a distance artifact. Same metric caveat as 4a: SHOTS_FIRED/accuracy
contaminated, KILLCOUNT/HITCOUNT/DAMAGE_TAKEN clean. Demo:
`runs/vzd/demos/corridor_s5_ft.mp4`.

Still owed: seeds. Every decisive number in this doc is n=1. No transfer
claim goes in the README before at least 2 more seeds of the s3->s5 pipeline.

## 5. Results table (fill on eval, do not pre-populate)

| Stage | Method | Score (raw scenario, 30-ep deterministic) | Verdict |
|---|---|---|---|
| skill-5 flat | PPO CnnPolicy gamma 0.99, no curriculum | mean 130.5 / IQM 93.6, 14/30 identical eps | FAILED, entropy collapse |
| skill-3 flat | PPO CnnPolicy gamma 0.99, curriculum only | mean 891.1 / IQM 683.9, 15/30 identical eps | FAILED, IQM below the ~767 untrained anchor; entropy collapse |
| skill-3 shaped+norm | + game-var shaping + VecNormalize returns | mean 2279.1 / IQM 2279.4, 30/30 armor reached | **PASSED**, 3.3x the 683.9 bar; combat-verified 5 kills/5 hits every episode, 0 deaths; single seed |
| skill-5 resume-finetune | resume shaped weights at skill 5, ent-coef 0.05 reapplied | mean 2199.7 / IQM 2279.67, 29/30 armor reached, 1 death | **PASSED**, vs 93.6 flat collapse and ~93.3 own cold start; combat-verified kills_mean 5.83, 30/30 episodes with a kill; single seed |


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

## 8. Infra, resolved: it was never ViZDoom (2026-07-15)

Runs 2 and 3 died the same way, at ~290k and ~500k. Diagnosis at 3db4777 was
"stochastic ViZDoom engine death" and SafeDoom was shipped for it. **That
diagnosis was wrong.** Recorded because the error cost three runs.

Evidence against it, gathered after run 3:

- **Zero `[SafeDoom]` rebuilds in the completed 1.5M-step run**, and zero in
  run 3's 500k. If engines were dying at the rate ViZDoom#169 describes, the
  wrapper would have caught and logged some. It never fired once.
- **No worker-side traceback** in any crash. The worker did not raise, it
  vanished. SafeDoom only catches faults that surface as Python exceptions.
- **The supervisor died too**, and so did the monitor server, which shares
  nothing with the training tree. Every python process on the box went at once.
- Ruled out by tool output: Claude Desktop / MCP restart (up since 07:45),
  sleep/wake (no Kernel-Power events, wake count 0), Application error events
  (none), vizdoom crash dumps (none).

**Method defect that produced the wrong answer.** The SafeDoom fault-injection
test called `game.close()` and asserted the resulting
`ViZDoomIsNotRunningException` was caught. That exercises the path the fix was
built for, not the path that was failing. It produced confidence, not evidence.
The rule this earns: verify against the observed failure signature, not against
the fix.

**The change that worked: OS-owned processes.** FOUND-ART ADOPT, search
"Windows keep process running after parent exits scheduled task vs NSSM
service". Task Scheduler is the built-in packaged answer (NSSM is equivalent
plus a dependency). Nothing built. Detached children had died 3x for training
and 3x for the monitor; the method changed rather than the retry count.

- `runs\vzd\_run_s3_shaped_task.cmd` + schtasks `Sight-VZD3`.
- `runs\_run_monitor_task.cmd` + schtasks `Sight-Monitor`.
- `/RL HIGHEST` needs elevation and was dropped. Normal privilege is still
  OS-owned, which is the property that matters.

Result: the run completed 1.0M steps in a single leg, 8,641s, rc=0, zero
restarts, zero engine faults. Confidence that Task Scheduler is the fix:
**MEDIUM, not HIGH.** n=1 against three detached failures is suggestive, not
proof, and the root cause of the mass kill remains **UNKNOWN**. `nhi`
(Thunderbolt) 9007/9008 events bracket the run-3 death at 16:59 and 17:08 but
are not decisive. If it recurs, the supervisor now writes a timestamped
`supervisor.log` and a `SUP_HEARTBEAT` file: a stalled heartbeat with no
"leg N exited rc=" line proves killed rather than crashed, which run 3 could
not distinguish.

Kept regardless: SafeDoom (cheap, verified on its own path, a recoverable
engine exit is still worth catching), checkpoints every 50k with
`save_vecnormalize=True`, and the supervisor. The resume path was exercised for
real: run 4 restored VecNormalize stats from the step-matched 500k `.pkl` and
continued to 1.5M. Without that, a restart re-estimates the return std from 1.0
and re-inflates the returns whose scale caused the section-3a collapse, so a
naive restart would undo the fix it exists to protect.
