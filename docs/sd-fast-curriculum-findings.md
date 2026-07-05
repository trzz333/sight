# sd-fast start-state curriculum (from-scratch lever)

Scope call ruling (Jeff, this session): KEEP PURSUING FROM-SCRATCH. So the
imitation-vs-from-scratch decision is settled in favor of continuing from
scratch on a structurally NEW lever (every pre-registered lever had failed:
CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet exploration, PBRS reward
geometry; contract forbids retrying a failed method harder).

## Lever

found-art verdict ADAPT. Generalized problem: from-scratch deep RL trapped by a
deceptive local optimum (a passive constant-action policy already scores ~746
replica / 845.7 Godot) plus long-horizon credit assignment (+1/step survival
credited across ~800 steps). Web search (curriculum learning / deceptive reward
/ hard exploration) surfaced the un-tried branch: a curriculum, distinct from
the exploration-pressure family (CMA-MAE, NoisyNet) and the reward-geometry
family (PBRS) that already failed. Prior art: curriculum learning (Bengio 2009),
reset-state / return-to-states (Go-Explore, Ecoffet 2019), adaptive task
generation (arXiv 2007.00350), and a direct analog escaping a strong local
optimum via curriculum (arXiv 2410.16790, 42%->66%).

Implementation: `tools\sd_fast_ppo_curriculum.py`. A start-state curriculum,
`CurriculumSDF` subclass of `SignalDodgeFast` (base env left byte-identical, so
the eval harness and the imitation number are untouched). At reset, inject
`curriculum_n_init` hazards above the player (headroom 100px, no reset
collision, no insta-death). `AnnealCurriculum` callback anneals the count
linearly from n_init_max=6 to 0 over the first anneal_frac=0.7 of training, so
the run ENDS on the true clean-start distribution. Everything else is the m21
recipe verbatim (gamma 0.999, gae 0.95, n_steps 512, batch 512, ent 0.01, lr
3e-4, 8 envs, MlpPolicy [64,64], VecNormalize). Eval is UNCHANGED: greedy on the
standard clean-start env, held-out seeds 5000-5029, via `sd_fast_ppo.evaluate`,
so numbers are directly comparable to the m21 none arm. Reward stays "none".

Smoke-verified this session: clean-start (curriculum off) obs byte-identical to
base env; n_init_max=6 injects 6 hazards with no reset collision; live mutation
of curriculum_n_init reflected in reset; 20k-step end-to-end train+eval ran.

## Result: 5-seed arm COMPLETE, judged by rliable

All five curriculum seeds trained (5M, m21-verbatim + start-curriculum) and
eval'd greedy on the held-out block 5000-5029. The `tools\sd_fast_reliability.py`
curriculum arm reload-eval reproduced every summary mean exactly (independent
cross-check, so the summaries are trustworthy). Per-seed held-out means:

| seed | m21 none | curriculum | curr clears bar? |
|---|---|---|---|
| 0 | 1119.4 | 1743.1 | yes (af 0.40/0.22/0.39, diverse) |
| 1 |  598.0 | 1704.3 | yes (af 0.78/0.13/0.10, skewed but dodging) |
| 2 |  887.7 |  887.7 | no (collapse) |
| 3 |  669.9 |  669.9 | no (collapse, below best-constant 746.3) |
| 4 |  643.1 | 1591.6 | yes (af 0.42/0.20/0.38, diverse) |

rliable (Agarwal 2021, seed-level bootstrap, BOOT=50000):

| arm | IQM | 95% CI | clears 930 | pool mean |
|---|---|---|---|---|
| none | 733.6 | [613.0, 1042.2] | 1/5 | 783.6 |
| curriculum | 1394.5 | [742.5, 1730.1] | 3/5 | 1319.3 |

Comparison curr vs none: IQM diff **+661.0**; P(IQM_curr > IQM_none) [paired seed
bootstrap] = **0.970**; POI [rliable Mann-Whitney] = **0.743**. HIGH (anchor:
`tools\sd_fast_reliability.py` output this session; cache
`runs\sd_fast\reliability_eval_cache.json`).

## Verdict: BETTER THAN NONE, NOT YET RELIABLE ENOUGH TO PORT

The curriculum is a large, real improvement over the from-scratch none arm (+661
IQM, P=0.970, 3/5 vs 1/5 clears). But it is NOT reliable in the absolute sense
the port decision requires. Two of five seeds (s2 887.7, s3 669.9) fall well
below the 930.27 bar, and the curriculum IQM 95% CI [742.5, 1730.1] straddles
the bar (lower bound 742.5 < 930.27). The earlier two-seed interim (s0, s1 both
~1700) overstated reliability: it was a lucky pair. Porting a recipe that fails
40% of seeds to the expensive Godot 5M eval-of-record would likely reproduce the
coin-flip there. HOLD the port. (The verdict gate in the harness is
P(IQM)>=0.975; 0.970 misses it, but the load-bearing reason to hold is the 2/5
sub-bar seeds and the bar-straddling CI, not the 0.005 threshold gap.)

## Root-cause finding: a shared constant-collapse attractor

Self-audit surfaced that several per-seed length arrays are BYTE-IDENTICAL across
genuinely different trained models: none-s3, curr-s3, shaped-s0, shaped-s1 all
produce the exact same 30 episode lengths (mean 669.9), and shaped-s3 differs by
one step (670.0). Verified against `reliability_eval_cache.json` (element-wise
equality True). This is not a cache bug; it is a real failure mode: from-scratch
PPO on Signal Dodge converges, on a subset of seeds, to ONE specific near-constant
policy that yields ~670 on the held-out block. So the wall is VARIANCE: some
seeds escape the basin, some do not. The curriculum raises the escape rate from
1/5 to 3/5 but does not guarantee escape. HIGH.

## Mechanism: NOT entropy collapse (hypothesis tested and REFUTED)

The first-pass guess (premature entropy collapse, fix by raising ent_coef) was
FALSIFIED by a direct probe. Loaded the trained policies and measured mean policy
entropy over a 2000-state batch (nats, max 1.099): good seeds s0 0.552, s4 0.669;
failing seeds s2 0.639, s3 0.756. The WORST performer (s3, 670) has the HIGHEST
entropy of all. The failing policies are not more deterministic than the winners,
so an entropy bonus targets the wrong mechanism. This is why introspection is not
verification: the "collapse" read came from skewed action fractions, but skewed
argmax-in-rollout does not imply low per-state entropy. Corrected picture:
multi-modal convergence. Seeds settle into different policy basins and only some
basins dodge competently (winners go L-heavy and diverse; s2 went R-heavy, s3
stayed spread and never commits). It is an optimization / credit-assignment
variance problem. HIGH (anchor: probe this session; entropy numbers above).

found-art (search "PPO high seed variance reduce reliability, some seeds converge
good others fail"): the reliability wall for sparse long-horizon PPO is
value-estimation variance, converged across independent sources. arXiv 2301.05104:
sparse reward -> the critic never gets good value estimates for the rare good
states -> high-variance policy training. arXiv 2311.02129: reports the exact
"dichotomous convergence" (a performant group and a failed group of seeds).
arXiv 2111.04504: the lock-in mechanism, early noisy advantages boost one action
and it runs away. This recipe runs gamma 0.999 (effective horizon ~1000) on an
1800-step survival task, so the critic must regress near-undiscounted survival
~1000 steps out (VecNormalize also normalizes returns with gamma 0.999), a large
early-variance source. Verdict ADAPT: cut the discount, not the exploration.

## Next: gamma-0.99 variance-reduction arm (IN FLIGHT)

One-knob change on the same curriculum scaffold: `--gamma 0.99` (effective horizon
~1000 -> ~100 steps), which is plenty for a reactive dodging task and sharply
lowers value-target variance. Not a twice-failed lever (those were CMA-ES,
CMA-MAE, elite-BC, budget 5M, NoisyNet, PBRS reward geometry); the discount is
none of them. 5-seed arm `sd_fast_m21curr_g99_s{0..4}_5M` launched detached this
session (chain log `runs\sd_fast\curr_g99_chain.log`), gamma 0.99, everything
else m21 + curriculum verbatim. Judge with `tools\sd_fast_reliability.py` (the
g99 arm is wired in, guarded on all 5 models being on disk; port gate = clears
5/5 AND IQM CI lower bound > 930.27). If g99 clears reliably, port to a Godot 5M
eval-of-record. If it lifts but is still short, next un-tried knobs are
`anneal_frac` 0.7 -> 0.9 or higher `n_init_max` (hold the scaffold longer for
slow seeds). The retired ent_coef idea is NOT the next move; the probe refuted it.

## RESULT: gamma-0.99 arm CLEARS the port gate (2026-07-04)

All 5 seeds landed and `tools\sd_fast_reliability.py` ran on the full 15+5
model set. Held-out (greedy, seeds 5000-5029) per-seed means for the g99 arm:
[1671.1, 1800.0, 1778.6, 1800.0, 1800.0]. rliable IQM 1792.9, 95% CI
[1707.0, 1800.0], clears930 5/5, poolMean 1769.9. Note the CI ceiling 1800.0 is
the episode cap: four of five seeds saturate survival.

Gate: clears 5/5 AND IQM CI lower bound (1707.0) > BAR (930.27). BOTH true.
Verdict printed: PORT.

Comparisons:
- g99 vs none: IQM +1059.3, P(IQM)=1.000, POI 0.937.
- g99 vs curr: IQM +398.3, P(IQM)=0.995, POI 0.711. The discount cut, on the
  same curriculum scaffold, is what converted the curriculum arm from
  not-port-reliable (3/5, CI straddling the bar) to 5/5 saturating.

Self-audit (evidence-anchored): seed-0's gate held-out mean 1671.1 matches
seed-0's independently-written training summary mean_len 1671.13 (chain log),
so the reload-eval cache is not stale. Gate clears/CI logic matches source
line 192 read pre-run. First reliable from-scratch clear in project history;
the wall was critic-variance under the near-undiscounted (gamma 0.999,
horizon ~1000) return target on an 1800-step survival task, not exploration
or policy capacity. Confidence HIGH.

## Next: Godot 5M eval-of-record (discount-first port)

The recipe that cleared is curriculum + gamma 0.99. BUT the start-state
curriculum is not injectable into the real Godot env without GDScript +
protocol work: `godot_env.reset()` passes only a seed through the transport;
`active_hazard_count_above_player` is read-only telemetry, no hazard-injection
seam. Decision (technical, mine): port DISCOUNT-FIRST. Run from-scratch on
Godot with gamma 0.99, m21 recipe, NO curriculum. Rationale: the discount is
the load-bearing lever this session; it needs zero Godot-side changes; it
isolates whether the discount fix alone transfers to the real game. If it
clears the 930.27 bar reliably, the curriculum GDScript work is unnecessary.
If it lifts but is short, THEN build the curriculum injection path (GDScript
pre-spawn N hazards at reset + protocol option + `godot_env.reset(options=)`)
and port the full recipe.

Port mechanics (adapt `configs\rl\signal_dodge_ppo_h3.yaml`, do NOT write
blind): new config `signal_dodge_ppo_g99.yaml` with n_envs 8,
total_timesteps 5_000_000, and hyperparams gamma 0.99, n_steps 512,
batch_size 512, n_epochs 10, ent_coef 0.01, clip_range 0.2, learning_rate
3e-4, gae_lambda 0.95, policy_kwargs.net_arch [64,64]. Run via
`python -m sight_agent.rl.train --config ...`.

TWO faithfulness checks to run BEFORE launching the 5M run (both cheap, both
unverified as of this handoff):
1. Does `train.py._build_train_env` wrap VecNormalize with a configurable
   gamma? The m21/g99 recipe normalizes BOTH obs and reward with the training
   gamma (0.99). If the Godot path does not apply VecNormalize(gamma=0.99),
   the port is NOT faithful. Read `src\sight_agent\rl\train.py` lines ~157-204
   and `factories.py`.
2. Godot throughput. Launch the h3 smoke (1024 steps) or a short probe to get
   steps/sec, then size the 5M run. sd_fast did 6551 steps/s; Godot steps a
   real subprocess and will be far slower. This decides detached-run duration.
3. Eval-of-record harness: confirm the greedy held-out eval on seeds 5000-5029
   vs 930.27 exists for Godot (`h5_baseline_cli` / `rl.evaluate`) or wire it.

## Godot port: infra built and validated, 1M discount-first probe in flight (2026-07-05)

The three pre-launch checks are resolved:
1. VecNormalize is NOT in the train.py/factories pipeline (grep: zero hits).
   So the recipe cannot be ported faithfully through `train.py`. Decision
   (ADAPT, not BUILD-into-train.py): a dedicated Godot trainer that reuses the
   sd_fast_ppo recipe structure with the env swapped. train.py would need
   VecNormalize retrofit affecting all H2-H5 configs; the dedicated trainer
   gives byte-level recipe fidelity and leaves the shared pipeline untouched.
2. Throughput measured live. h3 smoke: ~30 steps/s single-env windowed. The
   g99 trainer smoke (2 envs, headless): 70.9 steps/s. The 1M run (8 envs,
   headless): ~119 steps/s aggregate. DummyVecEnv steps serially so n_envs
   barely changes aggregate rate. Sizing: 1M ~2.3h, 5M ~12-20h/seed, a 5-seed
   5M arm ~3-4 days. Anchors: M2.1 record (59.8 steps/s) + live logs.
3. No faithful Godot eval-of-record existed. `rl.evaluate` never applies
   VecNormalize and runs all episodes under one global seed, not the held-out
   5000-5029 protocol. Wired into the new trainer's `evaluate()` instead.

Godot binary: `Godot_v4.6.2-stable_win64.exe` under the WinGet packages path;
run via `SIGHT_GODOT_EXE`. `where godot` and the env var are otherwise unset.

Factory constraint: `make_env` rejects n_envs>1 for the Godot env
("vectorized parallel Godot envs explicitly out of scope"), so the 8 training
envs are constructed directly as a DummyVecEnv of 8 GodotSignalDodgeEnv, each
with its own kernel-allocated TCP port (the sanctioned direct-construction
path). Headless.

New durable tool `tools\sd_godot_ppo_g99.py`, SMOKE-VALIDATED end to end
(2 envs, 2000 steps, 3 eval seeds): multi-process Godot construction with
distinct ports, VecNormalize wrap+save, and greedy held-out-seed eval all
exercised. Smoke eval mean_len 323.0 (untrained-ish at 2000 steps, below bar,
diverse actions 0.12/0.77/0.11, explained_variance 0.2891). Confidence HIGH
that the infra is correct; the recipe's clearing behavior at real budget is
still UNKNOWN on Godot.

First real run IN FLIGHT: `g99_godot_1M_s0`, single seed, gamma 0.99, 1M steps,
no curriculum, 8 envs headless (pid 34116, log
`runs\sd_godot\g99_godot_1M_s0.log`). Staged on purpose: 1M is the controlled
contrast against M2.1 (gamma 0.999 / 1M / Godot -> IQM 418). If gamma 0.99
lifts clearly above 418 toward/past 930.27, scale to 5M then a 5-seed reliable
arm. If flat, the discount alone does not transfer within budget on Godot, and
the next lever is the curriculum injection (GDScript pre-spawn + protocol
option) or accepting imitation as the standing solution (a Jeff scope call).


## RESULT: discount-only port does NOT transfer to Godot; seed-curriculum ruled out (2026-07-05)

`g99_godot_1M_s0` landed (train 9871s, 101.3 steps/s). Held-out greedy eval
(seeds 5000-5029): mean_len 491.5, IQM ~476 (trim 0.25 of the 30 lengths),
std 219.5, min 183, max 933 (a single seed grazed the bar once), beats_bar
false, action_fracs 0.18/0.71/0.11. Final explained_variance -2.03 (critic
worse than predicting the mean). Anchor: `runs\sd_godot\g99_godot_1M_s0_summary.json`.

Judgement vs the controlled contrast M2.1 (gamma 0.999 / 1M / Godot -> IQM 418):
gamma 0.99 lifts the IQM ~418 -> ~476 (+14%) but does NOT approach 930.27, and
the critic never learns a usable value function (negative EV). This is the messy
middle, not the pre-scripted clean-lift. Decisive read: the recipe that cleared
the replica 5/5 was curriculum + gamma 0.99; the discount was the LAST knob on a
curriculum scaffold, not a standalone fix. The Godot port dropped the curriculum
(no injection seam) and kept only the discount. Two Godot from-scratch-no-curriculum
attempts now both fail (0.999 -> 418, 0.99 -> 476). Per the contract (method fails
twice, change the method; do not retry harder), the 5M discount-only run is NOT
launched: it retries a twice-failed, curriculum-omitting recipe with 5x budget.
Confidence HIGH.

Method change = put the load-bearing curriculum onto Godot. Before committing to
the GDScript injection, a Python-only shortcut was tested and KILLED: a
seed-selection curriculum (filter reset seeds by natural above-player hazard
density). Probe `runs\sd_godot\_probe_seedcurr.py` (shaped-telemetry env, 50
seeds, first 10 frames, action=stay): `active_hazard_count_above_player` = 0 for
ALL 50 seeds at every early frame. Clean Godot resets have zero early
above-player density, so seed selection cannot supply the 2-6 hazards the replica
curriculum injected. Seed-curriculum is impossible; the injection seam is
required. Confidence HIGH (direct measurement). Also confirmed: `godot_env.reset`
already accepts `options=` at the gym layer but does not plumb it into the
transport, and the Godot side has no injection handler, so the GDScript +
protocol work stands.

## NEXT: build the Godot start-state curriculum injection (pre-registered "if short" branch)

Port the FULL proven recipe (curriculum + gamma 0.99) to Godot. Build the seam:
1. GDScript (in `games\signal-dodge`): at reset, read a curriculum hazard count
   from the reset message; pre-spawn N hazards above the player (headroom ~100px,
   no reset collision, no insta-death), mirroring `CurriculumSDF` on the replica.
2. Protocol: carry `curriculum_n_init` (or similar) through the H3 reset wire
   message; keep the clean-start path (N=0) byte-identical so eval is untouched.
3. Python: plumb `godot_env.reset(options={...})` -> transport.reset -> wire;
   add an `AnnealCurriculum` callback annealing n_init_max=6 -> 0 over anneal_frac
   0.7, exactly as the replica. Eval stays greedy held-out 5000-5029 vs 930.27 on
   the clean env.
4. Smoke-validate like the replica: clean-start (N=0) obs byte-identical to the
   current env; N=6 injects 6 hazards with no reset collision; live count mutation
   reflected in reset; short train+eval end to end. THEN launch the Godot arm.
Downstream Jeff scope call (only if the curriculum-injection port ALSO fails to
clear on Godot): accept imitation (BC 1737.3, PPO-ft 1710.5) as the standing
solution vs keep pursuing from-scratch. Not triggered yet.
