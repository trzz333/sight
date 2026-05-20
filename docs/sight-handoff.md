# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.2 env-dynamics evidence pack landed; classification ENV-PASS; the K3.5c/K5.1 weight-invariant constant-action collapse is downstream of the env, in the learning pipeline)

**Last commit:** `e5c490a` K5.2 env-dynamics evidence pack -> ENV-PASS

**Current task:** K5.2 evidence on disk at `docs/k5-2-env-dynamics-sanity-evidence.md` (346 lines) and pushed. Probe at `tools/k5_2_env_dynamics_probe.py` (1078 lines) drives the production `GodotSignalDodgeEnv` + `GodotH3Transport` path with scripted policies and reads NDJSON streams plus wire payloads. Eight layers all PASS: Layer 0 collision-propagation preflight (sticky terminal flag from `_on_player_died` propagates through next step reply, step-after-terminal rejects, reset clears); Layer 1 action timing (240 scripted steps, 1 `h3_step` event per Python step, frame +1 monotonic, seq match, wire 0/1/2 -> mapped -1/0/+1); Layer 2 player kinematics (5 px/step on action != 0, left clamp x=16 observed on 12 steps); Layer 3 hazard kinematics (3.333 px/step, spawn every 30 frames at y=-24); Layer 4a state obs freshness (obs[0] = normalized post-step player_x, obs[1] = current applied mapped action, zero mismatches over 240 steps); Layer 4b pixel obs freshness (0 dynamics divergences vs state mode, first-left and first-right same-step shifts confirmed, action-transition hash predicate clean); Layer 5 observation_space shape (10,) and (1,84,84) matches K5.1 config; Layer 6 hazard-reactive 1-step geometry oracle survives 1762.8 mean frames per episode across 10 seeds with collision_rate 0.10, vs best constant `constant_left` at 845.7 (collision_rate 0.90), delta 917.1 exceeds materiality threshold 84.6 by an order of magnitude. The shaped-reward-greedy oracle on the K5.1 alpha=0.30 surface emits stay 81.8% and reaches 1462.8 mean ep len, confirming the shaped surface biases toward center-stay but this does not explain the K3.5c unshaped constant-left collapse.

**Next action:** Grok phase-gate sanity check on the K5.2 ENV-PASS evidence pack per the charter's ENV-PASS routing branch. Relay packet template prepared earlier in the session with the K5.2 scope inlined and the three questions revised to target the deterministic-argmax fixed point hypothesis (Grok itself proposed in the prior RED reply that stochastic-eval re-evaluation of the K5.1 checkpoint is the falsifying experiment). Jeff pastes the packet to Grok, then pastes Grok's reply back to a fresh Claude session for veto-on-evidence review under the Grok workspace instructions that landed this session.

**Blockers:** None requiring Jeff.

**Notes:**

- ENV-PASS rules out env mechanics AND task geometry as causes of the K3.5c/K5.1 collapse. The pathology is downstream of the env, in PPO + CnnPolicy single-frame (1,84,84) + deterministic-argmax + 10k-step budget.
- `tools/h5_stochastic_eval.py` already exists in the repo from a prior phase and is the candidate harness for the deterministic-argmax falsification test that GPT or Grok is likely to call next.
- Layer 4b's original "no consecutive duplicate hashes" predicate was reframed mid-session to `predicate_action_transition_hashes_differ`. 4 of 5 raw duplicates were pre-spawn stay frames where 84x84 nearest-neighbor downsample of a visually static world genuinely produces identical hashes; the 5th was a mid-stride low-motion step where 5 world-px maps to 0.58 obs-px. The transition-based predicate is the correct freshness instrument and is clean.
- Layer 6 wall-time was ~11 minutes for 50 episodes; standard MCP 4-min timeout pattern via `.bat` + sentinel under `C:\Users\maste\AppData\Local\Temp\` worked cleanly (see `Reproduction` section in the evidence doc).
- Grok workspace instructions for the Sight project were composed and validated GREEN earlier in the session (3902 chars under the 4000 cap that xAI imposed in March 2026). Persist them in the Grok workspace before sending the K5.2 packet so the calibration, sycophancy guards, refusal posture, and disagreement-first format are active.
