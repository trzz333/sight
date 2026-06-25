# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase L. d3rlpy offline-RL pivot, found-art scoping complete (no training code yet). Phase K verdicts (N=10 from-scratch; K6 self-supervised on/off) both FINAL negative.

**Last commit:** `<hash>` d3rlpy offline-RL pivot: found-art dig + scipy.stats.bootstrap adoption.

**Current task:** Offline-RL pivot is scoped and the toolchain is validated, but no training has run. d3rlpy 2.8.1 imports on cp314 inside isolated venv `C:\Projects\Sight\.venv-d3rlpy` (built --system-site-packages so it inherits torch 2.11 and leaves the global gymnasium 1.2.3 / SB3 stack untouched; legacy gym 0.26.2 builds from sdist). Applicable discrete algos confirmed present: DiscreteCQL, DiscreteBCQ, DiscreteBC; DiscreteIQL is absent (IQL is continuous-only) so IQL is dropped for this discrete task. d3rlpy MDPDataset takes (observations, actions, rewards, terminals, timeouts) with collision-vs-truncation handled natively, so MDP assembly is ADOPT not BUILD. Blocking finding: neither existing artifact is a usable mixed-quality offline dataset. The BC demo npz `runs\phase_k\k5_6_bc\dataset_2000_2035.npz` is all-expert (36 episodes all exactly 1800-step caps, zero collisions, uniform +1 reward), degenerate for value learning so offline RL on it reduces to BC (already 1737.3). The K6/QR-DQN ndjson logs carry reward/terminated/truncated and real crashes but NOT the full 10-dim observation (godot.ndjson logs only player_x/player_y; python.ndjson logs no obs/action), so a faithful dataset cannot be rebuilt from them. Env reward verified from godot_env.py as sparse survival (+1.0 per step, 0.0 on the collision step). Stats found-art also landed this session (scipy.stats.bootstrap; see notes).

**Next action:** Build `tools\collect_offline_dataset.py`: a thin rollout-and-log harness around `GodotSignalDodgeEnv` that rolls a mix of behavior policies (uniform-random, the saved QR-DQN checkpoints under `runs\phase_k\k6_dyn_*\ckpts`, and the BC policy) and logs full transitions (obs, action, reward, terminated, truncated) to an npz. Then load into a d3rlpy MDPDataset (terminals=collision, timeouts=truncation) and smoke-train DiscreteCQL in `.venv-d3rlpy`. Run everything in the venv, keep the global SB3 env clean. Include a filtered-BC comparator, not just raw BC.

**Blockers:** None requiring Jeff.

**Notes:**

- Stats found-art DONE: hand-rolled boot_ci replaced with scipy.stats.bootstrap in `tools\k5_8_reliability_report.py`, numbers bit-identical, K6 re-verified against on-disk eval reports (OFF IQM 660.4 CI [516.9,1019.0] frac_above_bar 0.20; ON IQM 626.1 CI [570.4,793.7] frac_above_bar 0.00; both below bar 930.27, no PASS). rliable NOT adopted: arch 8.0.0 has a cp314 win wheel but rliable 1.2.0 pins arch<8.0, dragging arch 7.2.0 sdist that needs MSVC.
- d3rlpy isolated in `.venv-d3rlpy` (gitignored via `.venv*/`). Never `pip install d3rlpy` into the global interpreter: its gymnasium==1.0.0 pin would downgrade the global 1.2.3 under the SB3 reproducibility stack.
- Offline-RL expectation (lit, BAIR rl-or-bc): value-based offline RL can beat the behavior policy via stitching, but on replay-style survival data the edge over a strong filtered-BC baseline is modest. This is a legitimate experiment, not a guaranteed BC-beater. CQL primary (robust to suboptimal data), BCQ secondary (degrades on very suboptimal data).
- BC 1737.3 and PPO-finetune 1710.5 still MEDIUM, not re-verified this session. Re-confirm from eval artifacts before any external/portfolio use. Two clean honest-negative results stand (N=10 from-scratch; K6 auxiliary) with BC as the reliable above-bar policy.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1, verified SET this session. No long run pending now. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
