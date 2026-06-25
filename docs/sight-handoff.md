# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase L (d3rlpy offline-RL). K7 mixed-quality offline pipeline built and smoke-validated end to end.

**Last commit:** `5e87bee` K7 offline-RL pipeline: mixed-quality collector + d3rlpy DiscreteCQL/filtered-BC trainer.

**Current task:** The offline-RL pipeline is built and runs end to end on real Godot rollouts. `tools\collect_offline_dataset.py` (global interp) rolls a mixed behavior set (uniform-random + QR-DQN off_s0 at 50k and 200k + BC) through one persistent GodotSignalDodgeEnv and logs full transitions to `runs\phase_k\k7_offline\offline_dataset.npz`, then subprocesses `tools\d3rlpy_offline_train.py` inside `.venv-d3rlpy` (MDPDataset terminals=collision timeouts=truncation; DiscreteCQL on the full mix + filtered DiscreteBC over top-return whole episodes; both exported via save_policy to TorchScript), then loads those TorchScript policies back in the global interp for greedy in-env eval vs bar 930.27 on held-out seeds. Smoke run (deliberately tiny: 10 eps, 2000 train steps each, 3 eval seeds) is GREEN: dataset 7722 transitions / 10 episodes, returns 182 to 1800, terminal_count 9 / timeout 1 (mixed quality confirmed). Smoke eval: DiscreteCQL mean 483.0 FAIL, degenerate stay-only (pooled action fractions [0.0, 1.0, 0.0]); filtered-BC mean 1052.33 PASS (delta_vs_bar +122.06), a real moving policy (fractions [0.205, 0.418, 0.377]) but high variance (1/3 seeds timeout, 2/3 collide). These are smoke-grade pipe-validation numbers, NOT a mission claim: CQL is starved at 2000 steps on 7722 transitions and filtered-BC clearing the bar at this scale just restates that BC-quality data clears the bar. CQL trained clean (td_loss 0.088, conservative_loss 0.53, no NaN).

**Next action:** Run a real-scale K7: scale collection (spread QR-DQN sources across more stages and seeds including the on_* runs and seeds 1-4, raise random/qrdqn/bc episode counts for a larger mixed set), raise `--cql-steps` and `--bc-steps` to a real budget (start 100000), eval on the full held-out seeds 1000-1009, then judge whether DiscreteCQL beats filtered-BC and the bar. Same two scripts, just larger flags. Command shape: `python tools\collect_offline_dataset.py --random-eps N --qrdqn-eps N --qrdqn-stages "k6_dyn_off_s0:50000,k6_dyn_off_s0:200000,k6_dyn_on_s0:200000,..." --bc-eps N --cql-steps 100000 --bc-steps 100000 --eval-seeds 1000-1009`.

**Blockers:** None requiring Jeff.

**Notes:**

- Interpreter split is load-bearing: d3rlpy pins gymnasium 1.0.0, the SB3/Godot stack runs gymnasium 1.2.3. Collection + in-env eval run in the GLOBAL interp; d3rlpy training runs only in `.venv-d3rlpy` via subprocess; the trained policy crosses back as a TorchScript file (`cql_policy.pt`, `fbc_policy.pt`) loaded with torch.jit in the global interp. Never import d3rlpy globally.
- Smoke-grade caveat stands: 2000 steps / 10 eps / 3 eval seeds is a pipe check, not evidence. Do not quote the 1052.33 / 483.0 numbers as a result. The real-scale run is the one that produces a claimable verdict.
- Reused in-repo prior art wholesale (found-art ADOPT): `_build_env` (k5_2), QR-DQN loader + VecNormalize norm (k6_dyn_eval_inenv), BC loader (k5_6_bc_eval_inenv). No env/loader code was rewritten.
- All `runs\phase_k\k7_offline\*` artifacts (npz, TorchScript policies, d3rlpy_logs, godot logs, reports) are gitignored. Only the two tools are tracked. Collection of the smoke took ~132s of Godot rollouts; real-scale will be much longer, use the .bat + WMI detached pattern if it runs long.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
