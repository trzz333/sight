# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase L (d3rlpy offline-RL). K7 real-scale verdict in.

**Last commit:** `010f6ed` K7 real-scale offline-RL verdict: DiscreteCQL FAIL, filtered-BC PASS.

**Current task:** Real-scale K7 ran end to end and produced a claimable verdict (HIGH, `runs\phase_k\k7_offline\real\k7_eval_report.json`, train/eval exits 0/0). Dataset: 80470 transitions / 98 episodes (30 random + 48 QR-DQN across 8 stages spanning off/on and seeds 0-4 at 50k/100k/200k + 20 BC, returns 182-1800, mean 820). Trained DiscreteCQL and filtered-BC 100000 steps each, greedy in-env eval on held-out seeds 1000-1009 vs bar 930.27. Result: DiscreteCQL mean 606.0 FAIL (delta_vs_bar -324.27), degenerate stay-only with pooled action fractions [0.0, 1.0, 0.0], collision on all 10 seeds. filtered-BC mean 1073.0 PASS (delta_vs_bar +142.73), a real moving policy [0.2, 0.453, 0.347], 2/10 timeout 8/10 collision, but below BC's own 1737.3. Headline: value-based offline RL collapsed to a single action while imitation cleared the bar. CQL collapsed identically at smoke scale, so scaling data 11x and steps 50x did not rescue it. The CQL failure (not the BC pass) is the substantive finding.

**Next action:** Run a single bounded K7.1 CQL-config experiment via `tools\run_k7_traineval.bat` on the existing npz (no recollect). Change the method, do not retry default harder: raise DiscreteCQL `n_critics` from the default 1 (low, collapse-prone for discrete control) to 3 and lower conservative `alpha` from 1.0 to 0.5 in `tools\d3rlpy_offline_train.py` (found-art the discrete-CQL-collapse / n_critics / alpha question first, ADOPT/ADAPT/BUILD). Re-eval on seeds 1000-1009. Stop condition: if CQL still collapses to stay-only [~0,1,~0], declare offline-RL DiscreteCQL a FINAL NEGATIVE for Signal Dodge and close the Phase L value-RL thread; do not open a third CQL retry.

**Blockers:** None requiring Jeff.

**Notes:**

- Interpreter split load-bearing: collection + in-env eval run in the GLOBAL interp (SB3, gymnasium 1.2.3, Godot); d3rlpy training runs only in `.venv-d3rlpy` via subprocess; policy crosses back as TorchScript loaded with torch.jit. Never import d3rlpy globally.
- scipy must live in `.venv-d3rlpy\Lib\site-packages`, NOT user-site (`%APPDATA%\Roaming\Python\Python314\site-packages`). WMI-detached processes run with a stripped env and cannot see user-site, so a user-site scipy crashes detached training with ModuleNotFoundError. Fixed by vendoring scipy + scipy.libs + dist-info into the venv. pip on this machine defaults to user installs; force `PIP_USER=0` or vendor.
- CQL collapse mode is stay-only argmax. Prime suspects for K7.1: `n_critics=1` default and `alpha=1.0` conservative penalty over-pushing argmax toward the modal in-distribution action (stay). Verify with a found-art search before tuning.
- filtered-BC PASS mostly restates that BC-quality data clears the bar (top-25% filter keeps the 20 BC eps + ~4 strong QR-DQN, threshold return 1653). It is not a value-RL win.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
