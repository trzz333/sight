# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase L (d3rlpy offline-RL). DiscreteCQL FINAL NEGATIVE. Value-RL thread CLOSED.

**Last commit:** `7f1d21d` K7.1 CQL-config retry: DiscreteCQL FINAL NEGATIVE for Signal Dodge.

**Current task:** K7.1 ran end to end (HIGH, `runs\phase_k\k7_offline\real\k7_eval_report.json`, train/eval exit 0/0). Changed the method per the stop-condition plan: DiscreteCQL `n_critics` 1->3 and conservative `alpha` 1.0->0.5, trained 100k CQL + 100k filtered-BC on the existing 80470-transition npz (no recollect), greedy in-env eval seeds 1000-1009 vs bar 930.27. Result: CQL mean 606.0 FAIL (delta -324.27), pooled action fractions [0.0, 1.0, 0.0], collision on all 10 seeds. Identical stay-only collapse to the K7 default run (also 606.0 to the decimal); per-seed lengths equal a constant-Stay policy and sit below best_constant 845.7. Found-art hypothesis FALSIFIED: lowering alpha moved the collapse by zero, so the conservative coefficient is not the cause in this range. Stop condition met -> DiscreteCQL declared FINAL NEGATIVE, Phase L value-RL thread closed, no third retry. filtered-BC mean 1299.7 PASS (delta +369.43), real moving policy [0.212, 0.307, 0.481], 5/10 timeout.

**Next action:** Value-RL is closed; the open fork is Jeff-owned DIRECTION: does Phase L continue on the imitation/BC side (the only thing that clears the bar), or does the project pivot? The substantive offline-RL claim is now in hand: value-based offline RL (DiscreteCQL) collapses to the modal action on the full mixed dataset regardless of conservative tuning, while imitation on filtered good data clears the bar. The lateral finding for any continuation: the working lever is trajectory filtering / data quality, not the CQL conservative coefficient. Await Jeff direction before opening a new thread.

**Blockers:** Continue-vs-pivot is a Jeff-owned direction call (not a technical blocker). Nothing else blocking.

**Notes:**

- Interpreter split load-bearing: collection + in-env eval run in the GLOBAL interp (SB3, gymnasium 1.2.3, Godot); d3rlpy training runs only in `.venv-d3rlpy` via subprocess; policy crosses back as TorchScript. Never import d3rlpy globally.
- scipy 1.18.0 now confirmed resolving from `.venv-d3rlpy\Lib\site-packages` even under a stripped env (APPDATA/USERPROFILE blanked, PYTHONNOUSERSITE=1). The WMI-detached ModuleNotFoundError blocker is fixed; keep scipy in the venv, never user-site.
- DiscreteCQL collapse is structural, not a tuning artifact: it equals a constant-Stay policy at both alpha=1.0/n_critics=1 and alpha=0.5/n_critics=3. Discrete-CQL penalty == BC-NLL toward the modal dataset action (Stay), and the full mixed set is Stay-dominant. Do not reopen with more CQL knobs.
- filtered-BC PASS (1299.7) trains on top-25% episodes (return_threshold 1653, 24 eps / 42937 transitions = the 20 BC eps + strong QR-DQN). It is imitation on good data, still below pure BC's 1737.3, not a value-RL win.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
