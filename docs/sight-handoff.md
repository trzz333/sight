# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (state-observation comparator slice in progress; Phase F frame-stack diagnostic closed as negative result, Phase G NOT triggered)

**Last commit on HEAD:** `e80614f` feat(rl): h5 state-observation comparator config

**Substantive code commit:** `e80614f` feat(rl): h5 state-observation comparator config

**Current task:** H5 state-observation comparator slice in progress. GPT-approved diagnostic-not-selection move to test whether PPO can learn Signal Dodge when perception is removed from the loop (`obs_shape (10,)`, `MlpPolicy`, recipe inherited verbatim from Phase D/F). YAML at `configs/rl/signal_dodge_ppo_h5_state_comparator.yaml` committed. Smoke at 512 timesteps (seed 0) verified clean: `run_start.env_smoke.obs_shape=[10]`, `action_n=3`, loaded model observation space `Box(-1.0, 1.0, (10,), float32)`, policy class `ActorCriticPolicy` (SB3 default for MlpPolicy), `summary.json status=ok`, `config_hash 32d926cea81eb30b8685aeb4ca430c8921d1c34e89bd2dbf5b1345bdb112d90e`. Three-seed 10k sweep and locked eval over seeds 1000-1009 pending.

**Next action:** Run train seeds 1, 2, 3 at 10000 timesteps each against `configs/rl/signal_dodge_ppo_h5_state_comparator.yaml`. After each train run completes, run `h5_baseline_cli --mode full --policies trained_cnn --seeds 1000-1009` against that run's `model.zip`. Then write `docs/h5-state-comparator-evidence.md`, update this handoff, run `pytest tests/rl -q`, commit, push. Success bar is diagnostic-not-selection per GPT plan: pooled collision rate <= 0.80; pooled reward or length >= 756.25 / 757.50 against frame-stack stay-only; pooled reward/length not worse than Phase E 764.87 / 765.80; no best-of-N seed selection. If clears, next lever is pixel-side recipe repair (likely `ent_coef=0.05` under frame-stack). If state PPO also collapses to stay, propose reward/profile charter amendment.

**Blockers:** None operational. Phase G trigger definitional ambiguity from Phase F resolved by GPT this session: "best frame-stack negative" locked as per-metric strongest comparator (highest reward/length negative for reward/length thresholds, lowest-collision negative for collision threshold). Phase F verdict unchanged because it failed under any defensible trigger reading.

**Notes:**

- Eval CLI label `trained_cnn` for the MlpPolicy run is a documented naming mismatch acceptable for this diagnostic only. SB3 loads `model.zip` through the policy class recorded in the archive regardless of the label string. Do not add a code alias unless the existing path fails. Record the mismatch in `docs/h5-state-comparator-evidence.md` when it lands.
- Diagnostic-not-selection. State PPO results do not close H5 (which requires the small CNN policy). They disambiguate which next experiment lever closes H5.
- Phase F seed 3 train hit a transient `GodotTransportError: recv timed out after 5.0s` at step 0 (Godot startup race). Retry from cleared run directory with same seed and `config_hash` completed `status=ok`. If recurs in the state-comparator sweep, add a startup retry loop to `src\sight_agent\rl\godot_transport.py` rather than rely on manual retries.
- Smoke run dir at `runs\rl\signal_dodge_ppo_h5_state_comparator\h5_train_state_comparator_smoke_seed0_512\` is gitignored per `.gitignore:67 runs/`. Smoke artifacts are local-only diagnostic.
- Handoff convention: `Last commit on HEAD` and `Substantive code commit` match at this checkpoint because the latest commit is itself a substantive (non-chore) code commit. A subsequent chore-refresh commit will create a one-commit lag; resume by running `git log --oneline -5` before claiming HEAD.
