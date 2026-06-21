"""Save/load round-trip test for NoisyQRDQNPolicy (K5.8 eval-path guard).

Confirms a QRDQN built with NoisyQRDQNPolicy can be saved and reloaded with
QRDQN.load, and that the reloaded model predicts deterministically in eval.
Dummy Box(10)/Discrete(3); no Godot. Safe alongside the detached 200k.
"""
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import QRDQN

from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy


class _DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1.0, 1.0, (10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 1.0, False, False, {}


def main() -> int:
    env = _DummyEnv()
    model = QRDQN(
        NoisyQRDQNPolicy, env, device="cpu", seed=0,
        learning_starts=10, buffer_size=1000, batch_size=8,
        exploration_fraction=0.0, exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        policy_kwargs=dict(sigma_0=0.5, n_quantiles=50, net_arch=[64, 64]),
    )
    model.learn(total_timesteps=40, progress_bar=False)
    x = np.zeros((10,), dtype=np.float32)
    before, _ = model.predict(x, deterministic=True)

    with tempfile.TemporaryDirectory() as d:
        zp = str(Path(d) / "m.zip")
        model.save(zp)
        # Mirror the eval script's load call exactly.
        m2 = QRDQN.load(zp, device="cpu",
                        custom_objects={"policy_class": NoisyQRDQNPolicy})
        m2.policy.set_training_mode(False)
        after, _ = m2.predict(x, deterministic=True)
        # Determinism across two reloaded predicts.
        after2, _ = m2.predict(x, deterministic=True)

    ok_reload = int(before) == int(after)
    ok_det = int(after) == int(after2)
    assert ok_det, f"reloaded predict nondeterministic: {after} vs {after2}"
    print(f"OK reload pred_before={int(before)} pred_after={int(after)} "
          f"match={ok_reload} det={ok_det}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
