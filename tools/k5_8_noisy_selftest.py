"""Self-test for the NoisyNet QR-DQN policy adaptation (K5.8).

Construction + noise-liveness contract, on a dummy Box(10)/Discrete(3) space
so it does NOT touch Godot and is safe to run alongside the detached 200k.

Asserts:
  1. QRDQN builds with NoisyQRDQNPolicy.
  2. TRAIN mode: two forwards on identical obs differ (noise is live).
  3. EVAL mode: two forwards on identical obs are identical (deterministic
     greedy -- required by the K5 eval contract predict(deterministic=True)).
  4. The online quantile head contains NoisyLinear layers (not create_mlp).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import QRDQN

from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy, NoisyLinear


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
    qn = model.policy.quantile_net

    # 4. Head is noisy.
    n_noisy = sum(1 for m in qn.quantile_net.modules() if isinstance(m, NoisyLinear))
    assert n_noisy == 3, f"expected 3 NoisyLinear layers, got {n_noisy}"

    obs = th.zeros(1, 10)

    # 2. TRAIN: noise live -> differing outputs.
    model.policy.set_training_mode(True)
    with th.no_grad():
        a = qn(obs)
        b = qn(obs)
    train_diff = (a - b).abs().max().item()
    assert train_diff > 1e-6, f"train-mode forwards identical (noise dead): {train_diff}"

    # 3. EVAL: deterministic -> identical outputs.
    model.policy.set_training_mode(False)
    with th.no_grad():
        c = qn(obs)
        d = qn(obs)
    eval_diff = (c - d).abs().max().item()
    assert eval_diff == 0.0, f"eval-mode forwards differ (nondeterministic): {eval_diff}"

    # Greedy predict is deterministic across calls in eval.
    x = np.zeros((10,), dtype=np.float32)
    p1, _ = model.predict(x, deterministic=True)
    p2, _ = model.predict(x, deterministic=True)
    assert int(p1) == int(p2), f"greedy predict nondeterministic: {p1} vs {p2}"

    # Short learn() to confirm the train loop runs with noisy layers.
    model.learn(total_timesteps=40, progress_bar=False)

    print(f"OK noisy_layers={n_noisy} train_diff={train_diff:.4f} "
          f"eval_diff={eval_diff} greedy={int(p1)} learn=ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
