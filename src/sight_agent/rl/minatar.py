"""MinAtar adoption layer for Sight.

ADOPT mainline `minatar` 1.0.15 (PyPI), which already targets gymnasium and
registers MinAtar/<Game>-v0 (full action set) and -v1 (minimal action set).

Provides:
- register(): idempotent env registration.
- ChannelFirstFloat: wraps (H,W,C) bool obs into (C,H,W) float32 in [0,1],
  so SB3 sees an unambiguous non-image Box and applies no auto image transforms.
- MinAtarCNN: the Young & Tian (2019) small conv net (one 3x3x16 conv + ReLU +
  FC), the standard MinAtar architecture reproduced in qlan3/gym-games. ADAPTed
  into an SB3 BaseFeaturesExtractor.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    from minatar.gym import register_envs
    try:
        register_envs()
    except gym.error.Error:
        pass  # already registered in this interpreter
    _REGISTERED = True


class ChannelFirstFloat(gym.ObservationWrapper):
    """(H,W,C) bool -> (C,H,W) float32 in [0,1]."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, obs):
        return np.transpose(np.asarray(obs, dtype=np.float32), (2, 0, 1))


def make_env(game="MinAtar/Breakout-v1", seed=None):
    register()
    env = gym.make(game)
    env = ChannelFirstFloat(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


# --- SB3 feature extractor (imported lazily to keep this module import-cheap) ---
def build_extractor_cls():
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class MinAtarCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            n_ch = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_ch, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

        def forward(self, x):
            return self.linear(self.cnn(x))

    return MinAtarCNN
