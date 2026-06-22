"""Learnable MLP encoder as an SB3 features extractor (K6).

Why this exists
---------------
In state mode SB3 defaults to FlattenExtractor, which has NO parameters. An
auxiliary self-supervised prediction loss bolted onto that would share no trunk
with the value head and so could not shape the policy's representation: it would
just train a useless parallel dynamics model. DynEncoder inserts a small
learnable obs->latent MLP so the NoisyNet quantile head AND the K6 next-state
prediction head sit on the SAME latent. That is what lets the prediction loss do
representation learning (the SPR idea, Schwarzer et al. 2021, adapted to a
low-dim STATE env: predict the next state directly, since there is no
pixel-reconstruction cost and therefore no need for an EMA latent-target trick).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DynEncoder(BaseFeaturesExtractor):
    """obs -> latent MLP, shared by the quantile head and the dynamics head."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 64,
                 hidden: int = 128):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, features_dim), nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
