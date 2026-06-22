"""K6: NoisyNet QR-DQN + self-supervised next-state prediction (model-based flavor).

Adds a forward-dynamics auxiliary loss to K5.8's NoisyNet QR-DQN. A prediction
head reads the SHARED DynEncoder latent plus the taken action (one-hot) and
predicts the next observation; its MSE is added (weight dyn_beta) to the
quantile-Huber loss, so gradients from BOTH objectives flow into the shared
encoder. This is the SPR / world-model idea (Schwarzer et al. 2021; Ha &
Schmidhuber 2018; Hafner et al., DreamerV3, Nature 2025) adapted to a low-dim
STATE env: predict the next state directly (no pixel cost, so no EMA
latent-target trick). dyn_beta=0 recovers K5.8 exactly, giving a fair on/off
comparison on the same env, eval harness, and reliability report.

The dynamics head is owned by the algorithm and trained by a second optimizer;
the shared encoder lives in the policy and is updated by the policy optimizer,
so a single total.backward() routes encoder grads through policy.optimizer and
head grads through dyn_optimizer. The head is auxiliary only: eval uses the
greedy quantile head, so the head is not needed at eval time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from torch import nn

from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss


class DynQRDQN(QRDQN):
    def __init__(self, *args: Any, dyn_beta: float = 1.0, dyn_hidden: int = 128,
                 dyn_lr: float = 2.3e-4, **kwargs: Any):
        self.dyn_beta = float(dyn_beta)
        self.dyn_hidden = int(dyn_hidden)
        self.dyn_lr = float(dyn_lr)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_dim = int(np.prod(self.observation_space.shape))
        n_act = int(self.action_space.n)
        feat_dim = int(self.policy.quantile_net.features_extractor.features_dim)
        self.dyn_head = nn.Sequential(
            nn.Linear(feat_dim + n_act, self.dyn_hidden), nn.ReLU(),
            nn.Linear(self.dyn_hidden, obs_dim),
        ).to(self.device)
        self.dyn_optimizer = th.optim.Adam(self.dyn_head.parameters(), lr=self.dyn_lr)
        self._obs_dim = obs_dim
        self._n_act = n_act

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        # shared encoder (online); feeding replay obs directly matches what the
        # quantile head sees (Box preprocess is identity; VecNormalize already
        # applied at sample time via env=self._vec_normalize_env).
        fe = self.policy.quantile_net.features_extractor

        losses: list[float] = []
        aux_losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

            current_quantiles = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
            q_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)

            # auxiliary self-supervised next-state prediction on the shared encoder
            feat = fe(replay_data.observations)
            a_oh = nn.functional.one_hot(replay_data.actions.long().flatten(), self._n_act).float()
            pred_next = self.dyn_head(th.cat([feat, a_oh], dim=1))
            target_next = replay_data.next_observations.reshape(replay_data.next_observations.shape[0], -1)
            aux = nn.functional.mse_loss(pred_next, target_next)

            total = q_loss + self.dyn_beta * aux
            self.policy.optimizer.zero_grad()
            self.dyn_optimizer.zero_grad()
            total.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            self.dyn_optimizer.step()

            losses.append(float(q_loss.item()))
            aux_losses.append(float(aux.item()))

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
        self.logger.record("train/dyn_loss", float(np.mean(aux_losses)))
