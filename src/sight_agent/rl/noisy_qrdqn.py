"""NoisyNet QR-DQN policy adaptation for K5.8 (state mode, Signal Dodge).

Why this exists
---------------
K5.1-K5.7 established a single cross-method failure in this env+pipeline:
the policy collapses into a single-direction basin (PPO -> constant action;
vanilla DQN -> right-only; QR-DQN+n-step -> left-only). The env is provably
left/right SYMMETRIC (player starts centered, symmetric speed and bounds,
uniform hazard spawn, antisymmetric observation), so the basin is NOT an env
asymmetry. It is an exploration pathology: epsilon-greedy is a "dithering"
strategy (Osband et al. 2016, Bootstrapped DQN) that injects only local,
temporally-incoherent jitter and cannot escape a directional basin once the
argmax commits. Per the project contract, a method that fails twice is
replaced, not retried harder -- and raising epsilon is "retry epsilon-greedy
harder." NoisyNet (Fortunato et al. 2017) REPLACES the exploration MECHANISM:
learnable per-weight Gaussian noise drives state-dependent, temporally-
coherent exploration that the network can anneal itself where it is confident
and keep where it is not.

Found-art verdict: ADAPT. sb3-contrib 2.8.0 ships QRDQN but no NoisyNet;
CleanRL ships rainbow_atari.py (CNN only, no MLP classic-control NoisyNet).
So we ADAPT SB3's QuantileNetwork by swapping its create_mlp linear stack for
NoisyLinear layers, inheriting the entire QR-DQN training loop, n-step target,
quantile Huber loss, and target network unchanged. Epsilon-greedy is disabled
(final_eps=0) so exploration comes purely from weight noise.

NoisyLinear: factorised Gaussian noise per Fortunato et al. 2017 section 3.2.
sigma_0 default 0.5. Noise is resampled every forward in train() mode and
zeroed in eval() mode so greedy eval is deterministic (the K5 eval contract
calls model.predict(deterministic=True), which puts the net in eval()).
"""

from __future__ import annotations

import math
from typing import Any

import torch as th
from torch import nn

from sb3_contrib.qrdqn.policies import QuantileNetwork, QRDQNPolicy


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet linear layer (Fortunato et al. 2017).

    y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b)

    Factorised noise: eps_w[i,j] = f(eps_in[j]) * f(eps_out[i]),
    eps_b[i] = f(eps_out[i]), with f(x) = sign(x) * sqrt(|x|). In train()
    mode noise is resampled on every forward; in eval() mode noise is zero
    so the layer is a plain affine map mu_w x + mu_b (deterministic greedy).
    """

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_0 = float(sigma_0)

        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))

        # Noise buffers are not parameters; registered so .to(device) moves them.
        self.register_buffer("weight_eps", th.zeros(out_features, in_features))
        self.register_buffer("bias_eps", th.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        # Fortunato 2017 factorised init: mu ~ U[-1/sqrt(p), 1/sqrt(p)],
        # sigma = sigma_0 / sqrt(p), p = in_features.
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        sigma_init = self.sigma_0 / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    @staticmethod
    def _f(x: th.Tensor) -> th.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._f(th.randn(self.in_features, device=self.weight_mu.device))
        eps_out = self._f(th.randn(self.out_features, device=self.weight_mu.device))
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


def _noisy_mlp(input_dim: int, output_dim: int, net_arch: list[int],
               activation_fn: type[nn.Module], sigma_0: float) -> list[nn.Module]:
    """Mirror of SB3 create_mlp but with NoisyLinear in place of nn.Linear."""
    if len(net_arch) > 0:
        layers: list[nn.Module] = [NoisyLinear(input_dim, net_arch[0], sigma_0),
                                   activation_fn()]
    else:
        layers = []
    for i in range(len(net_arch) - 1):
        layers.append(NoisyLinear(net_arch[i], net_arch[i + 1], sigma_0))
        layers.append(activation_fn())
    last_in = net_arch[-1] if len(net_arch) > 0 else input_dim
    layers.append(NoisyLinear(last_in, output_dim, sigma_0))
    return layers


class NoisyQuantileNetwork(QuantileNetwork):
    """QuantileNetwork whose head is a NoisyLinear stack, not create_mlp."""

    def __init__(self, *args: Any, sigma_0: float = 0.5, **kwargs: Any):
        # Pop before super: QuantileNetwork.__init__ builds self.quantile_net
        # via create_mlp; we rebuild it immediately after with noisy layers.
        super().__init__(*args, **kwargs)
        self._sigma_0 = float(sigma_0)
        action_dim = int(self.action_space.n)
        noisy = _noisy_mlp(self.features_dim, action_dim * self.n_quantiles,
                           self.net_arch, self.activation_fn, self._sigma_0)
        self.quantile_net = nn.Sequential(*noisy)

    def reset_noise(self) -> None:
        for m in self.quantile_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class NoisyQRDQNPolicy(QRDQNPolicy):
    """QRDQNPolicy that builds NoisyQuantileNetwork heads.

    sigma_0 is threaded via policy_kwargs. Everything else (target net,
    optimizer, _predict greedy argmax over mean quantiles) is inherited.
    """

    def __init__(self, *args: Any, sigma_0: float = 0.5, **kwargs: Any):
        self._sigma_0 = float(sigma_0)
        super().__init__(*args, **kwargs)

    def make_quantile_net(self) -> NoisyQuantileNetwork:
        net_args = self._update_features_extractor(self.net_args,
                                                   features_extractor=None)
        return NoisyQuantileNetwork(sigma_0=self._sigma_0, **net_args).to(self.device)
