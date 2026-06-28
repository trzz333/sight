"""C1 probe: inspect the SB3 MlpPolicy actor for ES flatten/load binding.

Builds an ActorCriticPolicy with Signal Dodge spaces (Box(-1,1,(10,)),
Discrete(3)), net_arch pi=[64,64] vf=[64,64], and reports the parameter
layout. ES optimizes ONLY the actor path: mlp_extractor.policy_net +
action_net. The value net is not optimized by ES (no critic).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy


def main() -> int:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    def lr_sched(_progress):
        return 3e-4

    pol = ActorCriticPolicy(
        obs_space, act_space, lr_sched,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    print("=== ALL named_parameters ===")
    total = 0
    for name, p in pol.named_parameters():
        n = p.numel()
        total += n
        print(name, tuple(p.shape), n)
    print("TOTAL params:", total)

    actor_keys = []
    actor_count = 0
    for name, p in pol.named_parameters():
        if name.startswith("mlp_extractor.policy_net") or name.startswith("action_net"):
            actor_keys.append(name)
            actor_count += p.numel()
    print("=== ACTOR-ONLY (ES optimizes these) ===")
    for k in actor_keys:
        print(" ", k)
    print("ACTOR param count:", actor_count)

    obs = torch.zeros((1, 10), dtype=torch.float32)
    with torch.no_grad():
        dist = pol.get_distribution(obs)
        logits = dist.distribution.logits
    print("logits shape:", tuple(logits.shape), "argmax:", int(logits.argmax()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
