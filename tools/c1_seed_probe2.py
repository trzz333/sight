"""Verify env_method('reset', seed=...) reaches per-env reset(seed=) in SB3 2.8.0."""
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RecorderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.last_seed = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_seed = seed
        return np.zeros(2, dtype=np.float32), {"seen_seed": seed}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 1.0, False, False, {"seen_seed": self.last_seed}


def probe(name):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    cls = DummyVecEnv if name == "dummy" else SubprocVecEnv
    fns = [(lambda: RecorderEnv()) for _ in range(3)]
    ve = cls(fns, start_method="spawn") if name == "subproc" else cls(fns)
    try:
        seeds = [101, 202, 303]
        # Reset each sub-env on its own seed via env_method.
        for i, s in enumerate(seeds):
            r = ve.env_method("reset", seed=s, indices=[i])
            # r is a list with one (obs, info) tuple
        _o, _r, _d, infos = ve.step(np.zeros(3, dtype=np.int64))
        print(name, "seen:", [info.get("seen_seed") for info in infos])
    finally:
        ve.close()


if __name__ == "__main__":
    probe("dummy")
    probe("subproc")
