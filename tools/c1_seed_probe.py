"""Verify SB3 2.8.0 VecEnv per-env seed propagation to env.reset(seed=...).

The C1 fitness 'mean over fresh seeds' is only valid if each worker env
actually resets on a DISTINCT seed. SB3's VecEnv.seed(seed) may take an
int or a list. This probe wraps a recorder env in DummyVecEnv and
SubprocVecEnv, calls .seed([list]) then .reset(), and prints the seed each
sub-env saw at reset.
"""
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


def probe(vec_cls_name):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    cls = DummyVecEnv if vec_cls_name == "dummy" else SubprocVecEnv
    fns = [(lambda: RecorderEnv()) for _ in range(3)]
    if vec_cls_name == "subproc":
        ve = cls(fns, start_method="spawn")
    else:
        ve = cls(fns)
    try:
        seeds = [101, 202, 303]
        try:
            ret = ve.seed(seeds)
            print(vec_cls_name, "seed(list) returned:", ret)
        except Exception as e:
            print(vec_cls_name, "seed(list) raised:", type(e).__name__, e)
        ve.reset()
        # Read the seed each sub-env saw via the info from a step.
        _o, _r, _d, infos = ve.step(np.zeros(3, dtype=np.int64))
        seen = [info.get("seen_seed") for info in infos]
        print(vec_cls_name, "seen seeds per env:", seen)
    finally:
        ve.close()


if __name__ == "__main__":
    probe("dummy")
    probe("subproc")
