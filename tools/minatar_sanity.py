"""MinAtar adoption sanity: register, step, random-policy floor on Breakout.
Deterministic seed. Reports obs shape/dtype, action space, and random-return stats."""
import numpy as np
import gymnasium as gym
from minatar.gym import register_envs

register_envs()

SEED = 0
N_EP = 30

env = gym.make("MinAtar/Breakout-v1")  # v1 = minimal action set (Breakout: 3)
obs, info = env.reset(seed=SEED)
print("obs.shape", obs.shape, "obs.dtype", obs.dtype)
print("action_space", env.action_space)
print("obs_space", env.observation_space)

rng = np.random.default_rng(SEED)
returns, lengths = [], []
for ep in range(N_EP):
    obs, info = env.reset(seed=SEED + ep)
    done = False
    R, L = 0.0, 0
    while not done:
        a = int(rng.integers(env.action_space.n))
        obs, r, term, trunc, info = env.step(a)
        R += r; L += 1
        done = term or trunc
        if L > 100000:
            break
    returns.append(R); lengths.append(L)

returns = np.array(returns); lengths = np.array(lengths)
print(f"random floor over {N_EP} eps: return mean={returns.mean():.3f} std={returns.std():.3f} "
      f"min={returns.min():.0f} max={returns.max():.0f}")
print(f"ep length mean={lengths.mean():.1f}")
env.close()
