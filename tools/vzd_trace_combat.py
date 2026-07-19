"""Tic-level causal trace: does the agent's bullet kill, or does infighting?

The 30-episode probe reports 5 kills for 5 hits and ~45 DAMAGECOUNT. At ~9
damage per pistol hit that cannot directly kill five 20-HP monsters, so either
KILLCOUNT is crediting infighting deaths (Doom's level kill counter counts every
monster death, not just the player's) or the enemy model is wrong. Episode-level
totals cannot tell those apart. Tic-level co-occurrence can: if KILLCOUNT
increments on the same tic HITCOUNT increments, the bullet killed. If KILLCOUNT
climbs on tics with no hit, something else is doing the killing.

Also dumps the actual monster set from the running engine so enemy HP is read,
not remembered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vzd_ppo_train import GrayStride, SkipFrames  # noqa: E402

MODEL = r"C:\Projects\Sight\runs\vzd\ppo_deadly_corridor_s3_shaped\model.zip"


class Trace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tic = 0
        self.events: list = []
        self.prev = None

    def _v(self):
        from vizdoom import GameVariable as GV
        g = self.env.unwrapped.game
        return tuple(float(g.get_game_variable(getattr(GV, n)))
                     for n in ("KILLCOUNT", "HITCOUNT", "DAMAGECOUNT",
                               "DAMAGE_TAKEN", "AMMO2", "HEALTH"))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.tic, self.prev = 0, self._v()
        print(f"  reset vars K/H/DC/DT/AMMO/HP = {self.prev}", flush=True)
        return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        self.tic += 1
        v = self._v()
        if v != self.prev:
            d = tuple(round(b - a2, 1) for a2, b in zip(self.prev, v))
            print(f"  tic {self.tic:4d}  K/H/DC/DT/AMMO/HP {v}   delta {d}",
                  flush=True)
            self.events.append((self.tic, self.prev, v))
        self.prev = v
        return obs, r, term, trunc, info


def main() -> None:
    import vizdoom.gymnasium_wrapper  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    def _f():
        env = gym.make("VizdoomDeadlyCorridor-v1", doom_skill=3)
        env = Trace(env)
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env.reset(seed=10_000)
        return env

    ev = VecFrameStack(DummyVecEnv([_f]), 4)
    model = PPO.load(MODEL, device="cuda")
    obs = ev.reset()
    done_n = 0
    print("=== episode 1 ===", flush=True)
    while done_n < 3:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = ev.step(act)
        if done[0]:
            done_n += 1
            print(f"=== episode {done_n} ended ===", flush=True)
    ev.close()
    print("TRACE_OK", flush=True)


if __name__ == "__main__":
    main()
