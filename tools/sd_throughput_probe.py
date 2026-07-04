"""One-off throughput probe for GodotSignalDodgeEnv (state, headless).

Measures raw steps/sec over the pause-world TCP transport so the budget
question (can Signal Dodge afford a MinAtar-scale from-scratch PPO run
directly on Godot?) is answered with a number, not a guess.
"""
from __future__ import annotations

import os
import time

from sight_agent.rl.godot_env import GodotSignalDodgeEnv

EXE = os.environ["SIGHT_GODOT_EXE"]
PROJ = os.path.join(os.getcwd(), "games", "signal-dodge")

env = GodotSignalDodgeEnv(
    godot_executable=EXE,
    project_path=PROJ,
    max_steps=1800,
    headless=True,
    observation_mode="state",
)
obs, info = env.reset(seed=1234)
rng = __import__("numpy").random.default_rng(0)

N = 3000
t0 = time.perf_counter()
steps = 0
eps = 0
for _ in range(N):
    a = int(rng.integers(0, 3))
    obs, r, term, trunc, info = env.step(a)
    steps += 1
    if term or trunc:
        eps += 1
        env.reset(seed=1234 + eps)
dt = time.perf_counter() - t0
env.close()
print(f"STEPS={steps} EPISODES={eps} WALL_S={dt:.2f} STEPS_PER_S={steps/dt:.1f}")
