"""
Phase M / M1 - CartPole-v1 PPO reference control.

Infra-loop audit: run the benchmarked RL-Zoo CartPole-v1 PPO config on SB3
on this machine and confirm it solves (mean reward ~500). This validates the
SB3 harness as known-good BEFORE any Signal Dodge from-scratch training.
If this fails, every prior Sight RL failure is suspect harness, not algorithm.

RL-Zoo CartPole-v1 PPO config (DLR-RM/rl-baselines3-zoo hyperparams/ppo.yml):
  MlpPolicy, n_envs 8, n_timesteps 1e5, n_steps 32, batch_size 256,
  gae_lambda 0.8, gamma 0.98, n_epochs 20, ent_coef 0.0,
  learning_rate lin_0.001, clip_range lin_0.2
"""
import json
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

OUT = r"C:\Projects\Sight\runs\phase_m\m1_cartpole_control_report.json"
SOLVED_THRESHOLD = 475.0  # CartPole-v1 caps at 500; "solved" is 475 over 100 eps


def lin(initial):
    def f(progress_remaining):
        return progress_remaining * initial
    return f


def main():
    train_env = make_vec_env("CartPole-v1", n_envs=8, seed=0)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=32,
        batch_size=256,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
        learning_rate=lin(0.001),
        clip_range=lin(0.2),
        seed=0,
        verbose=0,
    )
    model.learn(total_timesteps=100_000)

    eval_env = make_vec_env("CartPole-v1", n_envs=1, seed=1000)
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    verdict = "PASS" if mean_r >= SOLVED_THRESHOLD else "FAIL"
    report = {
        "phase": "M1",
        "env": "CartPole-v1",
        "config_source": "DLR-RM/rl-baselines3-zoo hyperparams/ppo.yml (tuned CartPole-v1)",
        "total_timesteps": 100_000,
        "n_eval_episodes": 20,
        "eval_deterministic": True,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "solved_threshold": SOLVED_THRESHOLD,
        "verdict": verdict,
        "sb3": __import__("stable_baselines3").__version__,
        "gymnasium": __import__("gymnasium").__version__,
        "python": sys.version.split()[0],
    }
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
