"""H4 Step 5: SB3 PPO CnnPolicy construction smoke test.

Per docs/sight-h4-plan.md section 8 and implementation sequence step 6.
Builds a PPO model with CnnPolicy over a fresh Gymnasium stub env that
exposes the H4 pixel observation contract, runs one predict call, then
runs model.learn for a small rollout that enters PPO's train path. Asserts
env reset/step counters advanced and the policy optimizer fired at least
once.

Default-tier. No live Godot. CPU-first. CUDA is not part of the
acceptance bar. The stub env intentionally does not subclass
GodotSignalDodgeEnv: this test is the SB3 CnnPolicy construction surface,
not a Godot transport test.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class _PixelStubEnv(gym.Env):
    """Minimal Gymnasium env exposing the H4 pixel observation contract.

    observation_space = Box(0, 255, (1, 84, 84), uint8).
    action_space      = Discrete(3).
    Episodes terminate every EPISODE_HORIZON steps so an 8-step rollout
    forces SB3 to exercise the reset path mid-rollout.
    Observations are deterministic non-blank uint8 patterns keyed on
    step_count, so predict and learn never feed a degenerate all-zero
    image into the conv stack.
    """

    metadata = {"render_modes": []}

    EPISODE_HORIZON = 4

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(3)
        self.reset_count = 0
        self.step_count = 0
        self._episode_step = 0

    def _make_obs(self, frame: int) -> np.ndarray:
        # Deterministic non-blank pattern: a vertical stripe whose column
        # advances with the frame counter, plus a horizontal band so the
        # conv stack sees more than one non-zero element per axis.
        obs = np.zeros((1, 84, 84), dtype=np.uint8)
        col = frame % 84
        row = (frame * 7) % 84
        obs[0, :, col] = 255
        obs[0, row, :] = 128
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self._episode_step = 0
        return self._make_obs(self.step_count), {"reset_count": self.reset_count}

    def step(self, action):
        assert self.action_space.contains(int(action)), action
        self.step_count += 1
        self._episode_step += 1
        terminated = self._episode_step >= self.EPISODE_HORIZON
        truncated = False
        reward = 0.0
        info = {"step_count": self.step_count, "episode_step": self._episode_step}
        return self._make_obs(self.step_count), reward, terminated, truncated, info


def test_cnn_policy_construct_predict_and_train_smoke():
    """PPO CnnPolicy must construct, predict a valid action, and run a short
    rollout that enters the train path. Acceptance is reset/step/optimizer
    activity, not learning quality."""
    from stable_baselines3 import PPO

    env = _PixelStubEnv()

    model = PPO(
        "CnnPolicy",
        env,
        device="cpu",
        seed=0,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-4,
        verbose=0,
    )

    obs, _info = env.reset()
    action, _state = model.predict(obs, deterministic=True)
    action_int = int(np.asarray(action).item())
    assert action_int in {0, 1, 2}, action_int

    optimizer_step_count = {"n": 0}
    real_step = model.policy.optimizer.step

    def counted_step(*args, **kwargs):
        optimizer_step_count["n"] += 1
        return real_step(*args, **kwargs)

    model.policy.optimizer.step = counted_step  # type: ignore[assignment]

    model.learn(total_timesteps=8)

    assert env.reset_count >= 1, env.reset_count
    assert env.step_count >= 8, env.step_count
    assert optimizer_step_count["n"] >= 1, optimizer_step_count["n"]
