"""Production fixed reward-magnitude scaler for RL training.

Ports the K3.5-validated ``FixedRewardScaleVecEnv`` from
``tools/h5_training_entropy_probe.py`` (lines 564-595, commit a4239af) into
the production trainer surface so K3.5c can run the standard
``python -m sight_agent.rl.train`` path with reward scaling enabled.

K3.5 mechanism finding (evidence:
``docs/k3-5b-reward-scaling-10k-confirmation-evidence.md``):
absolute reward magnitude is load-bearing under Adam at the value-head
update; dividing per-step env reward by a fixed scalar before SB3 sees it
keeps ``latent_vf_live_post = 128/128`` across the 10k confirmation while
the K3.x bias-init and architecture interventions did not.

Scope: applied to the **training** VecEnv only. The in-training eval env,
the H5 external eval pipeline (``sight_agent.rl.h5_baseline_cli``), the
fixed-observation conditioning panel, and any other read-only evaluation
surface MUST NOT be wrapped; scaling them would distort eval-time reward
metrics relative to the canonical Phase D baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class FixedRewardScaleVecEnv(VecEnvWrapper):
    """K3.5 fixed reward-magnitude scaler at the VecEnv layer.

    Divides per-step env reward by a fixed scalar before SB3 sees it, so
    returns / advantages / value targets are computed in the scaled stream
    consistently. Applied only to the training env in the production
    trainer; not applied to the in-training eval env or any external
    eval surface.

    Args:
        venv: the underlying ``VecEnv`` to wrap.
        divisor: positive float reward divisor. Must be ``> 0``.

    Raises:
        ValueError: if ``divisor <= 0``.
    """

    def __init__(self, venv: Any, divisor: float) -> None:
        if float(divisor) <= 0.0:
            raise ValueError(
                f"reward_scale_divisor must be > 0, got {divisor!r}"
            )
        super().__init__(venv)
        self.reward_scale_divisor: float = float(divisor)

    def reset(self) -> Any:
        return self.venv.reset()

    def step_wait(self) -> Any:
        obs, rewards, dones, infos = self.venv.step_wait()
        scaled_rewards = (
            np.asarray(rewards, dtype=np.float32) / self.reward_scale_divisor
        )
        return obs, scaled_rewards, dones, infos


def maybe_wrap_train_env(venv: Any, divisor: float | None) -> tuple[Any, bool]:
    """Conditionally wrap a training VecEnv with ``FixedRewardScaleVecEnv``.

    Wrapping fires only when ``divisor`` is a positive float not equal to
    ``1.0``. ``None`` and ``1.0`` are explicit no-ops and return the
    underlying ``venv`` unchanged. The second tuple element is the
    ``reward_scale_applied`` flag that the trainer records in run_start
    and summary metadata.

    Args:
        venv: the underlying training ``VecEnv``.
        divisor: positive float or ``None``.

    Returns:
        ``(venv, reward_scale_applied)``.
    """
    if divisor is None:
        return venv, False
    div = float(divisor)
    if div == 1.0:
        return venv, False
    return FixedRewardScaleVecEnv(venv, div), True
