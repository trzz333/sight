"""H1 SB3 callbacks and logger writers that emit NDJSON events.

NDJSONKVWriter plugs into SB3's Logger so every dump becomes a train_metrics event.
NDJSONCallback runs periodic eval and writes the run_end event.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import KVWriter

from .ndjson_logger import NDJSONLogger, to_jsonable


class NDJSONKVWriter(KVWriter):
    """SB3 logger output that writes one train_metrics NDJSON event per dump."""

    def __init__(self, ndjson: NDJSONLogger) -> None:
        self._ndjson = ndjson

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, Any],
        step: int = 0,
    ) -> None:
        metrics = {k: to_jsonable(v) for k, v in key_values.items()}
        self._ndjson.log_event("train_metrics", step=int(step), metrics=metrics)

    def close(self) -> None:
        # NDJSONLogger lifecycle owned by train.py.
        return None


class NDJSONCallback(BaseCallback):
    """Periodic eval + run_end NDJSON event emitter.

    eval_freq is measured in environment steps as counted by self.n_calls. When
    eval_freq <= 0 no periodic eval runs; a final eval is always attempted in
    _on_training_end.
    """

    def __init__(
        self,
        ndjson: NDJSONLogger,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        artifact_paths: dict[str, str],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._ndjson = ndjson
        self._eval_env = eval_env
        self._eval_freq = int(eval_freq)
        self._n_eval_episodes = int(n_eval_episodes)
        self._deterministic = bool(deterministic)
        self._artifact_paths = dict(artifact_paths)
        self._t_start: float | None = None
        self._final_eval_done = False

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        if self._eval_freq > 0 and self.n_calls % self._eval_freq == 0:
            self._do_eval()
        return True

    def _on_training_end(self) -> None:
        if not self._final_eval_done:
            try:
                self._do_eval()
            except Exception as exc:  # eval failure must not mask training success
                self._ndjson.log_event(
                    "error",
                    step=int(self.num_timesteps),
                    status="error",
                    error_type=type(exc).__name__,
                    message=f"final eval failed: {exc}",
                )
        elapsed = time.time() - (self._t_start or time.time())
        self._ndjson.log_event(
            "run_end",
            step=int(self.num_timesteps),
            status="ok",
            elapsed_seconds=float(elapsed),
            total_timesteps=int(self.num_timesteps),
            artifact_paths=self._artifact_paths,
        )

    def _do_eval(self) -> None:
        ep_rewards, _ep_lengths = evaluate_policy(
            self.model,
            self._eval_env,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
            return_episode_rewards=True,
            warn=False,
        )
        ep_rewards_list = [float(r) for r in ep_rewards]
        mean_r = float(statistics.fmean(ep_rewards_list)) if ep_rewards_list else 0.0
        std_r = float(statistics.pstdev(ep_rewards_list)) if len(ep_rewards_list) > 1 else 0.0
        self._ndjson.log_event(
            "eval",
            step=int(self.num_timesteps),
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
            metrics={
                "mean_reward": mean_r,
                "std_reward": std_r,
                "episode_rewards": ep_rewards_list,
            },
        )
        self._final_eval_done = True


def _resolve_schedule(value: Any) -> Any:
    """If value is an SB3 schedule (callable), call with progress_remaining=1.0."""
    if callable(value):
        try:
            return value(1.0)
        except Exception:
            return str(value)
    return value


def introspect_effective_hyperparams(model: Any) -> dict[str, Any]:
    """Best-effort JSON-safe snapshot of PPO algorithm hyperparameters.

    Source of truth is the instantiated model. Values are runtime-introspected;
    they are not web-verified library defaults.
    """
    candidates = (
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "sde_sample_freq",
        "seed",
    )
    out: dict[str, Any] = {}
    for name in candidates:
        if not hasattr(model, name):
            continue
        raw = getattr(model, name)
        out[name] = to_jsonable(_resolve_schedule(raw))
    policy_obj = getattr(model, "policy", None)
    out["policy_class"] = type(policy_obj).__name__ if policy_obj is not None else None
    device = getattr(model, "device", None)
    out["device"] = str(device) if device is not None else None
    return out
