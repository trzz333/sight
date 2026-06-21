"""K5.7 -> genuine from-scratch RL: SB3 DQN on Signal Dodge (state mode).

The deliverable PPO never produced: a policy trained by reinforcement
learning FROM SCRATCH (no BC, no warm start, no oracle distillation) that
clears the 930.27 constant-action baseline in-env on held-out seeds.

Why DQN, not PPO: PPO cold-start collapsed to a constant action across
K5.1-K5.5 (pixel and state, shaped reward, value-head / plasticity
failure). Per the project contract, a method that fails repeatedly is
replaced, not retried harder. DQN is structurally different: off-policy,
value-based, replay buffer, and epsilon-greedy exploration that forces
non-constant action sampling early -- a direct attack on the
constant-action basin that killed every PPO run. It is also more
sample-efficient, which matters under the single-Godot-env (~55 fps)
throughput ceiling (the factory caps Godot at n_envs=1 by charter).

Reward: reward_shaping "none" = +1 per surviving step, 0 on collision, so
the return equals the episode length. Critically this reward has NO
constant-action optimum: the best constant action survives 845.7 < the
930.27 bar, so any policy that clears the bar MUST dodge. K5.5's shaped
reward (alpha 0.30) was satisfiable by a constant action and produced a
degenerate optimum; "none" removes that trap.

Obs scaling: VecNormalize(norm_obs=True, norm_reward=False). Running
mean/std over states, computed live during training (NOT borrowed from
the oracle/BC dataset), saved to vecnormalize.pkl for eval parity. Reward
is left untouched so the logged return stays in episode-length units.

Output (under <out>/):
  dqn_sb3.zip          full SB3 model (resume + eval)
  vecnormalize.pkl     frozen obs stats for eval
  train_metrics.ndjson per-rollout ep_rew_mean / loss / exploration_rate
  ckpts/               periodic dqn_<steps>.zip + vecnorm_<steps>.pkl

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_7_dqn_train.py --timesteps 200000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize  # noqa: E402

from sight_agent.rl.factories import make_env, GODOT_SIGNAL_DODGE_V0  # noqa: E402


class TrainMetricsCallback(BaseCallback):
    """Dump rollout health to ndjson so the audit reads a learning curve."""

    def __init__(self, out_path: Path, ckpt_dir: Path, ckpt_every: int,
                 model: DQN, vecnorm: VecNormalize):
        super().__init__()
        self.out_path = out_path
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = int(ckpt_every)
        self._model = model
        self._vecnorm = vecnorm
        self.records: list[dict] = []
        self._next_ckpt = self.ckpt_every

    def _ep_buffer_mean(self, key: str):
        buf = self.model.ep_info_buffer
        if not buf:
            return float("nan")
        vals = [ep[key] for ep in buf if key in ep]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_ckpt:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = self.num_timesteps
            self._model.save(str(self.ckpt_dir / f"dqn_{tag}.zip"))
            self._vecnorm.save(str(self.ckpt_dir / f"vecnorm_{tag}.pkl"))
            self._next_ckpt += self.ckpt_every
        return True

    def _on_rollout_end(self) -> None:
        logs = self.logger.name_to_value
        self.records.append({
            "num_timesteps": int(self.num_timesteps),
            "ep_rew_mean": self._ep_buffer_mean("r"),
            "ep_len_mean": self._ep_buffer_mean("l"),
            "exploration_rate": float(logs.get("rollout/exploration_rate", float("nan"))),
            "loss": float(logs.get("train/loss", float("nan"))),
        })
        self.out_path.write_text(
            "\n".join(json.dumps(r) for r in self.records) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_7_dqn_train")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k5_7_dqn")
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--exploration-fraction", type=float, default=0.3)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ckpt-every", type=int, default=50000)
    args = p.parse_args(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    venv = make_env(
        GODOT_SIGNAL_DODGE_V0, n_envs=1, seed=int(args.seed), mode="train",
        run_dir=str(out_dir / "godot"), max_steps=int(args.max_steps),
        headless=True, observation_mode="state", reward_shaping="none",
    )
    venv = VecMonitor(venv)  # populate ep_info_buffer so ep_rew_mean/ep_len_mean log
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = DQN(
        "MlpPolicy", venv, seed=int(args.seed), device="cpu", verbose=1,
        learning_rate=float(args.lr),
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        batch_size=int(args.batch_size),
        tau=1.0, gamma=float(args.gamma),
        train_freq=int(args.train_freq),
        gradient_steps=int(args.gradient_steps),
        target_update_interval=int(args.target_update_interval),
        exploration_fraction=float(args.exploration_fraction),
        exploration_initial_eps=1.0,
        exploration_final_eps=float(args.exploration_final_eps),
        max_grad_norm=10.0,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.ReLU),
    )

    cb = TrainMetricsCallback(
        out_dir / "train_metrics.ndjson", out_dir / "ckpts",
        int(args.ckpt_every), model, venv,
    )
    t0 = time.monotonic()
    try:
        model.learn(total_timesteps=int(args.timesteps), callback=cb,
                    progress_bar=False, log_interval=4)
    finally:
        try:
            venv.close()
        except Exception:
            pass
    wall = time.monotonic() - t0

    model.save(str(out_dir / "dqn_sb3.zip"))
    venv.save(str(out_dir / "vecnormalize.pkl"))
    fps = int(args.timesteps) / wall if wall > 0 else float("nan")
    meta = {
        "timesteps": int(args.timesteps), "seed": int(args.seed),
        "wall_s": round(wall, 1), "fps": round(fps, 1),
        "lr": float(args.lr), "buffer_size": int(args.buffer_size),
        "learning_starts": int(args.learning_starts),
        "exploration_fraction": float(args.exploration_fraction),
        "exploration_final_eps": float(args.exploration_final_eps),
        "net_arch": [128, 128], "reward_shaping": "none",
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE wall={wall:.1f}s fps={fps:.1f} model={out_dir / 'dqn_sb3.zip'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
