"""K6 -> NoisyNet QR-DQN + self-supervised next-state prediction. From-scratch RL.

Adapted from tools\\k5_8_noisy_qrdqn_train.py (ADAPT, not rebuild): same env,
same VecNormalize, same NoisyNet quantile head and exploration mechanism. The
ONLY structural changes from K5.8 are (a) a learnable DynEncoder trunk replaces
the parameter-free FlattenExtractor, so the value head and a next-state
prediction head share one latent, and (b) DynQRDQN adds an auxiliary next-state
MSE (weight dyn_beta) into the loss. dyn_beta>0 = self-supervision ON;
dyn_beta=0 = same architecture, auxiliary loss OFF. The on/off pair is the
controlled test of whether self-supervised prediction lifts from-scratch
reliability. NOTE: dyn_beta=0 is the architecture-matched baseline for THIS
experiment; it is NOT identical to K5.8 (K5.8 had no DynEncoder trunk), so the
comparison is strictly internal (beta on vs off at fixed architecture).

Output mirrors K5.8 so the existing eval + reliability machinery just works:
  qrdqn_noisy_sb3.zip   full SB3 model (DynQRDQN; eval uses greedy quantile head)
  vecnormalize.pkl      frozen obs stats for eval parity
  train_metrics.ndjson  per-rollout ep_len_mean / loss / dyn_loss
  train_meta.json       hyperparams + dyn_beta + wall/fps

Eval with: tools\\k6_dyn_eval_inenv.py (loads DynQRDQN, imports DynEncoder).

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k6_dyn_train.py --timesteps 200000 --seed 0 --dyn-beta 1.0 --out <dir>
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

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize  # noqa: E402

from sight_agent.rl.factories import make_env, GODOT_SIGNAL_DODGE_V0  # noqa: E402
from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy  # noqa: E402
from sight_agent.rl.dyn_encoder import DynEncoder  # noqa: E402
from sight_agent.rl.dyn_qrdqn import DynQRDQN  # noqa: E402


class TrainMetricsCallback(BaseCallback):
    """Dump rollout health to ndjson so the audit reads a learning curve."""

    def __init__(self, out_path: Path, ckpt_dir: Path, ckpt_every: int,
                 model: DynQRDQN, vecnorm: VecNormalize):
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
            self._model.save(str(self.ckpt_dir / f"qrdqn_noisy_{tag}.zip"))
            self._vecnorm.save(str(self.ckpt_dir / f"vecnorm_{tag}.pkl"))
            self._next_ckpt += self.ckpt_every
        return True

    def _on_rollout_end(self) -> None:
        logs = self.logger.name_to_value
        self.records.append({
            "num_timesteps": int(self.num_timesteps),
            "ep_rew_mean": self._ep_buffer_mean("r"),
            "ep_len_mean": self._ep_buffer_mean("l"),
            "loss": float(logs.get("train/loss", float("nan"))),
            "dyn_loss": float(logs.get("train/dyn_loss", float("nan"))),
        })
        self.out_path.write_text(
            "\n".join(json.dumps(r) for r in self.records) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k6_dyn_train")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k6_dyn")
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dyn-beta", type=float, default=1.0)
    p.add_argument("--dyn-hidden", type=int, default=128)
    p.add_argument("--dyn-lr", type=float, default=2.3e-4)
    p.add_argument("--features-dim", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--lr", type=float, default=2.3e-4)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--n-quantiles", type=int, default=200)
    p.add_argument("--sigma0", type=float, default=0.5)
    p.add_argument("--ckpt-every", type=int, default=50000)
    args = p.parse_args(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    venv = make_env(
        GODOT_SIGNAL_DODGE_V0, n_envs=1, seed=int(args.seed), mode="train",
        run_dir=str(out_dir / "godot"), max_steps=int(args.max_steps),
        headless=True, observation_mode="state", reward_shaping="none",
    )
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


    # DynQRDQN = K5.8 NoisyNet QR-DQN + auxiliary next-state MSE on a shared
    # DynEncoder trunk. dyn_beta scales the auxiliary term; dyn_beta=0 turns
    # self-supervision OFF at identical architecture (the on/off control).
    model = DynQRDQN(
        NoisyQRDQNPolicy, venv, seed=int(args.seed), device="cpu", verbose=1,
        dyn_beta=float(args.dyn_beta), dyn_hidden=int(args.dyn_hidden),
        dyn_lr=float(args.dyn_lr),
        learning_rate=float(args.lr),
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        batch_size=int(args.batch_size),
        tau=1.0, gamma=float(args.gamma),
        train_freq=int(args.train_freq),
        gradient_steps=int(args.gradient_steps),
        n_steps=int(args.n_step),
        target_update_interval=int(args.target_update_interval),
        exploration_fraction=0.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        max_grad_norm=10.0,
        policy_kwargs=dict(sigma_0=float(args.sigma0),
                           n_quantiles=int(args.n_quantiles),
                           net_arch=[128], activation_fn=nn.ReLU,
                           features_extractor_class=DynEncoder,
                           features_extractor_kwargs=dict(
                               features_dim=int(args.features_dim),
                               hidden=int(args.enc_hidden))),
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

    model.save(str(out_dir / "qrdqn_noisy_sb3.zip"))
    venv.save(str(out_dir / "vecnormalize.pkl"))
    fps = int(args.timesteps) / wall if wall > 0 else float("nan")
    meta = {
        "timesteps": int(args.timesteps), "seed": int(args.seed),
        "dyn_beta": float(args.dyn_beta), "dyn_hidden": int(args.dyn_hidden),
        "dyn_lr": float(args.dyn_lr), "features_dim": int(args.features_dim),
        "enc_hidden": int(args.enc_hidden),
        "wall_s": round(wall, 1), "fps": round(fps, 1),
        "lr": float(args.lr), "buffer_size": int(args.buffer_size),
        "learning_starts": int(args.learning_starts),
        "n_step": int(args.n_step), "n_quantiles": int(args.n_quantiles),
        "sigma_0": float(args.sigma0),
        "exploration": "noisynet_only_epsilon_disabled",
        "trunk": "DynEncoder", "net_arch": [128], "reward_shaping": "none",
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE wall={wall:.1f}s fps={fps:.1f} dyn_beta={args.dyn_beta} "
          f"model={out_dir / 'qrdqn_noisy_sb3.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
