"""K5.6 -> literal RL: PPO-finetune from BC warm-start (state MLP, no CNN).

Loads bc_policy.pt (the BC actor that cleared 930.27 in-env), copies its
weights into an SB3 PPO MlpPolicy actor (net_arch pi/vf [64,64], ReLU),
and PPO-finetunes against the production GodotSignalDodgeEnv (state mode,
headless, reward_shaping "none", max_steps 1800).

Why warm-start survives where K5.1-K5.5 cold-start collapsed: net_arch
dict(pi, vf) gives the actor and critic SEPARATE networks, and the state
feature extractor is parameter-free (Flatten). The random value head's
gradient shock flows only through value_net, never through the actor.
The K5.1-K5.5 CnnPolicy shared a CNN trunk, so value collapse dragged
perception down; here it structurally cannot.

Obs parity: the BC actor was trained on (obs - mu)/sd. We wrap the train
VecEnv so the policy sees the same normalized obs. On export we write the
finetuned actor back into the BCPolicyNet checkpoint schema with the SAME
mu/sd, so tools/k5_6_bc_eval_inenv.py and tools/k5_6_bc_render_demo.py run
on it unchanged and the number is directly comparable to BC's 1737.3.

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_6_ppo_finetune.py --timesteps 20480 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecEnvWrapper  # noqa: E402

from sight_agent.rl.factories import make_env, GODOT_SIGNAL_DODGE_V0  # noqa: E402


class FixedObsNormalize(VecEnvWrapper):
    """Apply a frozen (obs - mu)/sd to every observation.

    Not VecNormalize: stats are fixed to the BC checkpoint's mu/sd and never
    update, so train-time policy input matches exactly what the eval tool
    feeds the exported BCPolicyNet.
    """

    def __init__(self, venv, mu: np.ndarray, sd: np.ndarray):
        super().__init__(venv)
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)

    def reset(self):
        return ((self.venv.reset() - self.mu) / self.sd).astype(np.float32)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return ((obs - self.mu) / self.sd).astype(np.float32), rew, done, info


class RolloutMetricsCallback(BaseCallback):
    """Dump per-update value/actor health to ndjson for the evidence doc."""

    def __init__(self, out_path: Path):
        super().__init__()
        self.out_path = out_path
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logs = self.logger.name_to_value
        self.records.append({
            "num_timesteps": int(self.num_timesteps),
            "explained_variance": float(logs.get("train/explained_variance", float("nan"))),
            "value_loss": float(logs.get("train/value_loss", float("nan"))),
            "policy_gradient_loss": float(logs.get("train/policy_gradient_loss", float("nan"))),
            "entropy_loss": float(logs.get("train/entropy_loss", float("nan"))),
            "approx_kl": float(logs.get("train/approx_kl", float("nan"))),
        })
        self.out_path.write_text(
            "\n".join(json.dumps(r) for r in self.records) + "\n", encoding="utf-8"
        )


def load_bc(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    arch = ck["arch"]
    mu = np.asarray(ck["feat_mean"], dtype=np.float32)
    sdv = np.asarray(ck["feat_std"], dtype=np.float32)
    sdv[sdv < 1e-6] = 1.0
    return sd, arch, mu, sdv


def warm_start(policy, bc_sd: dict) -> None:
    """Copy BC actor weights into the SB3 policy. Shapes asserted."""
    pairs = [
        (policy.mlp_extractor.policy_net[0], "net.0"),
        (policy.mlp_extractor.policy_net[2], "net.2"),
        (policy.action_net, "net.4"),
    ]
    with torch.no_grad():
        for layer, key in pairs:
            w, b = bc_sd[f"{key}.weight"], bc_sd[f"{key}.bias"]
            assert tuple(layer.weight.shape) == tuple(w.shape), f"{key} w {layer.weight.shape} vs {w.shape}"
            assert tuple(layer.bias.shape) == tuple(b.shape), f"{key} b {layer.bias.shape} vs {b.shape}"
            layer.weight.copy_(w)
            layer.bias.copy_(b)


def export_bc_ckpt(policy, arch: dict, mu: np.ndarray, sd: np.ndarray,
                   out_path: Path, meta: dict) -> None:
    """Write finetuned actor back into BCPolicyNet checkpoint schema."""
    e = policy.mlp_extractor.policy_net
    out_sd = {
        "net.0.weight": e[0].weight.detach().cpu().clone(),
        "net.0.bias": e[0].bias.detach().cpu().clone(),
        "net.2.weight": e[2].weight.detach().cpu().clone(),
        "net.2.bias": e[2].bias.detach().cpu().clone(),
        "net.4.weight": policy.action_net.weight.detach().cpu().clone(),
        "net.4.bias": policy.action_net.bias.detach().cpu().clone(),
    }
    ckpt = {
        "state_dict": out_sd,
        "arch": {"in_dim": arch["in_dim"], "hidden": arch["hidden"],
                 "n_actions": arch["n_actions"]},
        "feat_mean": mu.astype(np.float32).tolist(),
        "feat_std": sd.astype(np.float32).tolist(),
        "label_map": {"0": "left", "1": "stay", "2": "right"},
        "source": "ppo_finetune_from_bc",
        **meta,
    }
    torch.save(ckpt, out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_6_ppo_finetune")
    p.add_argument("--ckpt", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "bc_policy.pt")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "ppo_ft")
    p.add_argument("--timesteps", type=int, default=20480)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1800)
    args = p.parse_args(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_sd, arch, mu, sd = load_bc(args.ckpt.resolve())

    venv = make_env(
        GODOT_SIGNAL_DODGE_V0, n_envs=1, seed=int(args.seed), mode="train",
        run_dir=str(out_dir / "godot"), max_steps=int(args.max_steps),
        headless=True, observation_mode="state", reward_shaping="none",
    )
    venv = FixedObsNormalize(venv, mu, sd)

    model = PPO(
        "MlpPolicy", venv, seed=int(args.seed), device="cpu",
        learning_rate=float(args.lr), n_steps=int(args.n_steps),
        batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                           activation_fn=nn.ReLU),
        verbose=1,
    )
    warm_start(model.policy, bc_sd)
    print("warm-start applied: actor = BC weights, value head = fresh", flush=True)

    cb = RolloutMetricsCallback(out_dir / "finetune_metrics.ndjson")
    t0 = time.monotonic()
    try:
        model.learn(total_timesteps=int(args.timesteps), callback=cb,
                    progress_bar=False)
    finally:
        venv.close()
    wall = time.monotonic() - t0

    ckpt_out = out_dir.parent / "ppo_ft_policy.pt"
    export_bc_ckpt(model.policy, arch, mu, sd, ckpt_out, meta={
        "timesteps": int(args.timesteps), "n_steps": int(args.n_steps),
        "lr": float(args.lr), "seed": int(args.seed),
        "wall_s": round(wall, 1), "bc_source": str(args.ckpt),
    })
    model.save(str(out_dir / "ppo_ft_sb3.zip"))
    print(f"DONE wall={wall:.1f}s exported={ckpt_out}", flush=True)
    print(f"METRICS {out_dir / 'finetune_metrics.ndjson'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
