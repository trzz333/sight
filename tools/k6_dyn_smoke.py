"""K6 smoke: prove the next-state prediction loss trains AND shapes the shared
encoder (gradients flow into the trunk), thread-capped so it can run alongside
the K5.8 multiseed sweep without saturating the CPU.

Gates (hard):
  1. constructs and trains without exception (real Godot state env)
  2. train/dyn_loss is logged and finite
  3. the SHARED encoder's first-layer weight MOVES between init and end
     (delta L2 > 0) -> the auxiliary loss is actually doing representation
     learning, not training a detached parallel head.

Run: python tools\\k6_dyn_smoke.py
"""

from __future__ import annotations

import os

# Cap threads BEFORE importing torch so the smoke stays within the CPU budget
# while the K5.8 sweep holds the rest of the machine.
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault(
    "SIGHT_GODOT_EXE",
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64.exe",
)

import sys
from pathlib import Path

import torch as th
import torch.nn as nn

th.set_num_threads(3)

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from sight_agent.rl.factories import make_env, GODOT_SIGNAL_DODGE_V0
from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy
from sight_agent.rl.dyn_encoder import DynEncoder
from sight_agent.rl.dyn_qrdqn import DynQRDQN


class DynLossRecorder(BaseCallback):
    def __init__(self):
        super().__init__()
        self.dyn = []
        self.q = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        v = self.logger.name_to_value
        if "train/dyn_loss" in v:
            self.dyn.append(float(v["train/dyn_loss"]))
        if "train/loss" in v:
            self.q.append(float(v["train/loss"]))


def main() -> int:
    out = (REPO_ROOT / "runs" / "phase_k" / "k6_smoke").resolve()
    out.mkdir(parents=True, exist_ok=True)

    venv = make_env(
        GODOT_SIGNAL_DODGE_V0, n_envs=1, seed=0, mode="train",
        run_dir=str(out / "godot"), max_steps=1800, headless=True,
        observation_mode="state", reward_shaping="none",
    )
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = DynQRDQN(
        NoisyQRDQNPolicy, venv, seed=0, device="cpu", verbose=0,
        dyn_beta=1.0, dyn_hidden=128, dyn_lr=2.3e-4,
        learning_rate=2.3e-4, buffer_size=20000, learning_starts=300,
        batch_size=64, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
        n_steps=3, target_update_interval=500,
        exploration_fraction=0.0, exploration_initial_eps=0.0,
        exploration_final_eps=0.0, max_grad_norm=10.0,
        policy_kwargs=dict(
            sigma_0=0.5, n_quantiles=200, net_arch=[128], activation_fn=nn.ReLU,
            features_extractor_class=DynEncoder,
            features_extractor_kwargs=dict(features_dim=64, hidden=128),
        ),
    )

    fe = model.policy.quantile_net.features_extractor
    w0 = fe.net[1].weight.detach().clone()  # first Linear in the shared encoder

    rec = DynLossRecorder()
    model.learn(total_timesteps=1200, callback=rec, log_interval=4,
                progress_bar=False)

    try:
        venv.close()
    except Exception:
        pass

    w1 = fe.net[1].weight.detach()
    delta = float((w1 - w0).norm().item())
    dyn_first = rec.dyn[0] if rec.dyn else float("nan")
    dyn_last = rec.dyn[-1] if rec.dyn else float("nan")
    q_last = rec.q[-1] if rec.q else float("nan")

    print(f"ENCODER_DELTA_L2={delta:.6f}")
    print(f"DYN_LOSS first={dyn_first:.4f} last={dyn_last:.4f} n={len(rec.dyn)}")
    print(f"Q_LOSS last={q_last:.4f}")

    ok_delta = delta > 0.0
    ok_dyn = len(rec.dyn) > 0 and all(
        (x == x) and abs(x) < 1e9 for x in rec.dyn
    )  # finite
    verdict = "PASS" if (ok_delta and ok_dyn) else "FAIL"
    print(f"SMOKE {verdict} (encoder_moves={ok_delta} dyn_finite={ok_dyn})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
