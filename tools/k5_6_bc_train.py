"""K5.6 behavioral-cloning trainer.

Supervised classification: map the 10-dim Signal Dodge state obs to the
K5.2 oracle's wire action {0 left, 1 stay, 2 right}. Small MLP, CPU.
Sidesteps the PPO value-head collapse seen across K5.1-K5.5 by training
the policy with supervised cross-entropy on expert (state, action) pairs
instead of RL. The deployable policy is greedy argmax over MLP logits.

Input: dataset npz from k5_6_bc_dataset.py (X (N,10) f32, y (N,) i64).
Output:
  <out>/bc_policy.pt          torch checkpoint (state_dict, arch,
                              feat mean/std baked in, label map)
  <out>/bc_train_report.json  accuracy / per-class recall / confusion

No env, no Godot here. Pure supervised learning.

Usage:
  python tools\\k5_6_bc_train.py \\
    --dataset runs\\phase_k\\k5_6_bc\\dataset_2000_2035.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]


class BCPolicyNet(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> list[list[int]]:
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m.tolist()


def train(
    dataset_path: Path,
    out_dir: Path,
    *,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    val_frac: float = 0.1,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    n = len(Xn)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr = torch.from_numpy(Xn[tr_idx])
    ytr = torch.from_numpy(y[tr_idx])
    Xva = torch.from_numpy(Xn[val_idx])
    yva = torch.from_numpy(y[val_idx])

    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    class_w = torch.tensor(w, dtype=torch.float32)

    model = BCPolicyNet(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=class_w)

    best_val_acc = -1.0
    best_state: dict | None = None
    ntr = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            out = model(Xtr[b])
            loss = lossf(out, ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(Xva).argmax(1)
            va_acc = float((va_pred == yva).float().mean())
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  val_acc={va_acc:.4f}  best={best_val_acc:.4f}", flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = model(Xva).argmax(1).numpy()
    yva_np = yva.numpy()
    conf = _confusion(yva_np, va_pred)
    per_class_recall = []
    for c in range(3):
        denom = int((yva_np == c).sum())
        rec = float((va_pred[yva_np == c] == c).mean()) if denom else None
        per_class_recall.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "arch": {"in_dim": 10, "hidden": hidden, "n_actions": 3},
        "feat_mean": mu.astype(np.float32).tolist(),
        "feat_std": sd.astype(np.float32).tolist(),
        "label_map": {"0": "left", "1": "stay", "2": "right"},
        "dataset": str(dataset_path),
        "epochs": epochs,
    }
    torch.save(ckpt, out_dir / "bc_policy.pt")

    report = {
        "dataset": str(dataset_path),
        "n_samples": int(n),
        "n_train": int(ntr),
        "n_val": int(len(Xva)),
        "action_counts_LSR": counts.astype(int).tolist(),
        "class_weights_LSR": w.round(4).tolist(),
        "best_val_acc": round(best_val_acc, 4),
        "val_per_class_recall_LSR": per_class_recall,
        "val_confusion_rows_true_cols_pred": conf,
        "hidden": hidden,
        "lr": lr,
        "batch_size": batch_size,
    }
    (out_dir / "bc_train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("REPORT", json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_6_bc_train")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc"
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    train(
        args.dataset.resolve(),
        args.out.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
