"""ViZDoom behavioral-cloning trainer (pixels, small CNN).

Adapts the K5.6 recipe to images: supervised cross-entropy from stacked
grayscale frames to human button-combo classes. Split is by episode to
avoid within-episode leakage. Deployable policy is greedy argmax.

Input: npz from vzd_extract_dataset.py.
Output: <out>/vzd_bc_policy.pt, <out>/vzd_bc_report.json

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_bc_train.py --scenario defend_the_center
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
STACK = 4


class VzdBCNet(nn.Module):
    def __init__(self, n_classes: int, stack: int = STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, stack, 120, 160)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.conv(x))


def stacked_indices(ep_ids: np.ndarray) -> np.ndarray:
    """For each sample i, indices of the STACK frames ending at i,
    clamped to the episode start (repeat-first padding)."""
    out = np.zeros((len(ep_ids), STACK), dtype=np.int64)
    ep_start = np.zeros(len(ep_ids), dtype=np.int64)
    for i in range(1, len(ep_ids)):
        ep_start[i] = ep_start[i - 1] if ep_ids[i] == ep_ids[i - 1] else i
    for k in range(STACK):
        out[:, STACK - 1 - k] = np.maximum(np.arange(len(ep_ids)) - k, ep_start)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds_path = Path(args.dataset) if args.dataset else \
        REPO_ROOT / "runs" / "vzd" / f"bc_dataset_{args.scenario}.npz"
    d = np.load(ds_path)
    frames, labels, ep_ids = d["frames"], d["labels"], d["episode_ids"]
    combo_map = json.loads(str(d["combo_map"]))
    n_classes = len(combo_map)

    eps = np.unique(ep_ids)
    rng.shuffle(eps)
    n_val_ep = max(1, int(len(eps) * args.val_frac)) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val_ep].tolist())
    is_val = np.isin(ep_ids, list(val_eps))
    idx_all = stacked_indices(ep_ids)
    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    print(f"{len(frames)} samples, {n_classes} classes, "
          f"train {len(tr_idx)} / val {len(va_idx)} (by episode)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = VzdBCNet(n_classes).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    ften = torch.from_numpy(frames)  # uint8, indexed per batch

    def batch(ix):
        st = ften[idx_all[ix]].to(dev).float() / 255.0
        return st, torch.from_numpy(labels[ix]).to(dev)

    best_va, best_state = -1.0, None
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(tr_idx)
        tot, corr, lsum = 0, 0, 0.0
        for i in range(0, len(order), args.batch_size):
            x, y = batch(order[i:i + args.batch_size])
            opt.zero_grad()
            logits = net(x)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            lsum += float(loss.detach()) * len(y)
            corr += int((logits.argmax(1) == y).sum())
            tot += len(y)
        net.eval()
        vcorr, vtot = 0, 0
        with torch.no_grad():
            for i in range(0, len(va_idx), 256):
                x, y = batch(va_idx[i:i + 256])
                vcorr += int((net(x).argmax(1) == y).sum())
                vtot += len(y)
        va = vcorr / vtot if vtot else float("nan")
        print(f"epoch {ep:02d} loss {lsum/tot:.4f} tr_acc {corr/tot:.3f} va_acc {va:.3f}")
        if vtot and va > best_va:
            best_va = va
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / "vzd" / f"bc_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state or net.state_dict(),
                "n_classes": n_classes, "combo_map": combo_map,
                "stack": STACK, "scenario": args.scenario},
               out_dir / "vzd_bc_policy.pt")
    (out_dir / "vzd_bc_report.json").write_text(json.dumps({
        "dataset": str(ds_path), "n_samples": int(len(frames)),
        "n_classes": n_classes, "val_acc_best": best_va,
        "epochs": args.epochs, "seed": args.seed}, indent=2))
    print(f"saved {out_dir}\\vzd_bc_policy.pt  best_va {best_va:.3f}")


if __name__ == "__main__":
    main()
