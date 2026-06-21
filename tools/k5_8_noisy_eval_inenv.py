"""K5.8 in-env greedy eval for NoisyNet QR-DQN. THE verdict for K5.8.

Loads qrdqn_noisy_sb3.zip + vecnormalize.pkl, runs greedy argmax against the
production GodotSignalDodgeEnv (state, headless, reward_shaping "none",
max_steps 1800) on held-out seeds 1000-1009 -- identical loop and kwargs to the
K5.7 / BC / PPO-finetune evals so the mean is directly comparable to the 930.27
bar, best-constant 845.7, and BC 1737.3.

NoisyNet eval semantics: model.predict(deterministic=True) sets the policy to
eval mode, which zeroes the NoisyLinear noise, so greedy action selection uses
the learned mean weights only -- fully deterministic, no exploration noise
leaking into the verdict.

PASS requires mean episode length >= 930.27 AND non-degenerate
(frac_L >= 0.03, frac_R >= 0.03, max(frac) < 0.97). A single-direction collapse
that happens to clear the bar is reported as degenerate, never relabeled a win.

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_8_noisy_eval_inenv.py --run runs\\phase_k\\k5_8_noisy_qrdqn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sb3_contrib import QRDQN  # noqa: E402

# Import the custom policy so SB3's policy registry / unpickle can resolve it.
from sight_agent.rl.noisy_qrdqn import NoisyQRDQNPolicy  # noqa: E402,F401

from k5_2_env_dynamics_probe import _build_env  # noqa: E402

BAR = 930.27
BEST_CONSTANT = 845.7
BC_MEAN = 1737.3


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_obs_stats(vecnorm_path: Path):
    import pickle
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float32)
    var = np.asarray(vn.obs_rms.var, dtype=np.float32)
    eps = float(getattr(vn, "epsilon", 1e-8))
    clip = float(getattr(vn, "clip_obs", 10.0))
    return mean, var, eps, clip


def norm_obs(obs, mean, var, eps, clip):
    x = (np.asarray(obs, dtype=np.float32) - mean) / np.sqrt(var + eps)
    return np.clip(x, -clip, clip).astype(np.float32)


def run(run_dir: Path, out_dir: Path, seeds: list[int], max_steps: int = 1800) -> dict:
    model = QRDQN.load(str(run_dir / "qrdqn_noisy_sb3.zip"), device="cpu",
                       custom_objects={"policy_class": NoisyQRDQNPolicy})
    # Belt-and-suspenders: force eval mode so NoisyLinear noise is zeroed even
    # before the first predict() call sets it.
    model.policy.set_training_mode(False)
    mean, var, eps, clip = load_obs_stats(run_dir / "vecnormalize.pkl")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(
        observation_mode="state", run_dir=out_dir / "godot",
        seed=int(seeds[0]), max_steps=max_steps, reward_shaping="none",
    )
    episodes: list[dict] = []
    try:
        for s in seeds:
            obs, info = env.reset(seed=int(s))
            steps = 0
            term = trunc = False
            reason = ""
            action_counts = [0, 0, 0]
            while steps < max_steps:
                x = norm_obs(obs, mean, var, eps, clip)
                a, _ = model.predict(x, deterministic=True)
                a = int(np.asarray(a).reshape(-1)[0])
                action_counts[a] += 1
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                if term or trunc:
                    reason = info.get("terminal_reason", "")
                    break
            episodes.append({
                "seed": int(s), "steps": steps,
                "terminated": bool(term), "truncated": bool(trunc),
                "terminal_reason": reason, "action_counts_LSR": action_counts,
            })
            print(f"seed {s}: steps={steps} reason={reason} acts={action_counts}",
                  flush=True)
    finally:
        env.close()

    lengths = [e["steps"] for e in episodes]
    mean_len = float(np.mean(lengths))
    n = len(episodes)
    coll = sum(1 for e in episodes if e["terminal_reason"] == "collision") / n
    tmo = sum(1 for e in episodes if e["terminal_reason"] == "timeout") / n
    tot = sum(sum(e["action_counts_LSR"]) for e in episodes)
    pooled = [sum(e["action_counts_LSR"][k] for e in episodes) for k in range(3)]
    frac = [round(c / tot, 3) for c in pooled] if tot else [0, 0, 0]
    nonstay = round(frac[0] + frac[2], 3)
    bar_ok = mean_len >= BAR
    nondegenerate = frac[0] >= 0.03 and frac[2] >= 0.03 and max(frac) < 0.97
    verdict = "PASS" if (bar_ok and nondegenerate) else "FAIL"
    report = {
        "run_dir": str(run_dir), "mode": "from_scratch_noisy_qrdqn",
        "eval_seeds": [int(s) for s in seeds], "max_steps": max_steps,
        "per_seed": episodes,
        "mean_episode_length": round(mean_len, 2),
        "min_episode_length": int(min(lengths)),
        "max_episode_length": int(max(lengths)),
        "collision_rate": round(coll, 3), "timeout_rate": round(tmo, 3),
        "pooled_action_counts_LSR": pooled,
        "pooled_action_fractions_LSR": frac,
        "nonstay_fraction": nonstay,
        "nondegenerate": bool(nondegenerate),
        "bar": BAR, "best_constant": BEST_CONSTANT, "bc_mean": BC_MEAN,
        "delta_vs_bar": round(mean_len - BAR, 2),
        "delta_vs_best_constant": round(mean_len - BEST_CONSTANT, 2),
        "delta_vs_bc": round(mean_len - BC_MEAN, 2),
        "verdict": verdict,
    }
    (out_dir / "noisy_eval_inenv_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"VERDICT {verdict} mean={round(mean_len, 2)} bar={BAR} "
        f"delta_vs_bar={round(mean_len - BAR, 2)} coll={round(coll, 3)} "
        f"frac_LSR={frac} nonstay={nonstay} nondegenerate={nondegenerate}",
        flush=True)
    print("REPORT", json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_8_noisy_eval_inenv")
    p.add_argument("--run", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k5_8_noisy_qrdqn")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--max-steps", type=int, default=1800)
    args = p.parse_args(argv)
    run_dir = args.run.resolve()
    out_dir = (args.out.resolve() if args.out else run_dir / "eval_inenv")
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("no seeds parsed")
    run(run_dir, out_dir, seeds, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
