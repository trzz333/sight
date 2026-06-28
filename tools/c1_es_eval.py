"""Phase N / C1 - gate eval of an ES best-actor vector on Signal Dodge.

Loads a C1 ES `best_actor_vec.npy` (actor-only weights, 5059 params) into
the SAME SB3 MlpPolicy the trainer uses, rolls it deterministically on the
held-out eval seeds 1000-1009, and applies the identical M2 PASS gate:

    PASS = mean_episode_length >= 930.27
           AND frac_left  >= 0.03
           AND frac_right >= 0.03
           AND max(action_fraction) < 0.97

Apples-to-apples with M2/M2.1: same env (state obs, reward "none",
max_steps 1800, headless), same deterministic argmax action rule
(predict_action in the trainer). ES trained on RAW obs with no
VecNormalize, so no normalization stats are applied here.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

from c1_es_train import (  # noqa: E402
    ACTION_NAMES,
    DEFAULT_EXE,
    DEFAULT_PROJECT,
    MAX_STEPS,
    SURVIVAL_BAR,
    _build_godot_env,
    build_policy,
    load_actor,
    predict_action,
)


def _alloc_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def _reset(env, seed):
    r = env.reset(seed=seed)
    return r[0] if isinstance(r, tuple) else r


def _step(env, action):
    r = env.step(action)
    if len(r) == 5:
        obs, _rew, terminated, truncated, _info = r
        return obs, bool(terminated) or bool(truncated)
    obs, _rew, done, _info = r
    return obs, bool(done)


def roll_episode(pol, env, seed: int) -> dict:
    obs = _reset(env, seed)
    counts = {0: 0, 1: 0, 2: 0}
    length = 0
    while length < MAX_STEPS:
        act = predict_action(pol, obs)
        obs, done = _step(env, act)
        counts[act] += 1
        length += 1
        if done:
            break
    return {"seed": int(seed), "length": int(length), "action_counts": counts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="c1_es_eval")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--vec", help="path to best_actor_vec.npy")
    src.add_argument("--run-dir", help="dir containing best_actor_vec.npy")
    p.add_argument("--label", default="C1")
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--exe", default=DEFAULT_EXE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    vec_path = Path(args.vec) if args.vec else Path(args.run_dir) / "best_actor_vec.npy"
    if not vec_path.is_file():
        raise SystemExit(f"best_actor_vec.npy not found: {vec_path}")
    best_vec = np.load(str(vec_path))

    seeds = parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_run_dir = out / "godot_eval"
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    pol, actor_keys, numel = build_policy()
    assert numel == best_vec.size == 5059, (
        f"param mismatch: policy {numel}, vec {best_vec.size}")
    load_actor(pol, actor_keys, best_vec)

    env = _build_godot_env(port=_alloc_port(), seed=None,
                           run_dir=str(eval_run_dir), exe=args.exe,
                           project=args.project)
    t0 = time.time()
    try:
        eps = [roll_episode(pol, env, s) for s in seeds]
    finally:
        try:
            env.close()
        except Exception:
            pass

    lengths = [e["length"] for e in eps]
    tot = {0: 0, 1: 0, 2: 0}
    for e in eps:
        for a in (0, 1, 2):
            tot[a] += e["action_counts"][a]
    total = sum(tot.values()) or 1
    frac = {ACTION_NAMES[a]: tot[a] / total for a in (0, 1, 2)}
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    max_frac = max(frac.values())
    passed = (
        mean_len >= SURVIVAL_BAR
        and frac["left"] >= 0.03
        and frac["right"] >= 0.03
        and max_frac < 0.97
    )
    summary = {
        "phase": "C1-eval",
        "label": args.label,
        "vec_path": str(vec_path),
        "survival_bar": SURVIVAL_BAR,
        "eval_seeds": seeds,
        "gate": "mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97",
        "mean_episode_length": mean_len,
        "min_episode_length": float(min(lengths)) if lengths else 0.0,
        "max_episode_length": float(max(lengths)) if lengths else 0.0,
        "n_seeds": len(seeds),
        "action_fractions": frac,
        "max_action_fraction": max_frac,
        "per_seed_length": {str(e["seed"]): e["length"] for e in eps},
        "gate_pass": bool(passed),
        "verdict": "C1-PASS" if passed else "C1-FAIL",
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with (out / "c1_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
