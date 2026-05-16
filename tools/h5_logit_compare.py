"""Phase H logit-distribution comparator (docs-and-tools-only slice).

Loads two or more SB3 PPO CnnPolicy checkpoints, drives the Godot
Signal Dodge env with a fixed behavior tape (so the trajectory does
not depend on any model), and at each step queries every loaded model
for per-action logits, probabilities, entropy, argmax, and top1-top2
margin against the same observation.

Outputs an NDJSON stream (one header row plus one row per env step)
and a sibling ``.summary.json`` with per-model and pairwise aggregates.

Model-loading verification is mandatory: every model.zip is fingerprinted
by file SHA-256, archive member list, parameter count, and state-dict
blake2b digest. Two models sharing identical SHA-256 abort the run with
a "model-loading bug suspected" error. Observation hashing is logged
per step so frame-stack and observation-freshness pathologies surface
as cheap signals rather than requiring an encoder probe.

No training. No env code changes. Loads only existing committed
artifacts from runs/rl/ ... train run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

# tools/ scripts run from repo root; src is the canonical import root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.factories import make_env  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402


TAPE_KEYWORDS = {"stay": 1, "left": 0, "right": 2}
ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}


def parse_models_arg(models_str: str) -> dict[str, Path]:
    """Parse ``--models`` into an ordered label->path map.

    Format: ``label1=path1,label2=path2``. Each path is a train run
    directory containing model.zip. At least two labels required.
    """
    out: dict[str, Path] = {}
    for tok in models_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--models token missing '=': {tok!r}")
        label, path = tok.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--models token has empty label or path: {tok!r}")
        if label in out:
            raise ValueError(f"--models label repeated: {label!r}")
        out[label] = Path(path)
    if len(out) < 2:
        raise ValueError("at least two --models entries required for a comparator")
    return out


def parse_tape(tape_arg: str, max_steps: int) -> list[int]:
    """Return the behavior tape as a list of action ints of length max_steps.

    Accepts a keyword in {stay, left, right} producing a constant tape, or
    a comma-separated sequence of action ints (cycled to max_steps).
    """
    key = tape_arg.strip().lower()
    if key in TAPE_KEYWORDS:
        return [TAPE_KEYWORDS[key]] * int(max_steps)
    parts = [int(t.strip()) for t in tape_arg.split(",") if t.strip()]
    if not parts:
        raise ValueError(f"empty tape: {tape_arg!r}")
    if any(a not in ACTION_NAMES for a in parts):
        raise ValueError(f"tape contains action outside {{0,1,2}}: {parts}")
    return [parts[i % len(parts)] for i in range(int(max_steps))]


def fingerprint_model_file(model_zip: Path) -> dict[str, Any]:
    """Return file-level fingerprint of a saved SB3 model archive."""
    if not model_zip.is_file():
        raise FileNotFoundError(f"model.zip not found: {model_zip}")
    size = int(model_zip.stat().st_size)
    h = hashlib.sha256()
    with model_zip.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with zipfile.ZipFile(model_zip, "r") as zf:
        members = sorted(zf.namelist())
    return {
        "path": str(model_zip),
        "size_bytes": size,
        "sha256": sha,
        "zip_members": members,
    }


def fingerprint_loaded_model(model: Any) -> dict[str, Any]:
    """Hash policy.state_dict for cheap parameter-equality detection.

    Concatenates each tensor's bytes in sorted key order under a single
    blake2b-128 hash. Also reports the total parameter count and the
    sorted state_dict key list.
    """
    sd = model.policy.state_dict()
    keys = sorted(sd.keys())
    total = 0
    h = hashlib.blake2b(digest_size=16)
    for k in keys:
        t = sd[k].detach().cpu().contiguous().to(torch.float32).numpy()
        h.update(k.encode("utf-8"))
        h.update(t.tobytes())
        total += int(t.size)
    return {
        "param_count": total,
        "state_dict_keys": keys,
        "state_dict_blake2b16": h.hexdigest(),
    }


def load_and_fingerprint(train_run_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load a PPO model from a train run dir and return (model, fingerprint)."""
    from stable_baselines3 import PPO

    model_zip = train_run_dir / "model.zip"
    file_fp = fingerprint_model_file(model_zip)
    model = PPO.load(str(model_zip), env=None, device="cpu")
    param_fp = fingerprint_loaded_model(model)
    fp = {"train_run_dir": str(train_run_dir), **file_fp, **param_fp}
    return model, fp


def get_action_logits(
    model: Any, obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (logits, probs, entropy, argmax) for one obs.

    The categorical distribution exposed by SB3's policy normalizes
    raw action-head logits in its constructor (PyTorch Categorical
    stores log_softmax under ``.logits``). Pairwise comparison metrics
    on these normalized log-probs and probabilities are well-defined
    regardless of any raw-logit offset.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        cat = dist.distribution  # torch.distributions.Categorical
        logits = cat.logits.detach().cpu().numpy().reshape(-1).astype(np.float64)
        probs = cat.probs.detach().cpu().numpy().reshape(-1).astype(np.float64)
        entropy_v = float(cat.entropy().detach().cpu().numpy().reshape(-1)[0])
    argmax = int(np.argmax(probs))
    return logits, probs, entropy_v, argmax


def centered(arr: np.ndarray) -> np.ndarray:
    """Subtract the mean across the action axis."""
    return arr - float(arr.mean())


def pair_metrics(
    logits_a: np.ndarray, probs_a: np.ndarray,
    logits_b: np.ndarray, probs_b: np.ndarray,
) -> dict[str, Any]:
    """Return per-step pairwise distribution distances."""
    ca = centered(logits_a)
    cb = centered(logits_b)
    eps = 1e-12
    pa = np.clip(probs_a, eps, 1.0)
    pb = np.clip(probs_b, eps, 1.0)
    kl_ab = float(np.sum(pa * (np.log(pa) - np.log(pb))))
    kl_ba = float(np.sum(pb * (np.log(pb) - np.log(pa))))
    return {
        "centered_logit_l2": float(np.linalg.norm(ca - cb)),
        "prob_l1": float(np.sum(np.abs(probs_a - probs_b))),
        "sym_kl": float(0.5 * (kl_ab + kl_ba)),
        "same_argmax": bool(int(np.argmax(probs_a)) == int(np.argmax(probs_b))),
    }


def obs_hash(obs: np.ndarray) -> str:
    """Short blake2b digest of obs bytes (12 hex chars)."""
    arr = np.ascontiguousarray(np.asarray(obs))
    return hashlib.blake2b(arr.tobytes(), digest_size=6).hexdigest()


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def aggregate(
    rows: list[dict[str, Any]], model_labels: list[str],
) -> dict[str, Any]:
    """Compute per-model and pairwise aggregates from collected step rows."""
    per_model: dict[str, Any] = {}
    for label in model_labels:
        argmax_counts = {0: 0, 1: 0, 2: 0}
        entropies: list[float] = []
        margins: list[float] = []
        per_action_probs: list[list[float]] = [[], [], []]
        same_as_tape = 0
        n = 0
        for r in rows:
            m = r["models"][label]
            argmax_counts[int(m["argmax"])] += 1
            entropies.append(float(m["entropy"]))
            sorted_probs = sorted(m["probs"], reverse=True)
            margins.append(float(sorted_probs[0] - sorted_probs[1]))
            for i in range(3):
                per_action_probs[i].append(float(m["probs"][i]))
            if int(m["argmax"]) == int(r["tape_action"]):
                same_as_tape += 1
            n += 1
        per_model[label] = _per_model_block(
            n, argmax_counts, entropies, margins, per_action_probs, same_as_tape,
        )
    pairs: dict[str, Any] = {}
    labels = model_labels
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            key = f"{labels[i]}__vs__{labels[j]}"
            l2s: list[float] = []
            l1s: list[float] = []
            kls: list[float] = []
            same_argmax = 0
            n = 0
            for r in rows:
                p = r["pairs"][key]
                l2s.append(float(p["centered_logit_l2"]))
                l1s.append(float(p["prob_l1"]))
                kls.append(float(p["sym_kl"]))
                if bool(p["same_argmax"]):
                    same_argmax += 1
                n += 1
            pairs[key] = {
                "n_steps": n,
                "same_argmax_fraction": (same_argmax / n) if n else 0.0,
                "centered_logit_l2": _stat_block(l2s),
                "prob_l1": _stat_block(l1s),
                "sym_kl": _stat_block(kls),
            }
    hashes = [r["obs_hash"] for r in rows]
    unique = sorted(set(hashes))
    runs_lengths: list[int] = []
    prev = None
    cur = 0
    for hh in hashes:
        if hh == prev:
            cur += 1
        else:
            if cur > 0:
                runs_lengths.append(cur)
            cur = 1
            prev = hh
    if cur > 0:
        runs_lengths.append(cur)
    obs_stats = {
        "n_steps": len(hashes),
        "unique_hash_count": len(unique),
        "all_distinct": (len(unique) == len(hashes)),
        "max_consecutive_repeat_run": int(max(runs_lengths) if runs_lengths else 0),
    }
    return {"per_model": per_model, "pairs": pairs, "obs": obs_stats}


def _stat_block(vs: list[float]) -> dict[str, float]:
    if not vs:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(statistics.fmean(vs)),
        "median": float(statistics.median(vs)),
        "p95": percentile(vs, 95.0),
        "max": float(max(vs)),
        "min": float(min(vs)),
    }


def _per_model_block(
    n: int,
    argmax_counts: dict[int, int],
    entropies: list[float],
    margins: list[float],
    per_action_probs: list[list[float]],
    same_as_tape: int,
) -> dict[str, Any]:
    return {
        "n_steps": int(n),
        "argmax_counts": {ACTION_NAMES[k]: int(argmax_counts[k]) for k in (0, 1, 2)},
        "argmax_fractions": {
            ACTION_NAMES[k]: (argmax_counts[k] / n if n else 0.0) for k in (0, 1, 2)
        },
        "entropy": _stat_block(entropies),
        "top1_top2_margin": _stat_block(margins),
        "per_action_probability_mean": {
            ACTION_NAMES[i]: (
                float(statistics.fmean(per_action_probs[i]))
                if per_action_probs[i] else 0.0
            )
            for i in (0, 1, 2)
        },
        "per_action_probability_std": {
            ACTION_NAMES[i]: (
                float(statistics.pstdev(per_action_probs[i]))
                if len(per_action_probs[i]) > 1 else 0.0
            )
            for i in (0, 1, 2)
        },
        "same_as_tape_fraction": (same_as_tape / n) if n else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="h5_logit_compare",
        description="Phase H docs-and-tools-only logit comparator.",
    )
    p.add_argument("--config", required=True, help="Path to YAML env config.")
    p.add_argument(
        "--models", required=True,
        help="Comma-separated label=train_run_dir pairs (>=2).",
    )
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument(
        "--behavior-tape", required=True,
        help="Keyword (stay|left|right) or comma-separated action ints.",
    )
    p.add_argument("--out", required=True, help="Output NDJSON path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    models_map = parse_models_arg(args.models)
    cfg = load_config(args.config)
    tape = parse_tape(args.behavior_tape, args.max_steps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    loaded: list[tuple[str, Any, dict[str, Any]]] = []
    seen_sha: dict[str, str] = {}
    seen_param_hash: dict[str, str] = {}
    for label, run_dir in models_map.items():
        model, fp = load_and_fingerprint(run_dir)
        if fp["sha256"] in seen_sha:
            other = seen_sha[fp["sha256"]]
            raise RuntimeError(
                f"models {label!r} and {other!r} share identical sha256; "
                "model-loading bug suspected. Aborting."
            )
        seen_sha[fp["sha256"]] = label
        ph = fp["state_dict_blake2b16"]
        if ph in seen_param_hash:
            other = seen_param_hash[ph]
            raise RuntimeError(
                f"models {label!r} and {other!r} share identical state_dict "
                "parameter hash with non-identical zip SHA-256; treat as a "
                "model-loading bug. Aborting."
            )
        seen_param_hash[ph] = label
        loaded.append((label, model, fp))
    labels = [l for l, _, _ in loaded]
    fingerprints = {l: fp for l, _, fp in loaded}

    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)
    eval_run_dir = out_path.parent / f"{out_path.stem}_godot"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    if godot_extra:
        env = make_env(
            env_id, n_envs=1, seed=base_seed, mode="eval",
            run_dir=str(eval_run_dir), **godot_extra,
        )
    else:
        env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval")

    rows: list[dict[str, Any]] = []
    started_at = time.time()
    try:
        try:
            env.seed(int(args.eval_seed))
        except (AttributeError, TypeError):
            pass
        obs = env.reset()
        for t in range(int(args.max_steps)):
            tape_action = int(tape[t])
            obs_arr = np.asarray(obs)
            row: dict[str, Any] = {
                "t": t,
                "tape_action": tape_action,
                "obs_hash": obs_hash(obs_arr),
                "obs_shape": list(obs_arr.shape),
                "obs_dtype": str(obs_arr.dtype),
                "obs_min": int(obs_arr.min()),
                "obs_max": int(obs_arr.max()),
                "obs_mean": float(obs_arr.mean()),
            }
            model_blocks: dict[str, Any] = {}
            log_probs_cache: dict[str, np.ndarray] = {}
            probs_cache: dict[str, np.ndarray] = {}
            for label, model, _ in loaded:
                logits, probs, entropy_v, argmax = get_action_logits(model, obs_arr)
                sp = np.sort(probs)[::-1]
                margin = float(sp[0] - sp[1])
                model_blocks[label] = {
                    "logits": [float(x) for x in logits],
                    "probs": [float(x) for x in probs],
                    "entropy": float(entropy_v),
                    "argmax": int(argmax),
                    "top1_top2_margin": margin,
                }
                log_probs_cache[label] = logits
                probs_cache[label] = probs
            pair_blocks: dict[str, Any] = {}
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = labels[i], labels[j]
                    key = f"{a}__vs__{b}"
                    pair_blocks[key] = pair_metrics(
                        log_probs_cache[a], probs_cache[a],
                        log_probs_cache[b], probs_cache[b],
                    )
            row["models"] = model_blocks
            row["pairs"] = pair_blocks
            rows.append(row)
            obs, _r, dones, _i = env.step(np.asarray([tape_action]))
            if bool(np.asarray(dones).any()):
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    elapsed = time.time() - started_at
    header = {
        "_header": True,
        "tool": "tools/h5_logit_compare.py",
        "config": str(args.config),
        "eval_seed": int(args.eval_seed),
        "max_steps": int(args.max_steps),
        "behavior_tape": args.behavior_tape,
        "models": fingerprints,
        "elapsed_seconds": float(elapsed),
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    summary = aggregate(rows, labels)
    summary["_header"] = header
    summary["n_steps_logged"] = len(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)

    print(f"phase_h_logit_compare steps={len(rows)} elapsed={elapsed:.1f}s")
    for label in labels:
        m = summary["per_model"][label]
        print(
            f"  {label}: argmax={m['argmax_fractions']} "
            f"entropy_mean={m['entropy']['mean']:.4f} "
            f"margin_mean={m['top1_top2_margin']['mean']:.4f} "
            f"margin_min={m['top1_top2_margin']['min']:.4f}"
        )
    for key, p in summary["pairs"].items():
        print(
            f"  pair {key}: same_argmax={p['same_argmax_fraction']:.4f} "
            f"logit_l2_mean={p['centered_logit_l2']['mean']:.4f} "
            f"prob_l1_mean={p['prob_l1']['mean']:.4f} "
            f"sym_kl_mean={p['sym_kl']['mean']:.4f}"
        )
    print(
        f"  obs_unique={summary['obs']['unique_hash_count']}/"
        f"{summary['obs']['n_steps']} max_repeat_run="
        f"{summary['obs']['max_consecutive_repeat_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
