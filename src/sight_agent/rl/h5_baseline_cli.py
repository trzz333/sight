"""H5 baseline CLI: live negative-control non-saturation gate entrypoint.

Durable command-line driver for ``run_h5_baseline``. Reuses the existing
H4 plumbing (``load_config``, ``resolve_godot_kwargs``, ``make_env``) so
Godot routing is decided in exactly one place. Writes the same evaluation
artifacts as the library (``<out_dir>/<run_name>/<run_id>/evaluation/``)
and records the canonical thresholds plus saturation decision under
``index.json``.

Two modes:

    --mode negative-controls   (default)
        Evaluates only stay_only, seeded_random, untrained_cnn. Rejects
        trained_cnn with a clear error. This is the pre-training H5
        non-saturation gate per ``docs/sight-h5-plan.md`` section 5.

    --mode full
        Allows trained_cnn as well. Requires --train-run-dir when
        trained_cnn is included. Used after the H5 training slice
        produces a model.zip; not part of the pre-training gate.

Examples (cmd.exe quoting; PowerShell users substitute backticks):

    set SIGHT_GODOT_EXE=C:\\path\\to\\Godot_v4.6.2-stable_win64.exe
    python -m sight_agent.rl.h5_baseline_cli ^
        --config configs/rl/signal_dodge_ppo_h4_pixel.yaml ^
        --run-id h5_negative_controls_smoke ^
        --seeds 1000,1001 ^
        --mode negative-controls

    python -m sight_agent.rl.h5_baseline_cli ^
        --config configs/rl/signal_dodge_ppo_h4_pixel.yaml ^
        --run-id h5_negative_controls_16seed ^
        --seeds 1000-1015 ^
        --mode negative-controls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .factories import make_env
from .godot_config import resolve_godot_kwargs
from .h5_baseline import (
    NEGATIVE_CONTROL_POLICIES,
    SeededRandomPolicy,
    StayOnlyPolicy,
    build_trained_cnn_policy,
    build_untrained_cnn_policy,
    canonical_non_saturation_thresholds,
    run_h5_baseline,
)
from .ndjson_logger import get_short_git_commit


VALID_POLICY_NAMES: tuple[str, ...] = (
    "stay_only",
    "seeded_random",
    "untrained_cnn",
    "trained_cnn",
)


class H5CLIError(ValueError):
    """Raised when CLI arguments are invalid or incompatible."""


def _repo_root_from_here() -> Path:
    """Repo root inferred from this file's location."""
    return Path(__file__).resolve().parents[3]


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed spec into an ordered list of integers.

    Accepts comma-separated values, inclusive ranges, and mixes of both:

        ``"1000,1001"``        -> ``[1000, 1001]``
        ``"1000-1015"``        -> ``[1000, 1001, ..., 1015]``
        ``"1000,1003-1005"``   -> ``[1000, 1003, 1004, 1005]``

    Duplicates are preserved (callers can dedupe if desired). The order
    of the input is preserved. Empty tokens or unparseable values raise
    ``H5CLIError`` with a message naming the offending token.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise H5CLIError("seeds spec must be a non-empty string")
    out: list[int] = []
    for raw_tok in spec.split(","):
        tok = raw_tok.strip()
        if not tok:
            raise H5CLIError(f"empty seed token in {spec!r}")
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise H5CLIError(f"malformed seed range token: {tok!r}")
            try:
                lo = int(parts[0].strip())
                hi = int(parts[1].strip())
            except ValueError as exc:
                raise H5CLIError(
                    f"non-integer in seed range token {tok!r}: {exc}"
                ) from exc
            if hi < lo:
                raise H5CLIError(
                    f"seed range {tok!r} has hi < lo (got {lo}..{hi})"
                )
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(tok))
            except ValueError as exc:
                raise H5CLIError(
                    f"non-integer seed token {tok!r}: {exc}"
                ) from exc
    return out


def resolve_policies_for_mode(
    mode: str,
    requested: list[str] | None,
) -> list[str]:
    """Return the policy names to evaluate for ``mode``.

    Defaults per mode:
        negative-controls -> NEGATIVE_CONTROL_POLICIES exactly
        full              -> NEGATIVE_CONTROL_POLICIES + ("trained_cnn",)

    Explicit ``requested`` lists override the default but are validated:
        - all entries must be in VALID_POLICY_NAMES
        - in negative-controls mode, trained_cnn is rejected
        - in full mode, trained_cnn is allowed
    Order is preserved from the requested list (or canonical when default).
    """
    if mode not in ("negative-controls", "full"):
        raise H5CLIError(
            f"mode must be 'negative-controls' or 'full', got {mode!r}"
        )
    if requested is None:
        if mode == "negative-controls":
            return list(NEGATIVE_CONTROL_POLICIES)
        return list(NEGATIVE_CONTROL_POLICIES) + ["trained_cnn"]
    unknown = [p for p in requested if p not in VALID_POLICY_NAMES]
    if unknown:
        raise H5CLIError(
            f"unknown policy name(s): {unknown}. "
            f"Valid: {list(VALID_POLICY_NAMES)}"
        )
    if mode == "negative-controls" and "trained_cnn" in requested:
        raise H5CLIError(
            "trained_cnn is not permitted in negative-controls mode; "
            "use --mode full and provide --train-run-dir to evaluate it. "
            "Pre-training non-saturation gate uses three negative controls only."
        )
    return list(requested)


def build_dummy_vec_env_for_cfg(cfg: dict[str, Any]) -> Any:
    """Build a lightweight DummyVecEnv whose spaces match the YAML env config.

    Used to construct an SB3 PPO CnnPolicy (``build_untrained_cnn_policy``)
    without launching Godot. The model's ``predict`` path consumes obs
    directly; the env passed at construction time only fixes the network
    input shape. As long as this dummy env's ``observation_space`` and
    ``action_space`` match the Godot env's, ``model.predict(obs)`` will
    accept the live Godot obs at rollout time.

    Matches the H4 pixel contract: ``Box(0, 255, (C, H, W), uint8)`` and
    ``Discrete(3)``. Reads pixel dimensions from the env config block,
    defaulting to 1x84x84 per ``docs/sight-h4-plan.md`` section 1.
    """
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    ch = int(env_cfg.get("pixel_channels", 1))
    h = int(env_cfg.get("pixel_height", 84))
    w = int(env_cfg.get("pixel_width", 84))

    class _DummyPixelEnv(gym.Env):
        metadata: dict = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(ch, h, w), dtype=np.uint8
            )
            self.action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):  # type: ignore[override]
            super().reset(seed=seed)
            return np.zeros((ch, h, w), dtype=np.uint8), {}

        def step(self, action):  # type: ignore[override]
            return (
                np.zeros((ch, h, w), dtype=np.uint8),
                0.0,
                False,
                True,
                {},
            )

    return DummyVecEnv([_DummyPixelEnv])


def build_policies_dict(
    *,
    policy_names: list[str],
    cfg: dict[str, Any],
    train_run_dir: Path | None,
) -> dict[str, Any]:
    """Construct the ``policies`` dict consumed by ``run_h5_baseline``.

    All policies are constructed up front. The untrained_cnn branch uses a
    lightweight dummy VecEnv (``build_dummy_vec_env_for_cfg``) so no
    Godot process is spawned solely for PPO construction. The trained_cnn
    branch loads ``model.zip`` from ``train_run_dir`` via
    ``build_trained_cnn_policy(env=None)``; SB3's PPO.load tolerates
    ``env=None`` for inference-only use, and the rollout layer attaches
    obs at predict time.
    """
    seed = int(cfg.get("run", {}).get("seed", 0))
    out: dict[str, Any] = {}
    for name in policy_names:
        if name == "stay_only":
            out[name] = StayOnlyPolicy()
        elif name == "seeded_random":
            out[name] = SeededRandomPolicy()
        elif name == "untrained_cnn":
            dummy = build_dummy_vec_env_for_cfg(cfg)
            try:
                out[name] = build_untrained_cnn_policy(dummy, seed=seed)
            finally:
                try:
                    dummy.close()
                except Exception:
                    pass
        elif name == "trained_cnn":
            if train_run_dir is None:
                raise H5CLIError(
                    "trained_cnn requires --train-run-dir pointing at a "
                    "completed train run with a model.zip"
                )
            out[name] = build_trained_cnn_policy(train_run_dir, env=None)
        else:
            raise H5CLIError(f"unknown policy name: {name!r}")
    return out


def build_env_factory_for_policy(
    cfg: dict[str, Any],
    run_dir: Path,
) -> Callable[[str], Any]:
    """Return a function that constructs a fresh Godot VecEnv per policy.

    The returned ``factory(policy_name)`` calls ``make_env`` with the
    Godot kwargs resolved from ``cfg`` and a per-policy ``godot-eval-
    <policy_name>`` sub-directory under ``run_dir`` so each policy's
    Godot NDJSON evidence lands in its own directory and TCP ports are
    re-allocated per construction.
    """
    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)

    def _factory(policy_name: str) -> Any:
        per_policy_run_dir = run_dir / f"godot-eval-{policy_name}"
        if godot_extra:
            return make_env(
                env_id,
                n_envs=1,
                seed=base_seed,
                mode="eval",
                run_dir=str(per_policy_run_dir),
                **godot_extra,
            )
        return make_env(env_id, n_envs=1, seed=base_seed, mode="eval")

    return _factory


def resolve_run_dir(
    cfg: dict[str, Any],
    run_id: str,
    out_dir_override: str | None,
) -> Path:
    """Resolve ``<out_dir>/<run_name>/<run_id>/`` for the eval run.

    Out-dir precedence: ``--out-dir`` flag > ``run.out_dir`` from YAML.
    Mirrors ``prepare_train_artifacts`` layout so eval runs sit next to
    train runs of the same name.
    """
    name = cfg.get("run", {}).get("name")
    if not isinstance(name, str) or not name.strip():
        raise H5CLIError("config run.name must be a non-empty string")
    if out_dir_override and out_dir_override.strip():
        out_root = Path(out_dir_override)
    else:
        out_root_raw = cfg.get("run", {}).get("out_dir")
        if not isinstance(out_root_raw, str) or not out_root_raw.strip():
            raise H5CLIError(
                "config run.out_dir must be set or --out-dir provided"
            )
        out_root = Path(out_root_raw)
    return out_root / name / run_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sight_agent.rl.h5_baseline_cli",
        description=(
            "H5 baseline / non-saturation evaluation CLI. "
            "Default mode evaluates the three negative controls only."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--run-id", required=True, dest="run_id",
        help="Identifier for this eval run (filesystem-safe).",
    )
    parser.add_argument(
        "--seeds", required=True,
        help="Comma list and/or inclusive ranges, e.g. '1000,1001' or '1000-1015'.",
    )
    parser.add_argument(
        "--policies", default=None,
        help="Comma-separated policy names. Default depends on --mode.",
    )
    parser.add_argument(
        "--mode", default="negative-controls",
        choices=("negative-controls", "full"),
        help="negative-controls (pre-training gate) or full (allows trained_cnn).",
    )
    parser.add_argument(
        "--train-run-dir", default=None, dest="train_run_dir",
        help="Required when evaluating trained_cnn; path to a completed train run.",
    )
    parser.add_argument(
        "--out-dir", default=None, dest="out_dir",
        help="Override run.out_dir from the YAML.",
    )
    return parser


def run_cli(
    *,
    config_path: str,
    run_id: str,
    seeds_spec: str,
    mode: str,
    requested_policies: list[str] | None,
    train_run_dir: str | None,
    out_dir_override: str | None,
) -> dict[str, Any]:
    """Execute one H5 baseline eval run from CLI args. Returns the index dict.

    Pure-Python entrypoint that ``main()`` wraps. Tests exercise this path
    directly with monkeypatched ``make_env`` and ``build_dummy_vec_env_for_cfg``
    so no real Godot launch is required.
    """
    seeds = parse_seeds(seeds_spec)
    if not seeds:
        raise H5CLIError("--seeds must yield at least one seed")
    policy_names = resolve_policies_for_mode(mode, requested_policies)

    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("id")
    max_steps = int(env_cfg.get("max_steps", 1800))
    observation_mode = env_cfg.get("observation_mode", "state")
    run_dir = resolve_run_dir(cfg, run_id, out_dir_override)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_run_path = Path(train_run_dir) if train_run_dir else None
    policies = build_policies_dict(
        policy_names=policy_names,
        cfg=cfg,
        train_run_dir=train_run_path,
    )
    env_factory = build_env_factory_for_policy(cfg, run_dir)
    git_commit = get_short_git_commit(_repo_root_from_here())

    index = run_h5_baseline(
        run_dir=run_dir,
        env_id=str(env_id),
        observation_mode=str(observation_mode),
        max_steps=max_steps,
        seeds=seeds,
        env_factory_for_policy=env_factory,
        policies=policies,
        git_commit=git_commit,
        deterministic=True,
        thresholds=canonical_non_saturation_thresholds(),
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    requested = (
        [p.strip() for p in args.policies.split(",") if p.strip()]
        if args.policies
        else None
    )
    try:
        index = run_cli(
            config_path=args.config,
            run_id=args.run_id,
            seeds_spec=args.seeds,
            mode=args.mode,
            requested_policies=requested,
            train_run_dir=args.train_run_dir,
            out_dir_override=args.out_dir,
        )
    except H5CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    decision = index.get("saturation_decision", {})
    passed = bool(decision.get("passed", False))
    sat_list = decision.get("saturated_negative_controls", [])
    print(f"H5 eval run_id={args.run_id} mode={args.mode}")
    print(f"  passed={passed} saturated_negative_controls={sat_list}")
    print(f"  artifacts: {index['policy_summaries']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
