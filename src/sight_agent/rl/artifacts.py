"""H2 run-artifact paths and config snapshot helpers.

A train run produces:
    <out_root>/<run_name>/<run_id>/
        events.ndjson
        summary.json
        config_effective.yaml
        model.zip                 (when checkpoint is enabled)

An eval against an existing train run produces:
    <out_root>/<run_name>/<run_id>/evals/<eval_id>/
        events.ndjson
        summary.json

This module centralizes path layout and the config_hash so train.py and
evaluate.py do not redefine it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainArtifacts:
    """Filesystem paths for a single train run."""

    run_id: str
    run_dir: Path
    events_path: Path
    summary_path: Path
    config_effective_path: Path
    model_path: Path  # populated even if checkpoint disabled; not written until save


@dataclass(frozen=True)
class EvalArtifacts:
    """Filesystem paths for a single eval against an existing train run."""

    eval_id: str
    eval_dir: Path
    events_path: Path
    summary_path: Path
    source_train_run_dir: Path


def build_run_id(name: str, seed: int, override: str | None, git_commit: str | None) -> str:
    """Stable run-id format: <utc-ts>_<name>_seed<seed>_<git>."""
    if override:
        return override
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    git_part = git_commit or "nogit"
    return f"{ts}_{name}_seed{seed}_{git_part}"


def build_eval_id(seed: int, n_eval_episodes: int, source_run_id: str | None = None) -> str:
    """Eval-id format: eval_<utc-ts>_seed<seed>_n<n>[_<src8>]."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    src_part = ""
    if source_run_id:
        # Take a short, filesystem-safe slice of the source run id for traceability.
        safe = "".join(ch for ch in source_run_id if ch.isalnum())
        if safe:
            src_part = f"_{safe[-8:]}"
    return f"eval_{ts}_seed{seed}_n{n_eval_episodes}{src_part}"


def prepare_train_artifacts(cfg: dict[str, Any], run_id: str) -> TrainArtifacts:
    """Build (and create) the directory layout for a train run."""
    out_root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    return TrainArtifacts(
        run_id=run_id,
        run_dir=out_root,
        events_path=out_root / "events.ndjson",
        summary_path=out_root / "summary.json",
        config_effective_path=out_root / "config_effective.yaml",
        model_path=out_root / _model_filename(cfg),
    )


def prepare_eval_artifacts(train_run_dir: Path, eval_id: str) -> EvalArtifacts:
    """Build (and create) the directory layout for an eval under a train run."""
    train_run_dir = Path(train_run_dir)
    eval_dir = train_run_dir / "evals" / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    return EvalArtifacts(
        eval_id=eval_id,
        eval_dir=eval_dir,
        events_path=eval_dir / "events.ndjson",
        summary_path=eval_dir / "summary.json",
        source_train_run_dir=train_run_dir,
    )


def write_config_effective(path: Path, cfg: dict[str, Any]) -> None:
    """Snapshot the effective (post-CLI-merge) config alongside artifacts."""
    Path(path).write_text(
        yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Stable sha256 over the canonical JSON form of cfg.

    Determinism: ``json.dumps(..., sort_keys=True, separators=(',', ':'))`` is
    a canonical serialization for JSON-compatible inputs. Same effective config
    -> same hash, across runs and machines, given the same Python and stdlib.
    """
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_checkpoint_enabled(cfg: dict[str, Any]) -> bool:
    """True if cfg has a checkpoint section with enabled: true."""
    ckpt = cfg.get("checkpoint")
    if not isinstance(ckpt, dict):
        return False
    return bool(ckpt.get("enabled", False))


def _model_filename(cfg: dict[str, Any]) -> str:
    ckpt = cfg.get("checkpoint")
    if isinstance(ckpt, dict):
        fname = ckpt.get("filename")
        if isinstance(fname, str) and fname:
            return fname
    return "model.zip"
