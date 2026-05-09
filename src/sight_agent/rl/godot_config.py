"""H3 step 8: config -> factory plumbing helpers for the Godot branch.

Resolves Godot-specific env config (``godot_executable``, ``project_path``)
from a loaded RL config dict using this precedence:

    explicit YAML value > env var (``SIGHT_GODOT_EXE`` /
    ``SIGHT_GODOT_PROJECT``) > (project_path only) factory default
    (``games/signal-dodge`` relative to the repo root)

Relative ``env.project_path`` values in YAML are resolved against the repo
root so ``project_path: games/signal-dodge`` works regardless of the
trainer's current working directory.

Pure: no factory or stable_baselines3 imports. Both ``train.py`` and
``evaluate.py`` import this so Godot routing is decided in exactly one
place at the plumbing layer rather than being re-derived in two.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


GODOT_SIGNAL_DODGE_V0 = "godot:signal-dodge-v0"


def is_godot_env_id(env_id: Any) -> bool:
    """True iff ``env_id`` is the H3 Signal Dodge env id (exact match).

    Other ``godot:`` prefixed ids are not routed in H3; the factory rejects
    them downstream with the unsupported-env_id message.
    """
    return isinstance(env_id, str) and env_id == GODOT_SIGNAL_DODGE_V0


def resolve_godot_kwargs(
    cfg: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, str | None]:
    """Build the Godot kwargs to pass to ``factories.make_env``.

    Returns ``{}`` for non-Godot configs; callers can splat the result
    unconditionally without leaking Godot kwargs into the Gymnasium path.

    For Godot configs, returns ``{"godot_executable": ..., "project_path": ...}``
    where each value is either an absolute string path or ``None``. ``None``
    for ``godot_executable`` means neither the YAML nor ``SIGHT_GODOT_EXE``
    were set; the factory will surface that as a clear ``ValueError`` when
    the Godot branch runs. ``None`` for ``project_path`` means neither the
    YAML nor ``SIGHT_GODOT_PROJECT`` were set; the factory falls back to
    its repo-root-relative default in that case.
    """
    env_cfg = cfg.get("env", {})
    if not isinstance(env_cfg, dict) or not is_godot_env_id(env_cfg.get("id")):
        return {}
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    return {
        "godot_executable": _resolve_executable(env_cfg.get("godot_executable")),
        "project_path": _resolve_project_path(env_cfg.get("project_path"), root),
    }


def _resolve_executable(yaml_value: Any) -> str | None:
    if isinstance(yaml_value, str) and yaml_value.strip():
        return yaml_value
    env_value = os.environ.get("SIGHT_GODOT_EXE")
    return env_value if env_value else None


def _resolve_project_path(yaml_value: Any, repo_root: Path) -> str | None:
    if isinstance(yaml_value, str) and yaml_value.strip():
        candidate: str | None = yaml_value
    else:
        env_value = os.environ.get("SIGHT_GODOT_PROJECT")
        candidate = env_value if env_value else None
    if candidate is None:
        return None
    p = Path(candidate)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return str(p)


def _default_repo_root() -> Path:
    """Repo root inferred from this file's location.

    ``godot_config.py`` lives at ``<repo>/src/sight_agent/rl/godot_config.py``
    so the repo root is four parents up.
    """
    return Path(__file__).resolve().parents[3]
