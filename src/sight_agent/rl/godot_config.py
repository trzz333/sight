"""H3 step 8 + H4 step 6: config -> factory plumbing helpers for the Godot branch.

Resolves Godot-specific env config (``godot_executable``, ``project_path``)
from a loaded RL config dict using this precedence:

    explicit YAML value > env var (``SIGHT_GODOT_EXE`` /
    ``SIGHT_GODOT_PROJECT``) > (project_path only) factory default
    (``games/signal-dodge`` relative to the repo root)

Relative ``env.project_path`` values in YAML are resolved against the repo
root so ``project_path: games/signal-dodge`` works regardless of the
trainer's current working directory.

H4 step 6 adds optional passthrough for env-construction kwargs that the
H4 pixel config needs to reach ``GodotSignalDodgeEnv``: ``max_steps``,
``headless``, ``observation_mode``, ``pixel_width``, ``pixel_height``,
``pixel_channels``. The resolver only includes these keys when they are
present in the YAML, so H3 configs that omit them stay byte-shape
identical to the H3 era at this layer. The factory layer is responsible
for dropping any ``None`` value before calling the env constructor (see
``factories._make_godot_signal_dodge_v0``).

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


_OPTIONAL_ENV_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "max_steps",
    "headless",
    "observation_mode",
    "pixel_width",
    "pixel_height",
    "pixel_channels",
)


def resolve_godot_kwargs(
    cfg: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the Godot kwargs to pass to ``factories.make_env``.

    Returns ``{}`` for non-Godot configs; callers can splat the result
    unconditionally without leaking Godot kwargs into the Gymnasium path.

    For Godot configs, always returns ``godot_executable`` and
    ``project_path`` where each value is either an absolute string path or
    ``None``. ``None`` for ``godot_executable`` means neither the YAML nor
    ``SIGHT_GODOT_EXE`` were set; the factory will surface that as a clear
    ``ValueError`` when the Godot branch runs. ``None`` for
    ``project_path`` means neither the YAML nor ``SIGHT_GODOT_PROJECT``
    were set; the factory falls back to its repo-root-relative default in
    that case.

    Additionally threads optional H4-era env-construction kwargs through
    when they are present in the YAML under ``env``: ``max_steps``,
    ``headless``, ``observation_mode``, ``pixel_width``, ``pixel_height``,
    ``pixel_channels``. The resolver does not invent defaults; if the
    YAML omits a key, the resolver omits it too. H3 configs that omit
    these keys keep their H3-era kwargs shape at this layer (apart from
    ``max_steps``, which the H3 YAML has always set). The factory layer
    drops any ``None`` value before calling the env constructor so a YAML
    that explicitly sets ``max_steps: null`` cannot override the env's
    own default with ``None``.
    """
    env_cfg = cfg.get("env", {})
    if not isinstance(env_cfg, dict) or not is_godot_env_id(env_cfg.get("id")):
        return {}
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    out: dict[str, Any] = {
        "godot_executable": _resolve_executable(env_cfg.get("godot_executable")),
        "project_path": _resolve_project_path(env_cfg.get("project_path"), root),
    }
    for key in _OPTIONAL_ENV_PASSTHROUGH_KEYS:
        if key in env_cfg:
            out[key] = env_cfg[key]
    return out


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
