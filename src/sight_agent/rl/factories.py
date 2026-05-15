"""H2 factory seams for env and algorithm construction.

Two responsibilities:
1. env factory: build a VecEnv for train or eval, dispatched by env_id. For H2
   only Gymnasium env ids are supported. The dispatch is a single function,
   make_env, so H3 can add a Godot state-env branch without touching train or
   evaluate.
2. algo factory: build an SB3 PPO model from a (validated) config. For H2 the
   only supported (framework, name) pair is ('stable-baselines3', 'PPO'). Any
   other pair is rejected with a clear error.

Both functions raise ValueError with explicit messages naming what is supported
when given an unsupported request.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack


# H3 step 7: exact env id routed through the Godot branch. Other ``godot:``
# prefixed ids fall through to the unsupported-env_id error so unknown
# Godot games cannot quietly slip in.
GODOT_SIGNAL_DODGE_V0 = "godot:signal-dodge-v0"


SUPPORTED_FRAMEWORKS = ("stable-baselines3",)
SUPPORTED_ALGOS_BY_FRAMEWORK = {
    "stable-baselines3": ("PPO",),
}


def make_env(
    env_id: str,
    n_envs: int,
    seed: int,
    mode: str = "train",
    *,
    run_dir: str | os.PathLike[str] | None = None,
    godot_executable: str | os.PathLike[str] | None = None,
    project_path: str | os.PathLike[str] | None = None,
    max_steps: int | None = None,
    headless: bool | None = None,
    observation_mode: str | None = None,
    pixel_width: int | None = None,
    pixel_height: int | None = None,
    pixel_channels: int | None = None,
    frame_stack: int | None = None,
) -> VecEnv:
    """Build a VecEnv for train or eval.

    Dispatches by env_id. Supports Gymnasium env ids (anything that
    ``gymnasium.make`` accepts) and the H3 Godot env ``godot:signal-dodge-v0``.
    Other ``godot:`` prefixed ids are rejected: H3 routes the signal-dodge
    env only.

    Args:
        env_id: env identifier (e.g. ``CartPole-v1``, ``godot:signal-dodge-v0``).
        n_envs: number of parallel envs (must be >= 1). For
            ``godot:signal-dodge-v0`` only ``n_envs=1`` is accepted in H3.
        seed: base seed. Eval mode uses ``seed + 10_000`` so eval and train
            seedings are disjoint by construction.
        mode: ``train`` or ``eval``.
        run_dir: keyword-only. Pass-through to ``GodotSignalDodgeEnv``;
            ignored for Gymnasium ids. The factory does not invent
            train/eval suffixes; logging is owned by the env.
        godot_executable: keyword-only. Path to the Godot binary. If unset,
            ``SIGHT_GODOT_EXE`` is consulted. Required for the Godot branch.
        project_path: keyword-only. Path to the Godot project. If unset,
            ``SIGHT_GODOT_PROJECT`` is consulted, then the repo-root-relative
            ``games/signal-dodge`` is used as a final fallback.
        max_steps, headless, observation_mode, pixel_width, pixel_height,
        pixel_channels: keyword-only optional pass-throughs forwarded to
            ``GodotSignalDodgeEnv`` only when non-None. Ignored for
            Gymnasium ids. ``None`` means "let the env constructor pick
            its default," preserving H3-era behavior when these are
            omitted by the resolver.
        frame_stack: keyword-only. When set to an integer >= 2 AND
            ``observation_mode == "pixel"``, the constructed VecEnv is
            wrapped in SB3's ``VecFrameStack`` with channels-first
            ordering so the policy sees temporal context. ``None`` and
            ``1`` are both no-ops (no wrapper applied), preserving the
            Phase D/E single-frame contract byte-shape. State-mode
            observations are never frame-stacked by this factory; that
            is not the H5 plan.
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    if int(n_envs) < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    if env_id == GODOT_SIGNAL_DODGE_V0:
        vec_env = _make_godot_signal_dodge_v0(
            n_envs=int(n_envs),
            seed=int(seed),
            mode=mode,
            run_dir=run_dir,
            godot_executable=godot_executable,
            project_path=project_path,
            max_steps=max_steps,
            headless=headless,
            observation_mode=observation_mode,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            pixel_channels=pixel_channels,
        )
    elif _looks_like_gymnasium(env_id):
        if mode == "train":
            vec_env = make_vec_env(env_id, n_envs=int(n_envs), seed=int(seed))
        else:
            vec_env = make_vec_env(env_id, n_envs=1, seed=int(seed) + 10_000)
    else:
        raise ValueError(
            f"Unsupported env_id: {env_id!r}. Sight H2 supports Gymnasium env ids only. "
            f"Future phases will add additional env families via this factory."
        )

    # H5 Phase F: optional pixel-mode frame stacking via SB3's
    # ``VecFrameStack``. Applied uniformly to train and eval VecEnvs so
    # train, in-training eval, and H5 eval CLI all expose the same stacked
    # observation contract; without parity, ``model.predict`` on the
    # trained policy would silently see a mismatched obs shape at eval.
    # Channel-first ordering matches the Godot pixel contract
    # ``(C, H, W)`` per ``docs/sight-h4-plan.md`` section 1. Only fires
    # when explicitly requested and only for pixel observations; state
    # observations are not stacked here (SB3's VecFrameStack on a 1-d Box
    # would produce a non-sensical shape and is not the H5 plan).
    if (
        observation_mode == "pixel"
        and frame_stack is not None
        and int(frame_stack) > 1
    ):
        vec_env = VecFrameStack(
            vec_env, n_stack=int(frame_stack), channels_order="first"
        )

    return vec_env


def smoke_check_env(env_id: str, seed: int) -> tuple[tuple[int, ...], int]:
    """Single-reset smoke for run_start observability."""
    env = gym.make(env_id)
    obs, _info = env.reset(seed=int(seed))
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(seed))
    obs_shape = tuple(getattr(obs, "shape", ()))
    action_n = int(getattr(env.action_space, "n", 0))
    env.close()
    return obs_shape, action_n


def make_algo(
    framework: str,
    name: str,
    policy: str,
    device: str,
    hyperparams: dict[str, Any],
    env: VecEnv,
    seed: int,
) -> Any:
    """Build an algorithm instance from a (framework, name) pair.

    Raises ValueError with a clear, enumerated message if the combination is
    not in SUPPORTED_ALGOS_BY_FRAMEWORK. H2 supports SB3 PPO only.
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework: {framework!r}. Sight H2 supports only "
            f"frameworks {list(SUPPORTED_FRAMEWORKS)}."
        )
    allowed = SUPPORTED_ALGOS_BY_FRAMEWORK[framework]
    if name not in allowed:
        raise ValueError(
            f"Unsupported algo {name!r} for framework {framework!r}. "
            f"Allowed algos for this framework: {list(allowed)}."
        )

    if framework == "stable-baselines3" and name == "PPO":
        kwargs: dict[str, Any] = {
            "policy": policy,
            "env": env,
            "seed": int(seed),
            "device": device,
        }
        if hyperparams:
            kwargs.update(hyperparams)
        return PPO(**kwargs)

    # Defensive: should be unreachable given the checks above.
    raise ValueError(
        f"No factory branch matched (framework={framework!r}, name={name!r})."
    )


def _looks_like_gymnasium(env_id: str) -> bool:
    """Very loose heuristic: H2 treats every non-Godot id as Gymnasium.

    Future H3 will add an explicit ``godot:`` (or similar) prefix branch in
    make_env. Until then, anything not flagged as a future env family is sent
    to Gymnasium and Gymnasium raises if it does not recognize the id.
    """
    if not isinstance(env_id, str) or not env_id:
        return False
    lowered = env_id.lower()
    if lowered.startswith("godot:") or lowered.startswith("godot/"):
        return False
    return True


def _make_godot_signal_dodge_v0(
    *,
    n_envs: int,
    seed: int,
    mode: str,
    run_dir: str | os.PathLike[str] | None,
    godot_executable: str | os.PathLike[str] | None,
    project_path: str | os.PathLike[str] | None,
    max_steps: int | None = None,
    headless: bool | None = None,
    observation_mode: str | None = None,
    pixel_width: int | None = None,
    pixel_height: int | None = None,
    pixel_channels: int | None = None,
) -> VecEnv:
    """H3 step 7 + H4 step 6: build a single-env ``DummyVecEnv`` wrapping ``GodotSignalDodgeEnv``.

    Lazy-imports the env to keep Gymnasium-only training paths from importing
    the Godot transport. Resolves Godot paths with the precedence:

        explicit kwarg > ``SIGHT_GODOT_EXE`` / ``SIGHT_GODOT_PROJECT`` env var
        > (project_path only) repo-root-relative ``games/signal-dodge``

    Env vars do not override explicit kwargs. The factory does not parse YAML;
    that is the plumbing layer's job (``godot_config.resolve_godot_kwargs``).

    ``run_dir`` is passed through to the env unchanged. The factory does not
    add train/eval suffixes; episode evidence files are owned by the env.

    Eval seeding mirrors the Gymnasium branch: ``seed`` for train, ``seed +
    10_000`` for eval. The effective seed is threaded both into the env
    constructor (used at first ``reset()``) and through ``VecEnv.seed``.

    H4 optional kwargs (``max_steps``, ``headless``, ``observation_mode``,
    ``pixel_width``, ``pixel_height``, ``pixel_channels``) are forwarded to
    the env constructor only when non-None. ``None`` means "let the env
    pick its default," preserving H3-era construction shape when the
    resolver omitted the key. This also means a YAML that explicitly sets
    ``headless: null`` cannot override the env default with ``None``.
    """
    if n_envs != 1:
        raise ValueError(
            f"env_id='{GODOT_SIGNAL_DODGE_V0}' supports n_envs=1 only in H3; "
            f"vectorized parallel Godot envs are explicitly out of scope. "
            f"Got n_envs={n_envs}."
        )

    exe = godot_executable if godot_executable is not None else os.environ.get("SIGHT_GODOT_EXE")
    if not exe:
        raise ValueError(
            f"godot_executable not provided and SIGHT_GODOT_EXE env var is not "
            f"set; pass godot_executable=... to make_env or set SIGHT_GODOT_EXE "
            f"for env_id='{GODOT_SIGNAL_DODGE_V0}'."
        )

    if project_path is not None:
        proj: str | os.PathLike[str] = project_path
    else:
        env_proj = os.environ.get("SIGHT_GODOT_PROJECT")
        proj = env_proj if env_proj else _default_signal_dodge_project_path()

    # Lazy import: keep Gymnasium-only paths from pulling in Godot transport.
    from .godot_env import GodotSignalDodgeEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    effective_seed = int(seed) if mode == "train" else int(seed) + 10_000

    # Each factory-built Godot env gets its own kernel-assigned loopback
    # TCP port so train and in-train eval can run simultaneously without
    # colliding on the historical 127.0.0.1:8765 default. Discovered by
    # the H4 live smoke: with eval_freq=64 the SB3 callback resets the
    # eval env while the train Godot subprocess still owns 8765, and
    # Godot's tcp_controller.gd surfaces a WSAEINVAL (22) on listen()
    # before the eval env's recv_timeout expires. Allocating per env at
    # construction time is the smallest correct fix; tests that need a
    # specific port still bypass the factory and construct
    # ``GodotSignalDodgeEnv`` directly with ``tcp_port=...``.
    tcp_port = _allocate_isolated_tcp_port()

    # Filter out None values so we never override the env constructor's own
    # default with a literal None. The H3-era call shape is exactly
    # ``{godot_executable, project_path, run_dir, seed, tcp_port}``; when all H4
    # optional kwargs are omitted by the resolver this set is unchanged.
    extra_env_kwargs: dict[str, Any] = {}
    for _name, _val in (
        ("max_steps", max_steps),
        ("headless", headless),
        ("observation_mode", observation_mode),
        ("pixel_width", pixel_width),
        ("pixel_height", pixel_height),
        ("pixel_channels", pixel_channels),
    ):
        if _val is not None:
            extra_env_kwargs[_name] = _val

    def _factory() -> GodotSignalDodgeEnv:
        return GodotSignalDodgeEnv(
            godot_executable=exe,
            project_path=proj,
            run_dir=run_dir,
            seed=effective_seed,
            tcp_port=tcp_port,
            **extra_env_kwargs,
        )

    vec_env = DummyVecEnv([_factory])
    # Mirror ``make_vec_env`` seeding behaviour. ``DummyVecEnv.seed`` only
    # touches action_space and stores the seed for the next reset; it does
    # not invoke ``env.reset()`` on its own, so this stays lazy w.r.t. the
    # Godot subprocess.
    try:
        vec_env.seed(effective_seed)
    except Exception:
        # Some VecEnv subclasses may raise on seed; do not block construction
        # on a non-load-bearing seed propagation path.
        pass
    return vec_env


def _allocate_isolated_tcp_port() -> int:
    """Bind ``127.0.0.1:0``, capture the kernel-assigned port, then release.

    Mirrors ``tests/rl/test_h3_godot_smoke._allocate_isolated_tcp_port``.
    The TOCTOU window between releasing the socket and Godot binding the
    same port is negligible on loopback inside one dev box; the env's
    ``connect_timeout_s`` retry covers transient races. The same caller
    must NOT cache and reuse the port for a second simultaneous env;
    each Godot env instance gets its own freshly-allocated port so
    train and in-train eval cannot collide on 127.0.0.1:8765 (the
    historical default that broke H4 live smoke until this slice).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _default_signal_dodge_project_path() -> str:
    """Repo-root-relative ``games/signal-dodge``.

    ``factories.py`` lives at ``<repo>/src/sight_agent/rl/factories.py`` so the
    repo root is four parents up. The path is returned as a string and is not
    validated; the env will surface any missing-path error when Godot launches.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent
    return str(repo_root / "games" / "signal-dodge")
