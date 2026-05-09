"""Gymnasium-compatible env wrapping ``GodotH3Transport`` for Signal Dodge.

Step 6 of ``docs/sight-h3-plan.md`` Implementation Sequence. Owns the Godot
subprocess lifecycle and the TCP connection lifecycle; defers all wire
shape, sequence numbering, and protocol validation to ``GodotH3Transport``.

Lifecycle model (locked here, see Decision 2 in the plan):
- One Godot subprocess across all episodes for this env instance.
- One persistent TCP connection across all episodes.
- ``reset()`` performs a soft reset through the transport. No subprocess
  relaunch per episode.
- ``__init__`` does NOT start Godot or open TCP. First ``reset()`` does.
- ``close()`` closes the transport, terminates the subprocess with a short
  grace window, then kills it if it has not exited. Idempotent.

Error model (mirrors the transport):
- ``GodotTransportError``, ``GodotProtocolError``, ``GodotRemoteError``
  raised by the transport propagate out of ``reset`` / ``step`` unchanged.
  A broken transport is NOT converted into ``terminated=True``; per plan
  section 5 a broken environment is not a terminal state.

Test seams:
- ``transport_factory`` and ``process_factory`` injectable for unit tests
  so the Gym surface, lifecycle, and info contract can be exercised
  without a real Godot binary on disk.

Wire-side env vars consumed by Godot (see games/signal-dodge/scripts):
- ``SIGHT_TCP_MODE=1``  : enables the TCP listener path in main.gd.
- ``SIGHT_TCP_PORT``    : overrides the default 8765.
- ``SIGHT_GODOT_LOG_PATH``: absolute path for godot.ndjson (logger.gd).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import IO, Any, Callable

import gymnasium as gym
import numpy as np

from .. import constants
from .. import protocol
from .godot_transport import (
    GodotH3Transport,
    GodotProtocolError,
    GodotRemoteError,
    GodotTransportError,
)


__all__ = ["GodotSignalDodgeEnv", "DEFAULT_MAX_STEPS"]


# Default episode budget. 30 seconds at 60 Hz. Plan and config (step 8) may
# override; the constructor accepts any positive int.
DEFAULT_MAX_STEPS: int = 1800

# Per-attempt TCP connect timeout. The env retries inside the larger
# ``connect_timeout_s`` budget while waiting for Godot to start listening.
_CONNECT_RETRY_INTERVAL_S: float = 0.1
_CONNECT_PER_ATTEMPT_TIMEOUT_S: float = 0.5

# Grace period between SIGTERM (Popen.terminate) and SIGKILL (Popen.kill)
# during ``close()``. Short on purpose; Godot has nothing to flush after the
# autoload logger's per-event flush.
_TERMINATE_GRACE_S: float = 2.0

# Filename for Python-side NDJSON evidence under ``run_dir``. Mirrors
# ``godot.ndjson`` produced by games/signal-dodge/scripts/logger.gd via
# the ``SIGHT_GODOT_LOG_PATH`` env var.
_PYTHON_NDJSON_NAME: str = "python.ndjson"


class _NdjsonWriter:
    """Append-only NDJSON writer with per-event flush.

    Minimal by design: one JSON object per line, UTF-8, newline-terminated,
    flushed after every write so a hard kill of the Python process does not
    truncate the latest event. ``write`` silently no-ops after ``close``.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: IO[str] | None = open(path, "a", encoding="utf-8", newline="")

    def write(self, event_type: str, fields: dict[str, Any]) -> None:
        if self._fh is None:
            return
        record: dict[str, Any] = {
            "ts_unix": time.time(),
            "type": event_type,
        }
        for k, v in fields.items():
            if v is not None:
                record[k] = v
        try:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._fh.flush()
        except (OSError, ValueError):
            # ValueError covers writes after the underlying file was closed
            # out from under us. Logging failures must never break the env.
            pass

    def close(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass


class GodotSignalDodgeEnv(gym.Env):
    """Gymnasium env for Godot Signal Dodge over the H3 TCP transport.

    Action space: ``Discrete(3)`` (0=left, 1=stay, 2=right).
    Observation space: ``Box(-1.0, 1.0, shape=(10,), dtype=float32)``.
    Reward: sparse survival per ``docs/sight-h3-plan.md`` section 4.
        - ``+1.0`` per non-terminal step
        - ``0.0`` on the collision terminal step
        - timeout produces ``truncated=True`` and the final-step reward is
          whatever Godot reports (no special bonus, see plan)

    Render mode: ``None`` (state-only, no pixels).

    Lifecycle:
        env = GodotSignalDodgeEnv(godot_executable=..., project_path=...)
        obs, info = env.reset(seed=0)         # launches Godot + connects
        obs, r, term, trunc, info = env.step(1)
        ...
        env.close()                            # idempotent

    Test seams:
        ``transport_factory(run_id, host, port, recv_timeout_s)`` returns a
        ``GodotH3Transport``-like object exposing ``connect``, ``send_hello``,
        ``reset``, ``step``, ``close``, plus ``episode_id`` property.
        ``process_factory(cmd, env=..., stdout=..., stderr=...)`` returns a
        ``Popen``-like object exposing ``pid``, ``poll``, ``terminate``,
        ``kill``, ``wait``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        godot_executable: str | os.PathLike[str],
        project_path: str | os.PathLike[str],
        tcp_host: str = constants.TCP_HOST,
        tcp_port: int = constants.TCP_PORT,
        run_dir: str | os.PathLike[str] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        connect_timeout_s: float = 10.0,
        step_timeout_s: float = 5.0,
        seed: int | None = None,
        headless: bool = True,
        transport_factory: Callable[..., GodotH3Transport] | None = None,
        process_factory: Callable[..., subprocess.Popen] | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError(f"max_steps must be positive int, got {max_steps!r}")
        if connect_timeout_s <= 0:
            raise ValueError(f"connect_timeout_s must be > 0, got {connect_timeout_s}")
        if step_timeout_s <= 0:
            raise ValueError(f"step_timeout_s must be > 0, got {step_timeout_s}")

        self._godot_executable = Path(godot_executable)
        self._project_path = Path(project_path)
        self._tcp_host = tcp_host
        self._tcp_port = int(tcp_port)
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._max_steps = int(max_steps)
        self._connect_timeout_s = float(connect_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        self._init_seed = seed
        self._headless = bool(headless)
        self._transport_factory = transport_factory or _default_transport_factory
        self._process_factory = process_factory or _default_process_factory

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Process / transport state. None until first reset().
        self._proc: subprocess.Popen | None = None
        self._transport: GodotH3Transport | None = None
        self._run_id: str = f"sight-h3-{uuid.uuid4().hex[:12]}"
        self._episode_count: int = 0
        self._episode_done: bool = True
        self._closed: bool = False
        # Diagnostic state. ``_launch_cmd`` is captured by ``_launch_godot``
        # and surfaced in early-exit error messages so the operator can see
        # exactly what was invoked when Godot died before listening.
        self._launch_cmd: list[str] = []
        # Godot stdout/stderr file handles. Opened by ``_launch_godot`` when
        # ``run_dir`` is set so post-mortem inspection can recover the
        # engine's stdio after a hang or crash. ``None`` when ``run_dir`` is
        # not supplied (DEVNULL is used instead). Closed by ``close()``.
        self._godot_stdout_file: IO[bytes] | None = None
        self._godot_stderr_file: IO[bytes] | None = None
        # Python-side NDJSON evidence writer. Opened lazily on first
        # ``_ensure_process_and_transport`` when ``run_dir`` is set; remains
        # ``None`` when ``run_dir`` is None so this slice has no effect for
        # callers that opt out of evidence capture.
        self._ndjson: _NdjsonWriter | None = None

    # --- public read-only accessors --------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def godot_pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def tcp_port(self) -> int:
        return self._tcp_port

    # --- Gymnasium API ---------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Soft-reset the active episode (lazy-launches Godot on first call).

        Per ``docs/sight-h3-plan.md`` section 6: gym parent seed handling
        first, then episode seed derivation, then process / TCP ensure,
        then reset wire exchange.
        """
        if self._closed:
            raise RuntimeError("reset called on closed env")

        super().reset(seed=seed)

        # Episode seed precedence: explicit reset(seed=...) > constructor
        # seed > seed drawn from gym's seeded np_random. Drawing from
        # np_random keeps reproducibility tied to the gym-seeded stream
        # without forcing the caller to pre-pick a seed.
        if seed is not None:
            episode_seed = int(seed)
        elif self._init_seed is not None:
            episode_seed = int(self._init_seed)
            # Burn one draw from np_random so subsequent unseeded resets
            # advance even when the constructor seed is reused. This is a
            # small defensive measure; gym's super().reset(seed=None) does
            # not re-seed np_random so the stream is stable.
            self.np_random.integers(0, 2**31 - 1)
        else:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))

        self._ensure_process_and_transport()
        assert self._transport is not None  # for type checkers

        self._episode_count += 1
        episode_id = f"ep-{self._episode_count:06d}"

        try:
            resp = self._transport.reset(
                seed=episode_seed,
                max_steps=self._max_steps,
                episode_id=episode_id,
            )
        except (GodotTransportError, GodotProtocolError, GodotRemoteError) as exc:
            # Transport / protocol failures must not be silently converted
            # into terminal observations. Caller decides whether to close
            # and rebuild the env.
            self._log_event(
                "error",
                where="reset",
                kind=type(exc).__name__,
                message=str(exc),
                episode_id=episode_id,
                seed=episode_seed,
            )
            raise

        obs = self._obs_to_np(resp["obs"])
        info = self._build_info(resp, episode_seed=episode_seed, episode_id=episode_id)
        self._episode_done = bool(resp.get("terminated") or resp.get("truncated"))
        self._log_event(
            "reset",
            episode_id=episode_id,
            seed=episode_seed,
            frame=int(resp.get("frame", 0)),
            terminated=bool(resp.get("terminated")),
            truncated=bool(resp.get("truncated")),
        )
        return obs, info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("step called on closed env")
        if self._transport is None:
            raise RuntimeError("step called before reset; call reset() first")
        if self._episode_done:
            raise RuntimeError(
                "step called after episode terminated/truncated; call reset()"
            )

        # Coerce numpy ints (SB3 sometimes hands these in) to plain int so
        # the transport's strict ``isinstance(action, int)`` check passes.
        if isinstance(action, np.integer):
            action = int(action)

        try:
            resp = self._transport.step(action)
        except (GodotTransportError, GodotProtocolError, GodotRemoteError) as exc:
            self._log_event(
                "error",
                where="step",
                kind=type(exc).__name__,
                message=str(exc),
            )
            raise

        obs = self._obs_to_np(resp["obs"])
        reward = float(resp["reward"])
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"])
        terminal_reason = str(resp.get("terminal_reason", protocol.TERMINAL_REASON_NONE))
        info = self._build_info(
            resp,
            episode_seed=None,
            episode_id=str(resp.get("episode_id", "")),
            terminal_reason=terminal_reason,
        )
        if terminated or truncated:
            self._episode_done = True
        self._log_event(
            "step",
            episode_id=info["episode_id"],
            frame=info["frame"],
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            terminal_reason=terminal_reason,
        )
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Idempotent. Closes the transport and tears down the subprocess."""
        if self._closed:
            return
        self._closed = True

        # Log close BEFORE tearing down the writer so the event makes it to
        # disk. _log_event no-ops when run_dir was not supplied.
        self._log_event("close")

        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                # close() must not raise on cleanup; transport.close() is
                # already idempotent but a misbehaving fake should not
                # block subprocess teardown.
                pass
            self._transport = None

        proc = self._proc
        self._proc = None
        if proc is not None:
            self._terminate_process(proc)

        # Release stdout/stderr file handles after the child has been
        # terminated so any final flush/close from Godot has landed before
        # we drop the parent's view. ``close()`` is idempotent; both fields
        # may already be ``None`` if ``run_dir`` was not supplied or if a
        # prior ``close()`` cleared them.
        if self._godot_stdout_file is not None:
            try:
                self._godot_stdout_file.close()
            except OSError:
                pass
            self._godot_stdout_file = None
        if self._godot_stderr_file is not None:
            try:
                self._godot_stderr_file.close()
            except OSError:
                pass
            self._godot_stderr_file = None

        if self._ndjson is not None:
            self._ndjson.close()
            self._ndjson = None

    def __del__(self) -> None:  # pragma: no cover - finaliser best-effort
        try:
            self.close()
        except Exception:
            pass

    # --- internals -------------------------------------------------------

    def _log_event(self, event_type: str, **fields: Any) -> None:
        """Write one Python-side NDJSON event. No-op without ``run_dir``.

        Decorates every record with the durable identity fields
        (``run_id``, ``godot_pid``, ``tcp_port``) so each line is
        self-contained and grep-friendly. Caller-supplied ``fields``
        override the auto-decoration only if a key collides; explicit
        ``None`` values are dropped so a missing ``frame`` does not pollute
        the schema.
        """
        if self._ndjson is None:
            return
        merged: dict[str, Any] = {
            "run_id": self._run_id,
            "godot_pid": self.godot_pid,
            "tcp_port": self._tcp_port,
        }
        merged.update(fields)
        self._ndjson.write(event_type, merged)

    def _ensure_process_and_transport(self) -> None:
        """Lazy-launch Godot and connect on first reset; reuse afterwards.

        Opens the Python NDJSON writer first (when ``run_dir`` is set) so
        any failure during launch / connect is recorded as an ``error``
        event before being re-raised.
        """
        first_launch = self._proc is None and self._transport is None
        if first_launch and self._ndjson is None and self._run_dir is not None:
            self._ndjson = _NdjsonWriter(self._run_dir / _PYTHON_NDJSON_NAME)

        if self._proc is None:
            try:
                self._proc = self._launch_godot()
            except Exception as exc:
                self._log_event(
                    "error",
                    where="launch",
                    kind=type(exc).__name__,
                    message=str(exc),
                )
                raise

        if self._transport is None:
            try:
                self._transport = self._connect_transport()
                self._transport.send_hello()
            except Exception as exc:
                self._log_event(
                    "error",
                    where="connect",
                    kind=type(exc).__name__,
                    message=str(exc),
                )
                raise
            if first_launch:
                self._log_event("env_start")

    def _launch_godot(self) -> subprocess.Popen:
        cmd: list[str] = [str(self._godot_executable), "--path", str(self._project_path)]
        if self._headless:
            cmd.append("--headless")

        env = os.environ.copy()
        env["SIGHT_TCP_MODE"] = "1"
        env["SIGHT_TCP_PORT"] = str(self._tcp_port)
        if self._run_dir is not None:
            log_path = Path(self._run_dir) / "godot.ndjson"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env["SIGHT_GODOT_LOG_PATH"] = str(log_path)

        # Persist the launch cmd so ``_connect_transport`` can include it in
        # early-exit diagnostics. Captured before the factory call so a
        # raising factory still leaves a useful breadcrumb.
        self._launch_cmd = list(cmd)

        # Stdout/stderr redirection. CRITICAL: ``subprocess.PIPE`` deadlocks
        # Godot 4.6.2 on Windows at startup before ``_ready()`` runs, with
        # no data ever written to the pipe. Verified via matrix test on
        # 2026-05-09 against both the windowed and console builds: PIPE
        # hangs the engine at the version banner; ``CREATE_NO_WINDOW``
        # does not help; ``DEVNULL`` and file redirection both work in
        # under one second. Root cause is Godot's Windows stdio handle
        # path interacting badly with anonymous pipes from a non-console
        # parent. File-based capture preserves stdout/stderr evidence on
        # crashes when ``run_dir`` is set so the operator can post-mortem
        # without reproducing the crash; ``DEVNULL`` is the fallback when
        # no ``run_dir`` is supplied.
        stdout_target: Any
        stderr_target: Any
        if self._run_dir is not None:
            run_dir_path = Path(self._run_dir)
            run_dir_path.mkdir(parents=True, exist_ok=True)
            self._godot_stdout_file = open(
                run_dir_path / "godot-stdout.log", "wb"
            )
            self._godot_stderr_file = open(
                run_dir_path / "godot-stderr.log", "wb"
            )
            stdout_target = self._godot_stdout_file
            stderr_target = self._godot_stderr_file
        else:
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL

        return self._process_factory(
            cmd,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
        )

    def _connect_transport(self) -> GodotH3Transport:
        """Build the transport and retry connect within ``connect_timeout_s``.

        Godot needs a moment to bind the listener after ``Popen`` returns.
        Retry with a small interval; on the final attempt, surface the
        underlying ``GodotTransportError``.

        Early-exit detection: between attempts, check ``self._proc.poll()``.
        If Godot has exited before the listener became reachable, raise
        immediately with the exit code, the launch cmd, and the port. This
        is distinct from a connect timeout: a process that died at startup
        will never start listening, so the operator should see the process
        cause not the connect cause.
        """
        transport = self._transport_factory(
            run_id=self._run_id,
            host=self._tcp_host,
            port=self._tcp_port,
            recv_timeout_s=self._step_timeout_s,
        )
        deadline = time.monotonic() + self._connect_timeout_s
        attempts = 0
        while True:
            attempts += 1
            self._raise_if_godot_exited()
            try:
                transport.connect(connect_timeout_s=_CONNECT_PER_ATTEMPT_TIMEOUT_S)
                return transport
            except GodotTransportError as e:
                # Re-check after a failed attempt: Godot may have died
                # exactly during the connect attempt rather than before it.
                self._raise_if_godot_exited(connect_error=e)
                if time.monotonic() >= deadline:
                    raise GodotTransportError(
                        f"Godot TCP listener not reachable within "
                        f"{self._connect_timeout_s}s after {attempts} attempts: {e}"
                    ) from e
                time.sleep(_CONNECT_RETRY_INTERVAL_S)

    def _raise_if_godot_exited(
        self, connect_error: Exception | None = None
    ) -> None:
        """Raise ``GodotTransportError`` if the subprocess has exited.

        Distinguishes early-exit failures from connect-timeout failures.
        Includes exit code, launch cmd, and the port being waited on so
        operators can diagnose without rerunning under a debugger.
        """
        proc = self._proc
        if proc is None:
            return
        rc = proc.poll()
        if rc is None:
            return
        cmd_repr = " ".join(self._launch_cmd) if self._launch_cmd else "<unknown>"
        msg = (
            f"Godot subprocess exited with code {rc} before TCP listener on "
            f"port {self._tcp_port} became reachable; cmd=[{cmd_repr}]"
        )
        if connect_error is not None:
            msg += f"; last connect error: {connect_error}"
        raise GodotTransportError(msg)

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        # Already exited?
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=_TERMINATE_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        # Hard kill if it did not exit during the grace window.
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=_TERMINATE_GRACE_S)
        except Exception:
            pass

    def _obs_to_np(self, obs: list) -> np.ndarray:
        """Convert wire-side JSON list to the numpy observation.

        The transport already validated length 10 and numeric per element;
        this method just dtype-casts. Out-of-range values are NOT clipped
        here; the Godot side is contractually required to clamp to [-1, 1]
        per docs/sight-h3-plan.md section 2 and the H3 step 4 obs builder.
        Treating violation as a bug rather than silently masking it keeps
        the contract enforceable.
        """
        arr = np.asarray(obs, dtype=np.float32)
        # Defensive shape check; transport guarantees this but a misbehaving
        # fake transport in tests could violate it.
        if arr.shape != (10,):
            raise GodotProtocolError(
                f"obs array has shape {arr.shape}, expected (10,)"
            )
        return arr

    def _build_info(
        self,
        resp: dict,
        *,
        episode_seed: int | None,
        episode_id: str,
        terminal_reason: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the ``info`` dict per docs/sight-h3-plan.md section 6.

        Keys ``run_id``, ``episode_id``, ``godot_pid``, ``tcp_port``,
        ``frame`` are always present. ``seed`` is present on reset (when
        ``episode_seed`` is supplied) and absent on step. ``terminal_reason``
        is present on step.
        """
        info: dict[str, Any] = {
            "run_id": self._run_id,
            "episode_id": episode_id,
            "godot_pid": self.godot_pid,
            "tcp_port": self._tcp_port,
            "frame": int(resp.get("frame", 0)),
        }
        if episode_seed is not None:
            info["seed"] = int(episode_seed)
        if terminal_reason is not None:
            info["terminal_reason"] = terminal_reason
        # Forward Godot-supplied per-step info as a nested dict so callers
        # can inspect collision details without colliding with Python info
        # keys above.
        wire_info = resp.get("info")
        if isinstance(wire_info, dict):
            info["godot_info"] = wire_info
        return info


# --- default factories ----------------------------------------------------


def _default_transport_factory(
    run_id: str,
    host: str,
    port: int,
    recv_timeout_s: float,
) -> GodotH3Transport:
    return GodotH3Transport(
        run_id=run_id, host=host, port=port, recv_timeout_s=recv_timeout_s
    )


def _default_process_factory(
    cmd: list[str],
    *,
    env: dict[str, str],
    stdout: Any = None,
    stderr: Any = None,
) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
