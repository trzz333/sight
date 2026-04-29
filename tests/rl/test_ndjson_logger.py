"""Tests for sight_agent.rl.ndjson_logger."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sight_agent.rl.ndjson_logger import (
    SCHEMA_VERSION,
    NDJSONLogger,
    to_jsonable,
)


def _make_logger(tmp_path: Path) -> NDJSONLogger:
    return NDJSONLogger(
        path=tmp_path / "events.ndjson",
        run_id="test_run_id",
        phase="H1",
        env_id="CartPole-v1",
        algo="PPO",
        framework="stable-baselines3",
        seed=0,
        git_commit="abc1234",
    )


def test_writes_one_json_object_per_line(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_event("run_start", step=0, config_path="x.yaml")
    logger.log_event("eval", step=10, metrics={"mean_reward": 1.5})
    logger.log_event("run_end", step=20, status="ok")
    logger.close()

    raw = (tmp_path / "events.ndjson").read_bytes()
    assert raw.endswith(b"\n"), "file must be newline-terminated"
    text = raw.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert len(lines) == 3
    parsed = [json.loads(ln) for ln in lines]
    assert [r["event"] for r in parsed] == ["run_start", "eval", "run_end"]


def test_common_fields_auto_filled(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_event("run_start", step=0)
    logger.close()
    rec = json.loads((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()[0])
    for key in (
        "schema_version", "run_id", "ts_utc", "phase", "env_id",
        "algo", "framework", "seed", "git_commit", "event", "step",
    ):
        assert key in rec, f"missing common field {key}"
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["phase"] == "H1"
    assert rec["env_id"] == "CartPole-v1"
    assert rec["algo"] == "PPO"
    assert rec["framework"] == "stable-baselines3"
    assert rec["seed"] == 0
    assert rec["git_commit"] == "abc1234"
    assert rec["event"] == "run_start"
    assert rec["step"] == 0


def test_step_can_be_none(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_event("info", step=None, note="no step")
    logger.close()
    rec = json.loads((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()[0])
    assert rec["step"] is None


def test_numpy_scalars_are_coerced(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.log_event(
        "train_metrics",
        step=42,
        metrics={
            "loss": np.float32(0.125),
            "n_updates": np.int64(7),
            "rewards": np.array([1.0, 2.0, 3.0]),
        },
    )
    logger.close()
    rec = json.loads((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()[0])
    assert rec["metrics"]["loss"] == pytest.approx(0.125)
    assert rec["metrics"]["n_updates"] == 7
    assert rec["metrics"]["rewards"] == [1.0, 2.0, 3.0]


def test_to_jsonable_handles_nested_and_nans() -> None:
    nested = {
        "a": np.float64(1.0),
        "b": [np.int32(2), {"c": np.array([3.0, 4.0])}],
        "d": float("nan"),
        "e": float("inf"),
    }
    out = to_jsonable(nested)
    json.dumps(out)  # must round-trip
    assert out["a"] == 1.0
    assert out["b"][0] == 2
    assert out["b"][1]["c"] == [3.0, 4.0]
    # NaN/Inf are stringified to keep JSON strict-compliant.
    assert isinstance(out["d"], str)
    assert isinstance(out["e"], str)


def test_unserializable_falls_back_to_str(tmp_path: Path) -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<Weird>"

    logger = _make_logger(tmp_path)
    logger.log_event("info", step=None, payload=Weird())
    logger.close()
    rec = json.loads((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()[0])
    assert rec["payload"] == "<Weird>"


def test_context_manager_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with NDJSONLogger(
        path=path,
        run_id="ctx",
        phase="H1",
        env_id="CartPole-v1",
        algo="PPO",
        framework="stable-baselines3",
        seed=1,
        git_commit=None,
    ) as logger:
        logger.log_event("run_start", step=0)
    assert path.exists()
    assert path.read_text(encoding="utf-8").count("\n") == 1
