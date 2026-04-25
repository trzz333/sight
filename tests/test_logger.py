"""Logger tests. Verify JSON-per-line, required fields, and append mode."""

from __future__ import annotations

import json

from sight_agent.logger import NDJSONLogger, new_run_id


def test_new_run_id_has_prefix():
    rid = new_run_id("run")
    assert rid.startswith("run_")
    assert len(rid) > 10


def test_logger_writes_valid_json_per_line(tmp_path):
    run_id = "run_test_logger"
    run_dir = tmp_path / run_id
    logger = NDJSONLogger(run_dir, side="python")
    logger.log("decision", seq=1, action="left", move_x=-1)
    logger.log("decision", seq=2, action="stay", move_x=0)
    logger.log("disconnect", reason="test_close")
    logger.close()

    path = run_dir / "python.ndjson"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    for rec in parsed:
        assert rec["run_id"] == run_id
        assert "ts_unix_ns" in rec
        assert isinstance(rec["ts_unix_ns"], int)
        assert "type" in rec

    assert parsed[0]["type"] == "decision"
    assert parsed[0]["seq"] == 1
    assert parsed[2]["type"] == "disconnect"


def test_logger_writes_manifest(tmp_path):
    run_dir = tmp_path / "run_manifest_check"
    logger = NDJSONLogger(run_dir, side="python", manifest_extra={"host": "StrongerJr"})
    logger.close()
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_manifest_check"
    assert manifest["sides"] == ["python"]
    assert manifest["host"] == "StrongerJr"


def test_logger_append_mode_keeps_existing_lines(tmp_path):
    run_dir = tmp_path / "run_append"

    a = NDJSONLogger(run_dir, side="python")
    a.log("decision", seq=1)
    a.close()

    b = NDJSONLogger(run_dir, side="python")
    b.log("decision", seq=2)
    b.close()

    lines = (run_dir / "python.ndjson").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [p["seq"] for p in parsed] == [1, 2]


def test_logger_context_manager(tmp_path):
    run_dir = tmp_path / "run_ctx"
    with NDJSONLogger(run_dir) as logger:
        logger.log("decision", seq=1)
    # Closing should not raise; file must be readable afterward.
    lines = (run_dir / "python.ndjson").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
