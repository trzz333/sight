"""Tests for run_phase_b. No live sockets."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase_b as rpb  # noqa: E402


def test_action_for_seq_stay_band():
    for seq in range(0, 10):
        assert rpb.action_for_seq(seq) == ("stay", 0)


def test_action_for_seq_left_band():
    for seq in range(10, 20):
        assert rpb.action_for_seq(seq) == ("left", -1)


def test_action_for_seq_right_band():
    for seq in range(20, 30):
        assert rpb.action_for_seq(seq) == ("right", 1)


def test_action_for_seq_wraps_modulo_30():
    assert rpb.action_for_seq(30) == ("stay", 0)
    assert rpb.action_for_seq(45) == ("left", -1)
    assert rpb.action_for_seq(59) == ("right", 1)


def test_build_hello_schema():
    h = rpb.build_hello("phase-b-test", "phase_b_stub")
    assert h["type"] == "hello"
    assert h["protocol"] == 1
    assert h["run_id"] == "phase-b-test"
    assert h["agent"] == "phase_b_stub"
    json.dumps(h)


def test_build_action_schema_stay():
    a = rpb.build_action(0, 1234567890)
    assert a["type"] == "action"
    assert a["seq"] == 0
    assert a["ts_unix_ns"] == 1234567890
    assert a["action"] == "stay"
    assert a["move_x"] == 0
    json.dumps(a)


def test_build_action_schema_left():
    a = rpb.build_action(15, 999)
    assert a["action"] == "left"
    assert a["move_x"] == -1


def test_build_action_schema_right():
    a = rpb.build_action(25, 999)
    assert a["action"] == "right"
    assert a["move_x"] == 1


def test_build_decision_includes_required_fields():
    d = rpb.build_decision("phase-b-test", 5, 100, 200)
    assert d["type"] == "decision"
    assert d["run_id"] == "phase-b-test"
    assert d["seq"] == 5
    assert d["action"] == "stay"
    assert d["move_x"] == 0
    assert d["capture_ts_unix_ns"] == 100
    assert d["decision_ts_unix_ns"] == 100
    assert d["sent_ts_unix_ns"] == 200
    json.dumps(d)