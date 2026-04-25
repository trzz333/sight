"""Sight external Python agent layer.

Package layout matches GPT's P2 scaffold decision:
    capture/       screen region capture (fake + MSS)
    perception/    HSV color thresholding vision
    policy/        rule-parity port of games/signal-dodge/scripts/agent.gd
    controller/    TCP loopback JSON-line client
    logger/        per-run NDJSON writer
    evaluator/     reconciler (python.decision.seq <-> godot.controller_cmd_applied.seq)

Wire contract and constants live in `sight_agent.constants`.
"""

from . import constants  # noqa: F401
