"""Rule policy - pure port of games/signal-dodge/scripts/agent.gd.

Semantics must match the GDScript exactly. Any discovered ambiguity is noted in
docs/sight-handoff.md, never "fixed" here. This module is the P2 baseline-parity agent; a
smarter policy belongs in a separate module and a separate phase.
"""

from .rule import decide, perceive

__all__ = ["perceive", "decide"]
