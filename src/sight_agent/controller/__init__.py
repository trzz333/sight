"""TCP loopback controller.

P2 uses JSON-line IPC (one JSON object per line, newline terminated) bound to 127.0.0.1 only.
Keyboard injection is explicitly out of scope for Sight: see docs/sight-charter.md ethics armor
(no bot-detection evasion, no anti-cheat surface). Loopback IPC gives deterministic command
sequencing and better test hygiene.

Wire messages:
    hello   {"type":"hello","protocol":1,"run_id":"<id>","agent":"<name>"}
    action  {"type":"action","seq":<int>,"ts_unix_ns":<int>,"action":"left|right|stay","move_x":-1|0|1}

Controller is the client. Godot hosts the listen socket in TCP mode (scripts/tcp_controller.gd).
"""

from .tcp_client import TcpController

__all__ = ["TcpController"]
