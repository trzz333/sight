# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 in progress (phase (a) verified; phase (b) blocked on missing Python entrypoint, sequencing inverted from prior plan)

**Last commit:** 1233677 handoff: phase (b) architecture inversion, runner spec pending

**Current task:** Stage 1 recovery resolved the Python interpreter and pytest gate. Stage 2 architecture audit found that GPT's "Python harness binds 127.0.0.1:8765 first" is inverted. Godot is the TCP server (`games\signal-dodge\scripts\tcp_controller.gd`), Python is the client (`src\sight_agent\controller\tcp_client.py`, docstring `Binds nowhere`). No Python file in the repo has a `__main__` block, so there is no runnable Python entrypoint that can drive a live phase (b) run today. Phase (b) is not failed, it is unbuilt on the Python side.

**Next action:** GPT to specify the phase (b) Python runner shape before any Godot launch. Two design questions: (1) what is the intended phase (b) lived flow, connection probe only (`controller_hello` + N `agent_tick` from a stub policy) or end-to-end (perception -> policy -> tcp action -> logger), and (2) where the runner lives, `src\sight_agent\__main__.py` vs `scripts\run_phase_b.py`. Once specified, Claude builds the runner, then launches Godot first to bind 8765, confirms LISTEN, then runs the Python client and captures NDJSON.

**Blockers:** Phase (b) runner shape and location need GPT spec. Listed in Next action.

**Notes:**

- Architecture inversion confirmed by code. `tcp_client.py` docstring states `Binds nowhere. Connects to 127.0.0.1:8765`. `tcp_controller.gd` (5779 bytes) is the Godot-side listener. Correct sequencing is Godot first as server, Python second as client. Tests already prove the wire schema via `_FakeTcpServer` in `tests\test_controller.py`.
- No `__main__` blocks anywhere in the repo (whole-tree grep, probe at `C:\Users\maste\AppData\Local\Temp\sight_harness_probe.txt`). No `console_scripts` entry in `pyproject.toml`. The only Python execution path today is pytest.
- Pytest gate resolved without `.venv`. `py -3 -m pytest` from `C:\Projects\Sight` passes 36 of 37 (1 deselected, `live_mss`). Python 3.14.4 at `C:\Users\maste\AppData\Local\Python\pythoncore-3.14-64\python.exe`. The package is installed editable into that user-site. The prior handoff note about needing `.venv` is stale and superseded.
- Bare `python` and `python3` are not on PATH in spawned shells; use `py -3` everywhere. WindowsApps stub for `python.exe` is still present and should be avoided.
- Reusable artifacts in `C:\Users\maste\AppData\Local\Temp`: `sight_harness_probe.ps1`, `sight_harness_probe.txt`, `sight_phase_b.ps1`. Keep until phase (b) runner lands.