# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** P2 (Phase B TCP live verify complete)

**Last commit:** a6e512d handoff: Phase B TCP live verify

**Current task:** Phase B TCP transport verified end-to-end on 127.0.0.1:8765. Godot is the server, scripts\run_phase_b.py is the client. Reconciliation evaluator confirms run_id parity and zero duplicate applied_seq across 89 of 90 sent actions in the live round.

**Next action:** Run a 300-action confirmation pass with the same runner against the same loopback, capture the diagnostics, then codify a tail-tolerance threshold in the evaluator before P3 measurement.

**Blockers:** Direction call pending. P2 closeout vs P3 measurement gate prep. Jeff to choose.

**Notes:**

- Godot NDJSON. C:\Users\maste\AppData\Roaming\Godot\app_userdata\Signal Dodge\runs\run_2026-04-25T17-12-17.ndjson
- Python NDJSON. C:\Projects\Sight\runs\phase_b_python_20260425T171218.ndjson
- Evaluator. joined=89/90, run_id_mismatch=false, duplicate_applied_seq_count=0, latency p50/p95 = 0 ms loopback.
- Tail unmatched=1 is structural at Godot quit. Codify a tolerance threshold before P3.
- Repo-root orphans quarantined. C:\Users\maste\AppData\Local\Temp\sight_orphans_20260425T170638\