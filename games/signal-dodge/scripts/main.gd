extends Node2D

# Sight - Main loop. Owns the deterministic per-tick pipeline:
#   world step (hazards) -> agent.capture -> agent.perceive -> agent.decide
#   -> player.move_action -> log -> spawn -> UI
# All logic runs in _physics_process at 60 Hz.
#
# TCP mode (opt-in): set env var SIGHT_TCP_MODE=1 (optionally SIGHT_TCP_PORT=<int>).
# In TCP mode the in-Godot agent pipeline is bypassed and player actions come from a
# loopback TCP client (see scripts/tcp_controller.gd and docs/sight-handoff.md).

const HAZARD_SCENE := preload("res://scenes/hazard.tscn")
const TCP_CONTROLLER := preload("res://scripts/tcp_controller.gd")
const SPAWN_INTERVAL_FRAMES := 30
const SCREEN_WIDTH := 720
const HAZARD_SIZE := 24
const RANDOM_SEED := 42

var _frame_counter := 0
var _run_start_ms: int = 0
var _alive := true
var _hazards: Array = []
var _hazard_id_counter := 0
var _tcp = null  # tcp_controller.gd instance; kept duck-typed so method calls resolve dynamically
var _tcp_mode := false
var _tcp_ignore_death := false  # SIGHT_TCP_IGNORE_DEATH=1 + TCP mode: keep the run alive past collision for transport endurance tests. Default false; gameplay unchanged in non-TCP mode.
# SIGHT_P3_ACTIONS_BUDGET is read once at _ready in TCP mode. Empty/missing/non-positive
# values disable the success-budget terminal entirely. Gameplay and non-TCP behavior unchanged.
var _actions_budget: int = 0

@onready var _player: Area2D = $Player
@onready var _agent: Node = $Agent
@onready var _survival_label: Label = $SurvivalLabel

func _ready() -> void:
	seed(RANDOM_SEED)
	_run_start_ms = Time.get_ticks_msec()
	_tcp_mode = OS.get_environment("SIGHT_TCP_MODE") == "1"
	_tcp_ignore_death = _tcp_mode and (OS.get_environment("SIGHT_TCP_IGNORE_DEATH") == "1")
	# SIGHT_P3_ACTIONS_BUDGET only consulted in TCP mode. int("") and int("abc") both
	# return 0 in GDScript; zero or negative values keep _actions_budget at 0, which the
	# physics_process check treats as "no success-budget terminal".
	if _tcp_mode:
		var parsed_budget := int(OS.get_environment("SIGHT_P3_ACTIONS_BUDGET"))
		if parsed_budget > 0:
			_actions_budget = parsed_budget
	var mode := "tcp" if _tcp_mode else "in_godot"
	var run_meta := {
		"seed": RANDOM_SEED,
		"mode": mode,
		"screen_width": SCREEN_WIDTH,
		"screen_height": 540,
		"spawn_interval_frames": SPAWN_INTERVAL_FRAMES,
		"hazard_size": HAZARD_SIZE,
		"physics_hz": 60,
	}
	if _tcp_mode:
		run_meta["tcp_ignore_death"] = _tcp_ignore_death
		if _actions_budget > 0:
			run_meta["actions_budget"] = _actions_budget
	SightLog.start_run(run_meta)
	_player.died.connect(_on_player_died)
	if _tcp_mode:
		_tcp = TCP_CONTROLLER.new()
		add_child(_tcp)
		var port_env := OS.get_environment("SIGHT_TCP_PORT")
		var port := 8765 if port_env == "" else int(port_env)
		var err: int = _tcp.start("127.0.0.1", port)
		if err != OK:
			push_error("tcp controller failed to start; falling back to in-Godot agent")
			_tcp_mode = false

func _physics_process(delta: float) -> void:
	if not _alive:
		return
	_frame_counter += 1

	# 1. World step: advance hazards, cull offscreen.
	var survivors: Array = []
	for h in _hazards:
		if not is_instance_valid(h):
			continue
		if h.step(delta):
			h.queue_free()
		else:
			survivors.append(h)
	_hazards = survivors

	# 2. Resolve action: TCP mode polls the loopback socket, otherwise run the in-Godot pipeline.
	var action := 0
	var state := {}
	var perceived := {"threat": false}
	if _tcp_mode:
		action = _tcp.poll(_frame_counter)
	else:
		state = _agent.capture(_player, _hazards)
		perceived = _agent.perceive(state)
		action = _agent.decide(perceived, state["player_x"], float(SCREEN_WIDTH))

	# 3. Controller: apply action.
	_player.move_action(action, delta)
	if _tcp_mode:
		_tcp.log_applied(_frame_counter)
		# Success-budget terminal. Distinct from _on_player_died: runs only while alive and
		# only when the configured number of distinct seqs have been first-applied. Logs the
		# terminal event, ends the run cleanly, stops TCP, and quits. SIGHT_TCP_IGNORE_DEATH
		# is not consulted here; this branch is gated by _alive (which the ignore-death path
		# keeps true after collision) and the budget alone.
		if _actions_budget > 0 and _alive:
			var ac := _tcp.applied_count()
			if ac >= _actions_budget:
				_alive = false
				var evt := {
					"frame": _frame_counter,
					"applied_count": ac,
					"actions_budget": _actions_budget,
				}
				var rid := _tcp.run_id()
				if rid != "":
					evt["run_id"] = rid
				var ep := OS.get_environment("SIGHT_EPISODE_ID")
				if ep != "":
					evt["episode_id"] = ep
				SightLog.log_event("success_budget_reached", evt)
				SightLog.end_run()
				_tcp.stop()
				_survival_label.text = "BUDGET MET  applied=%d" % ac
				get_tree().quit()
				return

	# 4. Log player/agent tick.
	var rec := {
		"frame": _frame_counter,
		"player_x": _player.position.x,
		"player_y": _player.position.y,
		"action": action,
	}
	if _tcp_mode:
		SightLog.log_event("player_tick", rec)
	else:
		rec["threat"] = perceived.get("threat", false)
		if perceived.get("threat", false):
			rec["threat_x"] = perceived["x"]
			rec["threat_y"] = perceived["y"]
			rec["threat_dist"] = perceived["dist"]
		SightLog.log_event("agent_tick", rec)

	# 5. Spawn.
	if _frame_counter % SPAWN_INTERVAL_FRAMES == 0:
		_spawn_hazard()

	# 6. UI.
	var t := (Time.get_ticks_msec() - _run_start_ms) / 1000.0
	_survival_label.text = "Survival: %.2fs  action=%d" % [t, action]

func _spawn_hazard() -> void:
	_hazard_id_counter += 1
	var hz := HAZARD_SCENE.instantiate()
	var x := randf_range(HAZARD_SIZE / 2.0, SCREEN_WIDTH - HAZARD_SIZE / 2.0)
	var y := float(-HAZARD_SIZE)
	hz.position = Vector2(x, y)
	add_child(hz)
	_hazards.append(hz)
	SightLog.log_event("spawn", {
		"hazard_id": _hazard_id_counter,
		"frame": _frame_counter,
		"x": x,
		"y": y,
	})

func _on_player_died(survival_time: float, hazard_pos: Vector2, player_pos: Vector2) -> void:
	# TCP test mode: when SIGHT_TCP_MODE=1 and SIGHT_TCP_IGNORE_DEATH=1, log a non-terminal
	# event and keep the run alive so the harness can verify TCP transport endurance over
	# the full ACTIONS budget. Default and non-TCP gameplay behavior is unchanged. The flag
	# is opt-in and only consulted when _tcp_mode is true.
	if _tcp_mode and _tcp_ignore_death:
		SightLog.log_event("tcp_death_ignored", {
			"survival_time": survival_time,
			"player_x": player_pos.x,
			"player_y": player_pos.y,
			"hazard_x": hazard_pos.x,
			"hazard_y": hazard_pos.y,
			"frame": _frame_counter,
		})
		return
	_alive = false
	SightLog.log_event("collision", {
		"player_x": player_pos.x,
		"player_y": player_pos.y,
		"hazard_x": hazard_pos.x,
		"hazard_y": hazard_pos.y,
	})
	SightLog.log_event("death", {"survival_time": survival_time})
	SightLog.end_run()
	if _tcp != null:
		_tcp.stop()
	_survival_label.text = "DEAD  t=%.2fs" % survival_time
	get_tree().quit()
