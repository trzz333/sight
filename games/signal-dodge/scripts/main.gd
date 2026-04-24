extends Node2D

# Sight - Main loop. Owns the deterministic per-tick pipeline:
#   world step (hazards) -> agent.capture -> agent.perceive -> agent.decide
#   -> player.move_action -> log -> spawn -> UI
# All logic runs in _physics_process at 60 Hz.

const HAZARD_SCENE := preload("res://scenes/hazard.tscn")
const SPAWN_INTERVAL_FRAMES := 30
const SCREEN_WIDTH := 720
const HAZARD_SIZE := 24
const RANDOM_SEED := 42

var _frame_counter := 0
var _run_start_ms: int = 0
var _alive := true
var _hazards: Array = []

@onready var _player: Area2D = $Player
@onready var _agent: Node = $Agent
@onready var _survival_label: Label = $SurvivalLabel

func _ready() -> void:
	seed(RANDOM_SEED)
	_run_start_ms = Time.get_ticks_msec()
	Logger.start_run({"seed": RANDOM_SEED})
	_player.died.connect(_on_player_died)

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

	# 2. Agent pipeline: capture -> perceive -> decide.
	var state := _agent.capture(_player, _hazards)
	var perceived := _agent.perceive(state)
	var action: int = _agent.decide(perceived, state["player_x"], float(SCREEN_WIDTH))

	# 3. Controller: apply action.
	_player.move_action(action, delta)

	# 4. Log agent tick (compact).
	var rec := {
		"frame": _frame_counter,
		"player_x": state["player_x"],
		"action": action,
		"threat": perceived.get("threat", false),
	}
	if perceived.get("threat", false):
		rec["threat_x"] = perceived["x"]
		rec["threat_y"] = perceived["y"]
		rec["threat_dist"] = perceived["dist"]
	Logger.log_event("agent_tick", rec)

	# 5. Spawn.
	if _frame_counter % SPAWN_INTERVAL_FRAMES == 0:
		_spawn_hazard()

	# 6. UI.
	var t := (Time.get_ticks_msec() - _run_start_ms) / 1000.0
	_survival_label.text = "Survival: %.2fs  action=%d" % [t, action]

func _spawn_hazard() -> void:
	var hz := HAZARD_SCENE.instantiate()
	var x := randf_range(HAZARD_SIZE / 2.0, SCREEN_WIDTH - HAZARD_SIZE / 2.0)
	var y := float(-HAZARD_SIZE)
	hz.position = Vector2(x, y)
	add_child(hz)
	_hazards.append(hz)
	Logger.log_event("spawn", {"x": x, "y": y})

func _on_player_died(survival_time: float, hazard_pos: Vector2, player_pos: Vector2) -> void:
	_alive = false
	Logger.log_event("collision", {
		"player_x": player_pos.x,
		"player_y": player_pos.y,
		"hazard_x": hazard_pos.x,
		"hazard_y": hazard_pos.y,
	})
	Logger.log_event("death", {"survival_time": survival_time})
	Logger.end_run()
	_survival_label.text = "DEAD  t=%.2fs" % survival_time
	get_tree().quit()
