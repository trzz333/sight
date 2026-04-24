extends Node2D

# Main - scene orchestrator. Spawns hazards on a frame counter, handles end-of-run.

const HAZARD_SCENE := preload("res://scenes/hazard.tscn")
const SPAWN_INTERVAL_FRAMES := 30
const SCREEN_WIDTH := 720
const HAZARD_SIZE := 24

var _frame_counter := 0
var _run_start_ms: int = 0
var _alive := true

@onready var _player: Area2D = $Player
@onready var _survival_label: Label = $SurvivalLabel

func _ready() -> void:
	_run_start_ms = Time.get_ticks_msec()
	Logger.start_run()
	_player.died.connect(_on_player_died)

func _physics_process(_delta: float) -> void:
	if not _alive:
		return
	_frame_counter += 1
	if _frame_counter % SPAWN_INTERVAL_FRAMES == 0:
		_spawn_hazard()
	var t := (Time.get_ticks_msec() - _run_start_ms) / 1000.0
	_survival_label.text = "Survival: %.2fs" % t

func _spawn_hazard() -> void:
	var hz := HAZARD_SCENE.instantiate()
	var x := randf_range(HAZARD_SIZE / 2.0, SCREEN_WIDTH - HAZARD_SIZE / 2.0)
	var y := float(-HAZARD_SIZE)
	hz.position = Vector2(x, y)
	add_child(hz)
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
