extends Area2D

# Player - Area2D. Agent-driven movement via move_action(). Emits died() on hazard contact.

signal died(survival_time: float, hazard_pos: Vector2, player_pos: Vector2)

const SPEED := 300.0
const SIZE := 32
const SCREEN_WIDTH := 720
const SCREEN_HEIGHT := 540

var _run_start_ms: int = 0

func _ready() -> void:
	_run_start_ms = Time.get_ticks_msec()
	position = Vector2(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT - SIZE)
	area_entered.connect(_on_area_entered)
	queue_redraw()

# Agent-driven. action: -1 left, 0 stay, +1 right. Called by Main each physics tick.
func move_action(action: int, delta: float) -> void:
	var dir: float = clampf(float(action), -1.0, 1.0)
	position.x += dir * SPEED * delta
	position.x = clamp(position.x, SIZE / 2.0, SCREEN_WIDTH - SIZE / 2.0)

func _draw() -> void:
	draw_rect(Rect2(-SIZE / 2.0, -SIZE / 2.0, SIZE, SIZE), Color.WHITE)

func _on_area_entered(other: Area2D) -> void:
	if other.is_in_group("hazard"):
		var survival := (Time.get_ticks_msec() - _run_start_ms) / 1000.0
		died.emit(survival, other.position, position)
