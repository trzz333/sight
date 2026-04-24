extends Area2D

# Hazard - Area2D. Driven by Main.step(delta). Returns true when offscreen (to be culled).

const SPEED := 200.0
const SIZE := 24
const SCREEN_HEIGHT := 540

func _ready() -> void:
	add_to_group("hazard")
	queue_redraw()

# Driven by Main each physics tick. Returns true if offscreen and should be freed.
func step(delta: float) -> bool:
	position.y += SPEED * delta
	return position.y > SCREEN_HEIGHT + SIZE

func _draw() -> void:
	draw_rect(Rect2(-SIZE / 2.0, -SIZE / 2.0, SIZE, SIZE), Color.RED)
