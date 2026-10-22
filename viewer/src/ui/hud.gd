extends Control
## Heads-up display: time and status (top left), help (H), timeline (bottom).

signal seek_requested(t: float)
signal play_toggled

var status: Label
var help: Label
var slider: HSlider
var play_button: Button
var rate_label: Label
var _dragging_slider := false

const HELP := """Mouse: drag to orbit, wheel to zoom
Space  play / pause        , .  slower / faster
1  spacecraft, LVLH        2  spacecraft, inertial
3  spacecraft, body        4  Earth-centred
L  follow live             Home  reset view
A  body axes               H  this help"""


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	status = _label(Vector2(16, 12))
	help = _label(Vector2(16, 190))
	help.text = HELP
	help.visible = false

	var bar := HBoxContainer.new()
	bar.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	bar.offset_top = -44
	bar.offset_left = 16
	bar.offset_right = -16
	bar.offset_bottom = -12
	add_child(bar)
	play_button = Button.new()
	play_button.custom_minimum_size = Vector2(64, 0)
	play_button.pressed.connect(func() -> void: play_toggled.emit())
	bar.add_child(play_button)
	slider = HSlider.new()
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.step = 0.0
	slider.drag_started.connect(func() -> void: _dragging_slider = true)
	slider.drag_ended.connect(func(_c: bool) -> void: _dragging_slider = false)
	slider.value_changed.connect(func(v: float) -> void: seek_requested.emit(v))
	bar.add_child(slider)
	rate_label = Label.new()
	rate_label.custom_minimum_size = Vector2(90, 0)
	bar.add_child(rate_label)


func _label(pos: Vector2) -> Label:
	var l := Label.new()
	l.position = pos
	l.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.9))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	add_child(l)
	return l


func set_range(lo: float, hi: float) -> void:
	slider.min_value = lo
	slider.max_value = maxf(hi, lo + 1e-6)


func refresh(t: float, clock_rate: float, paused: bool, lines: PackedStringArray) -> void:
	if not _dragging_slider:
		slider.set_value_no_signal(t)
	play_button.text = "Play" if paused else "Pause"
	rate_label.text = "%sx" % _fmt_rate(clock_rate)
	status.text = "\n".join(lines)


static func _fmt_rate(r: float) -> String:
	return ("%.2f" % r).rstrip("0").rstrip(".") if absf(r) < 10.0 else "%d" % int(r)
