extends Control
## Heads-up display: status (top left), help (H), timeline dock (bottom) and the
## state card above it while paused.

const Timeline := preload("res://src/ui/timeline/timeline.gd")
const StateCard := preload("res://src/ui/timeline/state_card.gd")

var status: Label
var help: Label
var timeline: Timeline
var card: StateCard

const HELP := """Mouse: drag to orbit, wheel to zoom
Space  play / pause        , .  slower / faster
1  spacecraft, LVLH        2  spacecraft, inertial
3  spacecraft, body        4  Earth-centred
L  follow live             Home  reset view
A  axes   O  all overlays   H  this help
E  physical / artistic exposure    [ ]  exposure -/+ 1 EV
Q  quality: high / balanced / fast

Timeline: drag to scrub, wheel to zoom, Shift+wheel or right-drag to pan
Left / Right  previous / next sample
Shift+Left / Right  previous / next event or bookmark
B  bookmark   T  type a time   F  fit the run   I  state card"""


func _ready() -> void:
	# The parent is not a Control, so anchors alone leave this at zero size: follow the viewport.
	set_anchors_preset(Control.PRESET_TOP_LEFT)
	size = get_viewport_rect().size
	get_viewport().size_changed.connect(func() -> void: size = get_viewport_rect().size)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	status = _label(Vector2(16, 12))
	help = _label(Vector2(16, 150))
	help.text = HELP
	help.visible = false
	timeline = Timeline.new()
	add_child(timeline)
	card = StateCard.new()
	add_child(card)


func _label(pos: Vector2) -> Label:
	var l := Label.new()
	l.position = pos
	l.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.9))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	add_child(l)
	return l


## Connect the timeline to the clock and the data (after the source is open).
func setup(clock: RefCounted, source: RefCounted) -> void:
	timeline.setup(clock, source)
	card.epoch = timeline.epoch


func refresh(clock: RefCounted, frame: Dictionary, lines: PackedStringArray) -> void:
	status.text = "\n".join(lines)
	timeline.refresh()
	var paused: bool = clock.paused and not clock.follow_live
	if card.theme == null or not is_equal_approx(card.get_meta("ui", 0.0), timeline.ui):
		card.theme = timeline.Th.ui_theme(timeline.ui)
		card.set_meta("ui", timeline.ui)
	card.show_at(paused, clock.t, frame, timeline.lanes.at(clock.t) if paused else {})
	if card.visible:
		card.reset_size()
		card.position = Vector2(timeline.position.x + timeline.size.x - card.size.x,
			timeline.position.y - card.size.y - 10.0 * timeline.ui)
