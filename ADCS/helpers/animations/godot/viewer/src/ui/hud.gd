extends Control
## Heads-up display: status (top left), help (H), timeline dock (bottom) and the
## state card above it while paused.

const Timeline := preload("res://src/ui/timeline/timeline.gd")
const StateCard := preload("res://src/ui/timeline/state_card.gd")
const Layers := preload("res://src/ui/layers.gd")

var status: Label
var help: Label
var timeline: Timeline
var card: StateCard
var layers: Layers
var focus: Label
var settings: MenuButton
var quality_menu: MenuButton
var model_menu: MenuButton
var quality_row: HBoxContainer
var exposure_label: Label
var quality_prefix: Label
var fps_status: Label
var branding: Label
var settings_world: RefCounted
var asset_notice: Label
var model_handler: Callable
var model_names := ["box", "simulation", "cloudsat", "cubesat_1ru", "cubesat_2ru", "icecube", "mirata", "icesat2"]
var model_labels := ["Box", "Simulation", "CloudSat (C)", "CubeSat 1 RU", "CubeSat 2 RU", "CubeSat - ICECube", "CubeSat MiRaTa", "ICESat-2"]

const HELP := """Mouse: drag to orbit, wheel to zoom
Space  play / pause        R  reset time        , .  slower / faster
1  spacecraft, LVLH        2  spacecraft, inertial
3  spacecraft, body        4  Earth-centred
L  follow live             Home  reset view
V  vectors   A  axes   G  tracks   O  all overlays   H  this help
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
	focus = _label(Vector2(16, 12))
	focus.text = "FOCUS: SPACECRAFT"
	status.position.y = 40
	asset_notice = _label(Vector2.ZERO)
	asset_notice.text = "An asset is missing. Check the terminal to approve downloading it."
	asset_notice.add_theme_font_size_override("font_size", 20)
	asset_notice.add_theme_color_override("font_color", Color(1.0, 0.86, 0.48))
	asset_notice.visible = false
	quality_row = HBoxContainer.new()
	quality_row.position = Vector2(16, 0)
	quality_row.add_theme_constant_override("separation", 8)
	add_child(quality_row)
	exposure_label = Label.new()
	quality_row.add_child(exposure_label)
	quality_prefix = Label.new()
	quality_prefix.text = "quality:"
	quality_row.add_child(quality_prefix)
	quality_menu = MenuButton.new()
	quality_menu.text = "Low  ▾"
	quality_menu.custom_minimum_size = Vector2(104, 28)
	var quality_style := StyleBoxFlat.new()
	quality_style.bg_color = Color(0.12, 0.16, 0.22, 0.94)
	quality_style.border_color = Color(0.58, 0.68, 0.82, 0.9)
	quality_style.set_border_width_all(1)
	quality_style.set_corner_radius_all(7)
	quality_menu.add_theme_stylebox_override("normal", quality_style)
	var quality_hover := quality_style.duplicate() as StyleBoxFlat
	quality_hover.bg_color = Color(0.2, 0.27, 0.36, 0.98)
	quality_menu.add_theme_stylebox_override("hover", quality_hover)
	quality_menu.add_theme_stylebox_override("pressed", quality_hover)
	quality_row.add_child(quality_menu)
	var quality_popup := quality_menu.get_popup()
	quality_popup.add_item("Low", 0)
	quality_popup.add_item("Balanced", 1)
	quality_popup.add_item("High", 2)
	quality_popup.id_pressed.connect(_on_quality_selected)
	var model_prefix := Label.new()
	model_prefix.text = "model:"
	quality_row.add_child(model_prefix)
	model_menu = MenuButton.new()
	model_menu.text = "Simulation  ▾"
	model_menu.custom_minimum_size = Vector2(150, 28)
	model_menu.add_theme_stylebox_override("normal", quality_style)
	model_menu.add_theme_stylebox_override("hover", quality_hover)
	model_menu.add_theme_stylebox_override("pressed", quality_hover)
	quality_row.add_child(model_menu)
	var model_popup := model_menu.get_popup()
	for i in model_labels.size():
		model_popup.add_item(model_labels[i], i)
	model_popup.id_pressed.connect(_on_model_selected)
	fps_status = _label(Vector2(16, 0))
	settings = MenuButton.new()
	settings.text = "Settings"
	settings.anchor_left = 1.0
	settings.anchor_right = 1.0
	settings.offset_left = -105
	settings.offset_right = -12
	settings.offset_top = 10
	settings.offset_bottom = 38
	add_child(settings)
	var menu := settings.get_popup()
	for item: String in ["Vectors", "Axes", "Tracks", "All overlays"]:
		menu.add_item(item)
	menu.id_pressed.connect(_on_setting)
	help = _label(Vector2(16, 150))
	help.text = HELP
	help.visible = false
	timeline = Timeline.new()
	timeline.z_index = 2
	add_child(timeline)
	card = StateCard.new()
	card.z_index = 1
	add_child(card)
	branding = _label(Vector2.ZERO)
	branding.z_index = 3
	branding.add_theme_color_override("font_color", Color(0.76, 0.81, 0.88, 0.92))
	branding.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.9))
	branding.add_theme_font_size_override("font_size", 13)
	layers = Layers.new()
	add_child(layers)


func setup_settings(world: RefCounted) -> void:
	settings_world = world


func setup_model_handler(handler: Callable, selected: String = "simulation") -> void:
	model_handler = handler
	_update_model_label(selected)


func show_asset_notice(show: bool) -> void:
	asset_notice.visible = show
	if show:
		asset_notice.position = Vector2((size.x - asset_notice.get_minimum_size().x) * 0.5, 54)


func _on_setting(id: int) -> void:
	if id == 0:
		settings_world.overlays.toggle_group("vectors")
	elif id == 1:
		settings_world.overlays.toggle_group("axes")
	elif id == 2:
		settings_world.overlays.toggle_group("tracks")
	else:
		settings_world.overlays.toggle_all()


func _on_quality_selected(id: int) -> void:
	settings_world.set_quality(["fast", "balanced", "high"][clampi(id, 0, 2)])
	_update_quality_label()


func _on_model_selected(id: int) -> void:
	if id >= 0 and id < model_names.size() and model_handler.is_valid():
		model_handler.call(model_names[id])


func _update_model_label(name: String) -> void:
	var index := model_names.find(name)
	if index >= 0 and model_menu != null:
		model_menu.text = model_labels[index] + "  ▾"


func _update_quality_label() -> void:
	if settings_world == null:
		return
	var labels := {"fast": "Low", "balanced": "Balanced", "high": "High"}
	var next_text := String(labels.get(settings_world.quality, "Low")) + "  ▾"
	if quality_menu.text != next_text:
		quality_menu.text = next_text


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
	var version := String(source.manifest.get("generator", {}).get("version", ""))
	branding.text = "Generalized ADCS" + ("  v" + version if version != "" else "")
	branding.size = branding.get_combined_minimum_size()


func refresh(clock: RefCounted, frame: Dictionary, lines: PackedStringArray) -> void:
	var visible_lines := lines.duplicate()
	fps_status.text = ""
	if not visible_lines.is_empty():
		fps_status.text = visible_lines[visible_lines.size() - 1]
		visible_lines.remove_at(visible_lines.size() - 1)
	status.text = "\n".join(visible_lines)
	status.size = status.get_combined_minimum_size()
	exposure_label.text = "Exposure: %s %+.0f EV" % [settings_world.exposure_mode, settings_world.exposure_ev]
	quality_row.size = quality_row.get_combined_minimum_size()
	quality_row.position.y = status.position.y + status.size.y + 2.0
	fps_status.position.y = quality_row.position.y + maxf(quality_row.get_combined_minimum_size().y, 28.0)
	fps_status.size = fps_status.get_combined_minimum_size()
	_update_quality_label()
	timeline.refresh()
	layers.refresh()
	var paused: bool = clock.paused and not clock.follow_live
	if card.theme == null or not is_equal_approx(card.get_meta("ui", 0.0), timeline.ui):
		card.theme = timeline.Th.ui_theme(timeline.ui)
		card.set_meta("ui", timeline.ui)
	card.show_at(paused, clock.t, frame, timeline.lanes.at(clock.t) if paused else {})
	if card.visible:
		card.reset_size()
		card.position = Vector2(timeline.position.x + timeline.size.x - card.size.x,
			timeline.position.y - card.size.y - 10.0 * timeline.ui)
	branding.size = branding.get_combined_minimum_size()
	branding.position = Vector2(timeline.position.x, timeline.position.y - branding.size.y - 7.0 * timeline.ui)
