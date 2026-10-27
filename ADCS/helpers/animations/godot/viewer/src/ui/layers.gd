extends Control
## Overlay menus (top right). Keys V, A, G toggle a whole group; each menu
## controls its individual overlays.

const Th := preload("res://src/ui/timeline/theme.gd")
const KEYS := {"vectors": "V", "axes": "A", "tracks": "G"}
const NAMES := {"vectors": "Vectors", "axes": "Axes", "tracks": "Tracks"}

var overlays: RefCounted  ## OverlayManager
var ui := 1.0
var font: Font
var _chips: Array = []  ## [{group, rect}]
var _hover := ""
var _key := []  ## what was drawn, to redraw only on change
var _menus: Dictionary = {}


func _ready() -> void:
	font = Th.font()
	mouse_filter = Control.MOUSE_FILTER_STOP
	for g: String in ["vectors", "axes", "tracks"]:
		var menu := PopupMenu.new()
		add_child(menu)
		_menus[g] = menu
		menu.id_pressed.connect(func(id: int) -> void:
			var keys: Array = menu.get_meta("keys", [])
			if id >= 0 and id < keys.size():
				overlays.toggle_item(g, String(keys[id]))
				queue_redraw())
	get_viewport().size_changed.connect(_relayout)


func setup(manager: RefCounted) -> void:
	overlays = manager
	_relayout()


func _relayout() -> void:
	if overlays == null or not is_inside_tree():
		return
	ui = clampf(get_viewport_rect().size.y / 1080.0, 1.0, 2.0)
	var fs := int(round(Th.FONT_SIZE * ui))
	var pad := 6.0 * ui
	var h := 30.0 * ui
	var x := pad
	_chips.clear()
	for g: String in overlays.groups():
		var w := 46.0 * ui + font.get_string_size(NAMES[g], HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x + 12.0 * ui
		_chips.append({"group": g, "rect": Rect2(x, pad, w, h)})
		x += w + 4.0 * ui
	visible = not _chips.is_empty()
	size = Vector2(x - 4.0 * ui + pad, h + 2.0 * pad)
	position = Vector2(get_viewport_rect().size.x - size.x - 16.0 * ui, 12.0 * ui)
	queue_redraw()


## Redraw only when a group's state or the hover changes (called every frame by the HUD).
func refresh() -> void:
	if overlays == null:
		return
	var states := []
	for g: String in overlays.groups():
		for item: Dictionary in overlays.menu_items(g):
			states.append([item["key"], item["checked"]])
	var key := [overlays.shown.duplicate(), states, _hover, ui]
	if key != _key:
		_key = key
		queue_redraw()


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var h := _chip_at((event as InputEventMouseMotion).position)
		if h != _hover:
			_hover = h
			tooltip_text = "%s options (%s toggles all)" % [NAMES[h], KEYS[h]] if h != "" else ""
			queue_redraw()
	elif event is InputEventMouseButton:
		var e := event as InputEventMouseButton
		if e.pressed and e.button_index == MOUSE_BUTTON_LEFT:
			var g := _chip_at(e.position)
			if g != "":
				_open_menu(g)
		accept_event()


func _open_menu(group: String) -> void:
	var menu: PopupMenu = _menus[group]
	menu.clear()
	var keys := []
	for item: Dictionary in overlays.menu_items(group):
		var id := keys.size()
		keys.append(item["key"])
		menu.add_check_item(String(item["label"]), id)
		menu.set_item_checked(id, bool(item["checked"]))
	menu.set_meta("keys", keys)
	var r := Rect2()
	for chip: Dictionary in _chips:
		if chip["group"] == group:
			r = chip["rect"]
			break
	menu.position = Vector2i(global_position + Vector2(r.position.x, r.end.y + 4.0 * ui))
	menu.popup()


func _notification(what: int) -> void:
	if what == NOTIFICATION_MOUSE_EXIT and _hover != "":
		_hover = ""
		queue_redraw()


func _chip_at(p: Vector2) -> String:
	for c: Dictionary in _chips:
		if (c["rect"] as Rect2).has_point(p):
			return c["group"]
	return ""


func _draw() -> void:
	if overlays == null or _chips.is_empty():
		return
	draw_style_box(Th.panel_box(ui), Rect2(Vector2.ZERO, size))
	var fs := int(round(Th.FONT_SIZE * ui))
	for c: Dictionary in _chips:
		var r: Rect2 = c["rect"]
		var g: String = c["group"]
		var on := bool(overlays.shown.get(g, false))
		var hot := g == _hover
		var box := StyleBoxFlat.new()
		box.set_corner_radius_all(int(r.size.y / 2))
		box.bg_color = Th.ACCENT_SOFT if on else (Th.BUTTON_HOVER if hot else Color(0, 0, 0, 0))
		box.border_color = Color(Th.ACCENT, 0.55) if on else Color(1, 1, 1, 0.10)
		box.set_border_width_all(1)
		draw_style_box(box, r)
		var col := Th.TEXT if on else (Th.TEXT_DIM if hot else Th.TEXT_FAINT)
		_icon(g, Vector2(r.position.x + 17.0 * ui, r.get_center().y), col if not on else Th.ACCENT)
		draw_string(font, Vector2(r.position.x + 30.0 * ui, r.get_center().y + fs * 0.36), NAMES[g],
			HORIZONTAL_ALIGNMENT_LEFT, -1, fs, col)
		var cx := r.end.x - 13.0 * ui
		var cy := r.get_center().y
		draw_colored_polygon(PackedVector2Array([Vector2(cx - 4.0 * ui, cy - 2.0 * ui),
			Vector2(cx + 4.0 * ui, cy - 2.0 * ui), Vector2(cx, cy + 3.0 * ui)]), col)


func _icon(group: String, c: Vector2, col: Color) -> void:
	var s := 6.0 * ui
	var w := 1.6 * ui
	match group:
		"vectors":  # an arrow
			var a := c + Vector2(-s, s * 0.7)
			var b := c + Vector2(s, -s * 0.7)
			draw_line(a, b, col, w)
			var d := (b - a).normalized()
			var n := Vector2(-d.y, d.x)
			draw_colored_polygon(PackedVector2Array([b + d * 1.5 * ui, b - d * 4.5 * ui + n * 3.0 * ui, b - d * 4.5 * ui - n * 3.0 * ui]), col)
		"axes":  # three axes from a corner
			var o := c + Vector2(-s * 0.6, s * 0.6)
			draw_line(o, o + Vector2(s * 1.6, 0), col, w)
			draw_line(o, o + Vector2(0, -s * 1.6), col, w)
			draw_line(o, o + Vector2(s * 1.0, -s * 1.0), col, w)
		"tracks":  # an orbit arc
			draw_arc(c + Vector2(0, s * 0.9), s * 1.2, PI * 1.05, PI * 1.95, 12, col, w)
			draw_circle(c + Vector2(s * 0.85, s * 0.05), 1.8 * ui, col)
