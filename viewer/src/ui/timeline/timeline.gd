extends Control
## The timeline dock at the bottom of the viewer.
##
##   [play][<|][|>][60x]      2026-03-14 12:41:07.2 UTC  T+00:41:07 · sample 1234     [copy][+bm][list][fit]
##   Sun     |  ░░░sunlit░░░▓▓umbra▓▓░░░░  (ruler with UTC ticks above, bookmark pins on it)
##   Boston  |        ▬▬▬                  (one lane per ground site: passes)
##           [====window====]              (whole run; the visible window)
##
## Mouse: drag on the track to scrub (snaps to logged samples when they are a
## few pixels apart), wheel to zoom about the cursor, Shift+wheel or right/middle
## drag to pan, double-click to fit. Click the time to type one.

const Th := preload("res://src/ui/timeline/theme.gd")
const TF := preload("res://src/ui/timeline/time_fmt.gd")
const Ticks := preload("res://src/ui/timeline/ticks.gd")
const Lanes := preload("res://src/ui/timeline/lanes.gd")
const Bookmarks := preload("res://src/ui/timeline/bookmarks.gd")
const Att := preload("res://src/ui/timeline/attitude.gd")
const RATES: Array[float] = [-1000.0, -100.0, -10.0, -1.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
	120.0, 300.0, 600.0, 1800.0, 3600.0, 10000.0]

const MIN_SPAN := 1.0  ## s, deepest zoom
const TIPS := {"play": "Play / pause (Space)", "back": "Previous sample (Left)", "fwd": "Next sample (Right)",
	"speed": "Playback speed (, .)", "live": "Follow live data (L)", "fit": "Show the whole run (F)",
	"list": "Bookmarks", "bookmark": "Bookmark this moment (B)", "copy": "Copy the time"}
const SNAP_PX := 5.0

var clock: RefCounted
var store: RefCounted
var live := false
var epoch := NAN  ## unix time of t = 0
var lanes := Lanes.new()
var bookmarks := Bookmarks.new()
var ui := 1.0  ## UI scale (1 at 1080p)
var font: Font

var view0 := 0.0  ## visible window [s since epoch]
var view1 := 1.0
var fit := true  ## window follows the whole run

var _rects := {}  ## layout: name -> Rect2
var _buttons: Array = []  ## [{name, rect, tip}]
var _lane_rows: Array = []  ## [{label, rect, kind: "sky"|"site", i}]
var _hover := Vector2(-1, -1)
var _hover_btn := ""
var _hover_pin := ""
var _hover_frame: Dictionary = {}
var _hover_t := NAN
var _drag := ""  ## "scrub" | "pan" | "overview"
var _drag_x0 := 0.0
var _drag_view := Vector2.ZERO
var _was_paused := false
var _ticks: Array = []
var _ticks_key := Vector3.ZERO
var _error_until := 0
var draw_us := 0  ## drawing cost this frame, both layers [us]
var static_draws := 0  ## how often the static layer was redrawn
var _ci: CanvasItem  ## canvas the draw helpers paint on (this control or the dynamic layer)
var _dyn: Layer
var _static_key := []
var _hover_lines: Array = []  ## hover card text, computed when the hover moves
var _hover_x_t := 0.0  ## x of the hover line / card
var _time_w := Vector2.ZERO  ## (length, width) of the time readout


class Layer:
	extends Control
	var paint: Callable

	func _draw() -> void:
		paint.call(self)
var _box := {}  ## cached style boxes, rebuilt on relayout

var time_edit: LineEdit
var list_popup: PopupPanel
var _list_box: VBoxContainer
var _list_scroll: ScrollContainer
var speed_menu: PopupMenu


func setup(data_clock: RefCounted, source: RefCounted) -> void:
	clock = data_clock
	store = source.store
	live = source.live
	epoch = TF.epoch_unix(source.manifest)
	var tl: Dictionary = source.scene.get("timeline", {})
	lanes.setup(store, tl)
	var first: PackedFloat64Array = store.get_row("r_sc", 0) if store.n_rows > 0 else PackedFloat64Array()
	bookmarks.setup(source.manifest, source.scene, String(source.run_id), first)
	bookmarks.changed.connect(_rebuild_list)
	bookmarks.changed.connect(queue_redraw)
	view0 = clock.t_min
	view1 = maxf(clock.t_max, clock.t_min + MIN_SPAN)
	_relayout()


func _ready() -> void:
	font = Th.font()
	mouse_filter = Control.MOUSE_FILTER_STOP
	set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	time_edit = LineEdit.new()
	time_edit.visible = false
	time_edit.placeholder_text = "2026-03-14 12:41:07, 12:41, T+41:07, +2460"
	time_edit.text_submitted.connect(_on_time_submitted)
	time_edit.text_changed.connect(func(_s: String) -> void: _clear_time_error())
	time_edit.focus_exited.connect(func() -> void: time_edit.visible = false)
	time_edit.gui_input.connect(func(e: InputEvent) -> void:
		if e is InputEventKey and (e as InputEventKey).pressed and (e as InputEventKey).keycode == KEY_ESCAPE:
			time_edit.visible = false)
	_dyn = Layer.new()
	_dyn.paint = _draw_dynamic
	_dyn.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_dyn.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(_dyn)
	add_child(time_edit)
	speed_menu = PopupMenu.new()
	for r in RATES:
		speed_menu.add_item(_fmt_rate(r) + "x")
	speed_menu.id_pressed.connect(func(i: int) -> void: clock.rate = RATES[i])
	add_child(speed_menu)
	_build_list_popup()
	get_viewport().size_changed.connect(_relayout)
	_relayout()


# ------------------------------------------------------------------ layout
func _relayout() -> void:
	if not is_inside_tree():
		return
	ui = clampf(get_viewport_rect().size.y / 1080.0, 1.0, 2.0)
	var pad := 12.0 * ui
	var n_lanes := 1 + lanes.sites.size()
	var h := pad + 32.0 * ui + 8.0 * ui + 24.0 * ui + n_lanes * 16.0 * ui + 8.0 * ui + 6.0 * ui + pad
	offset_left = 16.0 * ui
	offset_right = -16.0 * ui
	offset_bottom = -12.0 * ui
	offset_top = offset_bottom - h
	var w := get_viewport_rect().size.x - 32.0 * ui
	_rects.clear()
	_rects["panel"] = Rect2(0, 0, w, h)
	var top := Rect2(pad, pad, w - 2 * pad, 32.0 * ui)
	_rects["top"] = top
	_buttons.clear()
	var b := 32.0 * ui
	var x := top.position.x
	for bn: String in ["play", "back", "fwd"]:
		_buttons.append({"name": bn, "rect": Rect2(x, top.position.y, b, b)})
		x += b + 2.0 * ui
	_buttons.append({"name": "speed", "rect": Rect2(x + 4.0 * ui, top.position.y + 4.0 * ui, 62.0 * ui, b - 8.0 * ui)})
	x += 70.0 * ui
	if live:
		_buttons.append({"name": "live", "rect": Rect2(x, top.position.y + 4.0 * ui, 58.0 * ui, b - 8.0 * ui)})
	var xr := top.end.x
	for bn: String in ["fit", "list", "bookmark", "copy"]:
		xr -= b
		_buttons.append({"name": bn, "rect": Rect2(xr, top.position.y, b, b)})
		xr -= 2.0 * ui
	_rects["time"] = Rect2(x + 76.0 * ui, top.position.y, xr - x - 90.0 * ui, b)
	var gutter := 70.0 * ui
	var y := top.end.y + 8.0 * ui
	var track_x := pad + gutter
	var track_w := w - pad - track_x
	_rects["ruler"] = Rect2(track_x, y, track_w, 24.0 * ui)
	y += 24.0 * ui
	_lane_rows.clear()
	_lane_rows.append({"label": "Sun", "rect": Rect2(track_x, y + 3.0 * ui, track_w, 10.0 * ui), "kind": "sky", "i": 0})
	y += 16.0 * ui
	for i in lanes.sites.size():
		_lane_rows.append({"label": String(lanes.sites[i].get("name", "site")), "kind": "site", "i": i,
			"rect": Rect2(track_x, y + 3.0 * ui, track_w, 10.0 * ui)})
		y += 16.0 * ui
	_rects["track"] = Rect2(track_x, _rects["ruler"].position.y, track_w, y - _rects["ruler"].position.y)
	_rects["overview"] = Rect2(track_x, y + 8.0 * ui, track_w, 6.0 * ui)
	_rects["gutter_x"] = Rect2(pad, 0, gutter, 0)
	_box = {"panel": Th.panel_box(ui), "lane": _round(Th.TRACK, 5.0 * ui), "pill": _round(Th.BUTTON_HOVER, 12.0 * ui),
		"pill_hot": _round(Th.BUTTON_ACTIVE, 12.0 * ui), "pill_on": _round(Th.ACCENT_SOFT, 12.0 * ui),
		"hover": _round(Th.BUTTON_HOVER, 8.0 * ui), "over": _round(Th.TRACK, 3.0 * ui),
		"win": _round(Th.ACCENT_SOFT, 3.0 * ui), "win_fit": _round(Color(1, 1, 1, 0.06), 3.0 * ui)}
	var card := _round(Color(0.06, 0.07, 0.09, 0.96), 8.0 * ui)
	card.border_color = Th.PANEL_BORDER
	card.set_border_width_all(1)
	_box["card"] = card
	for i in lanes.sites.size():
		_box["site%d" % i] = _round(Color.from_string(String(lanes.sites[i].get("color", "#7cc6ff")), Th.ACCENT), 5.0 * ui)
	_time_w = Vector2.ZERO
	var th := Th.ui_theme(ui)
	for c: Node in [time_edit, list_popup, speed_menu]:
		c.set("theme", th)
	time_edit.position = _rects["time"].position + Vector2(0, 2.0 * ui)
	time_edit.size = Vector2(minf(_rects["time"].size.x, 360.0 * ui), b - 4.0 * ui)
	queue_redraw()


static func _round(c: Color, radius: float) -> StyleBoxFlat:
	var b := StyleBoxFlat.new()
	b.bg_color = c
	b.set_corner_radius_all(int(radius))
	return b


# ------------------------------------------------------------------ mapping
func t_to_x(t: float) -> float:
	var r: Rect2 = _rects["track"]
	return r.position.x + (t - view0) / maxf(view1 - view0, 1e-9) * r.size.x


func x_to_t(x: float) -> float:
	var r: Rect2 = _rects["track"]
	return view0 + (x - r.position.x) / maxf(r.size.x, 1e-9) * (view1 - view0)


func set_view(t0: float, t1: float) -> void:
	var lo: float = clock.t_min
	var hi: float = maxf(clock.t_max, lo + MIN_SPAN)
	var span := clampf(t1 - t0, MIN_SPAN, hi - lo)
	fit = span >= (hi - lo) * 0.9999
	if fit:
		view0 = lo
		view1 = hi
	else:
		view0 = clampf(t0, lo, hi - span)
		view1 = view0 + span
	queue_redraw()


func zoom_at(t_anchor: float, factor: float) -> void:
	var span := (view1 - view0) * factor
	var a := (t_anchor - view0) / maxf(view1 - view0, 1e-9)
	set_view(t_anchor - a * span, t_anchor - a * span + span)


## Nearest logged sample to t if it is within SNAP_PX on screen, else t.
func snap(t: float) -> float:
	if store.n_rows < 2:
		return t
	var ks: Array = store.locate(t)
	var k: int = ks[0] if float(ks[1]) < 0.5 else mini(int(ks[0]) + 1, store.n_rows - 1)
	var tk: float = store.t(k)
	return tk if absf(t_to_x(tk) - t_to_x(t)) <= SNAP_PX * ui else t


func seek(t: float) -> void:
	clock.seek(t)
	queue_redraw()


func fit_view() -> void:
	set_view(clock.t_min, clock.t_max)


# ------------------------------------------------------------------ per frame
## Called by the HUD every frame.
func refresh() -> void:
	var changed := lanes.update()
	if fit:
		view0 = clock.t_min
		view1 = maxf(clock.t_max, clock.t_min + MIN_SPAN)
	elif _drag == "" and (not clock.paused or clock.follow_live) and (clock.t > view1 or clock.t < view0):
		var span := view1 - view0  # playing: page so the playhead is near the left edge
		set_view(clock.t - 0.1 * span, clock.t + 0.9 * span)
	if changed and lanes.sites.size() + 1 != _lane_rows.size():
		_relayout()
	draw_us = 0
	var key := [view0, view1, size.x, ui, _hover_btn, _hover_pin, clock.paused, clock.rate, clock.follow_live,
		bookmarks.items.size()]
	if changed or key != _static_key:
		_static_key = key
		queue_redraw()
	_dyn.queue_redraw()


# ------------------------------------------------------------------ input
func _gui_input(event: InputEvent) -> void:
	if store == null:
		return
	if event is InputEventMouseMotion:
		var m := event as InputEventMouseMotion
		_hover = m.position
		_update_hover()
		match _drag:
			"scrub":
				seek(snap(clampf(x_to_t(m.position.x), clock.t_min, clock.t_max)))
			"pan":
				var dt: float = (m.position.x - _drag_x0) / _rects["track"].size.x * (_drag_view.y - _drag_view.x)
				set_view(_drag_view.x - dt, _drag_view.y - dt)
			"overview":
				_overview_to(m.position.x)
		queue_redraw()
		accept_event()
	elif event is InputEventMouseButton:
		var e := event as InputEventMouseButton
		var in_track: bool = _rects["track"].grow(4.0 * ui).has_point(e.position)
		var in_over: bool = _rects["overview"].grow_individual(0, 6.0 * ui, 0, 6.0 * ui).has_point(e.position)
		if e.pressed and e.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN] and (in_track or in_over):
			var up := e.button_index == MOUSE_BUTTON_WHEEL_UP
			if e.shift_pressed:
				var d := (view1 - view0) * 0.12 * (-1.0 if up else 1.0)
				set_view(view0 + d, view1 + d)
			else:
				zoom_at(x_to_t(e.position.x) if in_track else clock.t, 1.0 / 1.25 if up else 1.25)
		elif e.pressed and e.button_index in [MOUSE_BUTTON_WHEEL_LEFT, MOUSE_BUTTON_WHEEL_RIGHT] and in_track:
			var d := (view1 - view0) * 0.12 * (-1.0 if e.button_index == MOUSE_BUTTON_WHEEL_LEFT else 1.0)
			set_view(view0 + d, view1 + d)
		elif e.button_index == MOUSE_BUTTON_LEFT:
			if e.pressed:
				_press(e)
			elif _drag != "":
				if _drag == "scrub":
					clock.paused = _was_paused
				_drag = ""
		elif e.button_index in [MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE] and in_track:
			if e.pressed:
				_drag = "pan"
				_drag_x0 = e.position.x
				_drag_view = Vector2(view0, view1)
			else:
				_drag = ""
		_hover_t = NAN  # the view may have moved under the cursor
		_update_hover()
		queue_redraw()
		accept_event()


func _press(e: InputEventMouseButton) -> void:
	for b: Dictionary in _buttons:
		if (b["rect"] as Rect2).has_point(e.position):
			_button(String(b["name"]))
			return
	if (_rects["time"] as Rect2).has_point(e.position):
		open_time_entry()
		return
	if _hover_pin != "":
		var bm := bookmarks.find(_hover_pin)
		if not bm.is_empty():
			seek(bm["t_s"])
		return
	if _rects["overview"].grow_individual(0, 6.0 * ui, 0, 6.0 * ui).has_point(e.position):
		_drag = "overview"
		_overview_to(e.position.x)
		return
	if (_rects["track"] as Rect2).grow(4.0 * ui).has_point(e.position):
		if e.double_click:
			fit_view()
			return
		if e.shift_pressed:
			_drag = "pan"
			_drag_x0 = e.position.x
			_drag_view = Vector2(view0, view1)
			return
		_drag = "scrub"
		_was_paused = clock.paused
		clock.paused = true
		seek(snap(clampf(x_to_t(e.position.x), clock.t_min, clock.t_max)))


func _overview_to(x: float) -> void:
	var r: Rect2 = _rects["overview"]
	var tc: float = clock.t_min + (x - r.position.x) / r.size.x * (clock.t_max - clock.t_min)
	var span := view1 - view0
	if not fit:
		set_view(tc - span / 2.0, tc + span / 2.0)


func _button(bn: String) -> void:
	match bn:
		"play": clock.toggle_pause()
		"back": step_row(-1)
		"fwd": step_row(1)
		"speed":
			var r: Rect2 = _rect_of("speed")
			speed_menu.position = Vector2i(global_position + r.position + Vector2(0, -speed_menu.get_contents_minimum_size().y - 6))
			speed_menu.popup()
		"live": clock.follow_live = true
		"fit": fit_view()
		"copy": copy_time()
		"bookmark": add_bookmark()
		"list": open_list()


func _rect_of(bn: String) -> Rect2:
	for b: Dictionary in _buttons:
		if b["name"] == bn:
			return b["rect"]
	return Rect2()


func _update_hover() -> void:
	_hover_btn = ""
	for b: Dictionary in _buttons:
		if (b["rect"] as Rect2).has_point(_hover):
			_hover_btn = b["name"]
	_hover_pin = ""
	var ruler: Rect2 = _rects["ruler"]
	if ruler.grow(3.0 * ui).has_point(_hover):
		var best := 6.0 * ui
		for bm: Dictionary in bookmarks.items:
			var d := absf(t_to_x(bm["t_s"]) - _hover.x)
			if d < best:
				best = d
				_hover_pin = bm["id"]
	var in_track: bool = _rects["track"].has_point(_hover)
	var t := x_to_t(_hover.x) if in_track else NAN
	if _hover_pin != "":
		var b := bookmarks.find(_hover_pin)
		_hover_x_t = t_to_x(b["t_s"])
		_hover_lines = [String(b["label"]), "%s · %s" % [
			TF.clock_label(epoch + float(b["t_s"]), true) if not is_nan(epoch) else "", TF.elapsed(b["t_s"])]]
		_hover_t = NAN
	elif in_track and (is_nan(_hover_t) or absf(t_to_x(t) - t_to_x(_hover_t)) >= 1.0):
		_hover_t = clampf(t, clock.t_min, clock.t_max)
		_hover_frame = store.sample(_hover_t)
		_hover_x_t = t_to_x(_hover_t)
		_hover_lines = _hover_text(_hover_t, _hover_frame)
	elif not in_track:
		_hover_t = NAN
		_hover_lines = []


func _hover_text(t: float, f: Dictionary) -> Array:
	var lines := []
	lines.append("%s  ·  %s" % [TF.clock_label(epoch + t, true) + " UTC" if not is_nan(epoch) else "", TF.elapsed(t)])
	var at := lanes.at(t)
	var sky: String = ["Sunlit", "Penumbra", "Eclipse"][int(at["sky"])] if int(at["sky"]) >= 0 else ""
	var d := Att.describe(f)
	lines.append("%s%s altitude %.1f km" % [sky, "  ·" if sky != "" else "", d["altitude_km"]])
	if d.get("pointing_err_deg") != null:
		lines.append("Pointing error %.3f°" % d["pointing_err_deg"])
	for s: String in at["passes"]:
		lines.append("Over %s" % s)
	return lines


func _notification(what: int) -> void:
	if what == NOTIFICATION_MOUSE_EXIT:
		_hover = Vector2(-1, -1)
		_hover_btn = ""
		_hover_pin = ""
		_hover_t = NAN
		_hover_lines = []
		queue_redraw()


# ------------------------------------------------------------------ actions (also bound to keys by the app)
func step_row(direction: int) -> void:
	if store.n_rows < 2:
		return
	var ks: Array = store.locate(clock.t)
	var k: int = ks[0]
	var near: int = k if float(ks[1]) < 0.5 else mini(k + 1, store.n_rows - 1)
	if absf(clock.t - store.t(near)) < 1e-9:  # on a sample: the neighbouring one
		k = near + direction
	else:  # between samples k and k + 1
		k = k + 1 if direction > 0 else k
	clock.paused = true
	seek(store.t(clampi(k, 0, store.n_rows - 1)))


## Previous / next eclipse change, pass rise or set, or bookmark.
func jump_event(direction: int) -> void:
	var times := lanes.event_times()
	for b: Dictionary in bookmarks.items:
		times.append(b["t_s"])
	times.sort()
	var t: float = clock.t
	var target := NAN
	if direction > 0:
		for x in times:
			if x > t + 1e-6:
				target = x
				break
	else:
		for i in range(times.size() - 1, -1, -1):
			if times[i] < t - 1e-6:
				target = times[i]
				break
	if not is_nan(target):
		clock.paused = true
		seek(target)


func add_bookmark() -> void:
	var b := bookmarks.add(clock.t)
	open_list(String(b["id"]))


func copy_time() -> void:
	var text := "%s UTC (t = %.3f s)" % [TF.utc(epoch + clock.t, 3), clock.t] if not is_nan(epoch) else "t = %.3f s" % clock.t
	DisplayServer.clipboard_set(text)


func _clear_time_error() -> void:
	time_edit.remove_theme_color_override("font_color")
	time_edit.tooltip_text = ""


func open_time_entry() -> void:
	_clear_time_error()
	time_edit.text = TF.utc(epoch + clock.t, 0) if not is_nan(epoch) else TF.elapsed(clock.t)
	time_edit.visible = true
	time_edit.grab_focus()
	time_edit.select_all()


func _on_time_submitted(text: String) -> void:
	var t := TF.parse(text, epoch, clock.t)
	if is_nan(t) or t < clock.t_min - 1e-6 or t > clock.t_max + 1e-6:
		_error_until = Time.get_ticks_msec() + 1600
		time_edit.add_theme_color_override("font_color", Th.ERROR)
		time_edit.tooltip_text = "Can't read that time" if is_nan(t) else "Outside the run (%s to %s)" % [
			TF.elapsed(clock.t_min), TF.elapsed(clock.t_max)]
		return
	_clear_time_error()
	time_edit.visible = false
	clock.paused = true
	seek(t)
	if t < view0 or t > view1:
		var span := view1 - view0
		set_view(t - span / 2.0, t + span / 2.0)


# ------------------------------------------------------------------ bookmark list
func _build_list_popup() -> void:
	list_popup = PopupPanel.new()
	var root := VBoxContainer.new()
	root.custom_minimum_size = Vector2(380, 0)
	var head := HBoxContainer.new()
	var title := Label.new()
	title.text = "Bookmarks"
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(title)
	var export_btn := Button.new()
	export_btn.text = "Copy all as JSON"
	export_btn.flat = true
	export_btn.pressed.connect(func() -> void:
		DisplayServer.clipboard_set(bookmarks.export_json(func(t: float) -> Dictionary:
			var d := Att.describe(store.sample(t))
			d["utc"] = TF.utc(epoch + t, 3) if not is_nan(epoch) else null
			return d)))
	head.add_child(export_btn)
	root.add_child(head)
	var scroll := ScrollContainer.new()
	_list_scroll = scroll
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_list_box = VBoxContainer.new()
	_list_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_list_box)
	root.add_child(scroll)
	var hint := Label.new()
	hint.text = "B adds a bookmark at the playhead · bookmarks from the log are locked"
	hint.add_theme_color_override("font_color", Th.TEXT_FAINT)
	hint.add_theme_font_size_override("font_size", Th.FONT_SMALL)
	root.add_child(hint)
	list_popup.add_child(root)
	add_child(list_popup)


func _rebuild_list(focus_id: String = "") -> void:
	if _list_box == null:
		return
	for c in _list_box.get_children():
		c.queue_free()
	if bookmarks.items.is_empty():
		var none := Label.new()
		none.text = "No bookmarks yet."
		none.add_theme_color_override("font_color", Th.TEXT_DIM)
		_list_box.add_child(none)
	for b: Dictionary in bookmarks.items:
		var row := HBoxContainer.new()
		var dot := ColorRect.new()
		dot.color = Color.from_string(String(b["color"]), Th.BOOKMARK)
		dot.custom_minimum_size = Vector2(6, 18)
		row.add_child(dot)
		var go := Button.new()
		go.flat = true
		go.add_theme_font_override("font", font)
		go.text = TF.clock_label(epoch + float(b["t_s"]), true) if not is_nan(epoch) else TF.elapsed(b["t_s"])
		go.tooltip_text = "%s · %s" % [TF.utc(epoch + float(b["t_s"]), 1), TF.elapsed(b["t_s"])]
		var t: float = b["t_s"]
		go.pressed.connect(func() -> void:
			clock.paused = true
			seek(t))
		row.add_child(go)
		var edit := LineEdit.new()
		edit.text = String(b["label"])
		edit.editable = not b["locked"]
		edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		edit.flat = true
		var id := String(b["id"])
		edit.text_submitted.connect(func(txt: String) -> void: bookmarks.rename(id, txt))
		edit.focus_exited.connect(func() -> void:
			if edit.editable and edit.text != String(bookmarks.find(id).get("label", edit.text)):
				bookmarks.rename(id, edit.text))
		row.add_child(edit)
		if b["locked"]:
			var lock := Label.new()
			lock.text = "log"
			lock.add_theme_color_override("font_color", Th.TEXT_FAINT)
			row.add_child(lock)
		else:
			var del := Button.new()
			del.text = "×"
			del.flat = true
			del.tooltip_text = "Delete"
			del.pressed.connect(func() -> void: bookmarks.remove(id))
			row.add_child(del)
		_list_box.add_child(row)
		if id == focus_id:
			edit.call_deferred("grab_focus")
			edit.call_deferred("select_all")


func open_list(focus_id: String = "") -> void:
	_rebuild_list(focus_id)
	_list_scroll.custom_minimum_size = Vector2(0, clampf(maxf(bookmarks.items.size(), 1) * 34.0 * ui, 34.0 * ui, 300.0 * ui))
	var r := _rect_of("list")
	list_popup.reset_size()
	var sz := list_popup.get_contents_minimum_size()
	list_popup.position = Vector2i(global_position + Vector2(r.end.x - sz.x, r.position.y - sz.y - 12))
	list_popup.popup()


# ------------------------------------------------------------------ drawing
## Static layer (this control): redrawn only when something on it changes.
func _draw() -> void:
	if store == null or _rects.is_empty():
		return
	var t0 := Time.get_ticks_usec()
	_ci = self
	_ci.draw_style_box(_box["panel"], _rects["panel"])
	_draw_buttons()
	_draw_ruler()
	_draw_lanes()
	_draw_pins()
	_draw_overview()
	draw_us += Time.get_ticks_usec() - t0
	static_draws += 1


## Dynamic layer (a child drawn on top): every frame.
func _draw_dynamic(layer: CanvasItem) -> void:
	if store == null or _rects.is_empty():
		return
	var t0 := Time.get_ticks_usec()
	_ci = layer
	_draw_time()
	_draw_playhead()
	_draw_overview_head()
	_draw_hover()
	_draw_button_tip()
	draw_us += Time.get_ticks_usec() - t0


func _fs(base: int) -> int:
	return int(round(base * ui))


func _text(pos: Vector2, s: String, size: int, color: Color, align := HORIZONTAL_ALIGNMENT_LEFT, width := -1.0) -> void:
	_ci.draw_string(font, pos, s, align, width, _fs(size), color)


func _draw_buttons() -> void:
	for b: Dictionary in _buttons:
		var r: Rect2 = b["rect"]
		var bn: String = b["name"]
		var hot := bn == _hover_btn
		var c := Th.TEXT if hot else Th.TEXT_DIM
		if bn in ["speed", "live"]:
			var on: bool = bn == "live" and clock.follow_live
			_ci.draw_style_box(_box["pill_on"] if on else (_box["pill_hot"] if hot else _box["pill"]), r)
			var label := (_fmt_rate(clock.rate) + "×") if bn == "speed" else "LIVE"
			_text(Vector2(r.position.x, r.position.y + r.size.y * 0.5 + _fs(Th.FONT_SIZE) * 0.36), label, Th.FONT_SIZE,
				Th.ACCENT if on else c, HORIZONTAL_ALIGNMENT_CENTER, r.size.x)
			continue
		if hot:
			_ci.draw_style_box(_box["hover"], r)
		var ctr := r.get_center()
		var s := 7.0 * ui
		match bn:
			"play":
				if clock.paused and not clock.follow_live:
					_ci.draw_colored_polygon(PackedVector2Array([ctr + Vector2(-s * 0.7, -s), ctr + Vector2(s, 0), ctr + Vector2(-s * 0.7, s)]), Th.TEXT)
				else:
					_ci.draw_rect(Rect2(ctr + Vector2(-s * 0.75, -s), Vector2(s * 0.5, 2 * s)), Th.TEXT)
					_ci.draw_rect(Rect2(ctr + Vector2(s * 0.25, -s), Vector2(s * 0.5, 2 * s)), Th.TEXT)
			"back", "fwd":
				var d := -1.0 if bn == "back" else 1.0
				_ci.draw_colored_polygon(PackedVector2Array([ctr + Vector2(-s * 0.6 * d, -s * 0.8), ctr + Vector2(s * 0.5 * d, 0),
					ctr + Vector2(-s * 0.6 * d, s * 0.8)]), c)
				_ci.draw_rect(Rect2(ctr + Vector2(s * 0.55 * d - (0.0 if d > 0 else s * 0.3), -s * 0.8), Vector2(s * 0.3, s * 1.6)), c)
			"bookmark":
				var p := PackedVector2Array([ctr + Vector2(-s * 0.6, -s), ctr + Vector2(s * 0.6, -s), ctr + Vector2(s * 0.6, s),
					ctr + Vector2(0, s * 0.45), ctr + Vector2(-s * 0.6, s)])
				_ci.draw_colored_polygon(p, Th.BOOKMARK if hot else c)
				_ci.draw_line(ctr + Vector2(s * 0.95, -s * 1.05), ctr + Vector2(s * 0.95, -s * 0.25), c, 1.5 * ui)
				_ci.draw_line(ctr + Vector2(s * 0.55, -s * 0.65), ctr + Vector2(s * 1.35, -s * 0.65), c, 1.5 * ui)
			"list":
				for i in 3:
					var y := ctr.y + (i - 1) * s * 0.7
					_ci.draw_circle(Vector2(ctr.x - s * 0.8, y), 1.4 * ui, c)
					_ci.draw_line(Vector2(ctr.x - s * 0.4, y), Vector2(ctr.x + s, y), c, 1.6 * ui)
			"fit":
				_ci.draw_line(ctr + Vector2(-s, -s * 0.8), ctr + Vector2(-s, s * 0.8), c, 1.6 * ui)
				_ci.draw_line(ctr + Vector2(s, -s * 0.8), ctr + Vector2(s, s * 0.8), c, 1.6 * ui)
				_ci.draw_line(ctr + Vector2(-s * 0.6, 0), ctr + Vector2(s * 0.6, 0), c, 1.6 * ui)
				for d: float in [-1.0, 1.0]:
					_ci.draw_line(ctr + Vector2(s * 0.6 * d, 0), ctr + Vector2(s * 0.25 * d, -s * 0.35), c, 1.6 * ui)
					_ci.draw_line(ctr + Vector2(s * 0.6 * d, 0), ctr + Vector2(s * 0.25 * d, s * 0.35), c, 1.6 * ui)
			"copy":
				var r1 := Rect2(ctr + Vector2(-s * 0.85, -s * 0.5), Vector2(s * 1.15, s * 1.35))
				var r2 := Rect2(ctr + Vector2(-s * 0.3, -s * 0.95), Vector2(s * 1.15, s * 1.35))
				_ci.draw_rect(r2, c, false, 1.4 * ui)
				_ci.draw_rect(r1, Th.PANEL, true)
				_ci.draw_rect(r1, c, false, 1.4 * ui)


func _draw_button_tip() -> void:
	if _hover_btn == "":
		return
	var tip: String = TIPS.get(_hover_btn, "")
	if tip != "":
		var r := _rect_of(_hover_btn)
		var w := font.get_string_size(tip, HORIZONTAL_ALIGNMENT_LEFT, -1, _fs(Th.FONT_SMALL)).x + 16.0 * ui
		var x := clampf(r.get_center().x - w / 2.0, 0.0, size.x - w)
		_card(Rect2(x, -30.0 * ui, w, 22.0 * ui))
		_text(Vector2(x + 8.0 * ui, -14.0 * ui), tip, Th.FONT_SMALL, Th.TEXT)



func _card(r: Rect2) -> void:
	_ci.draw_style_box(_box["card"], r)


func _draw_time() -> void:
	var r: Rect2 = _rects["time"]
	if time_edit.visible:
		return
	var base := r.position.y + r.size.y * 0.5 + _fs(Th.FONT_TIME) * 0.36
	var t: float = clock.t
	var main := TF.utc(epoch + t, 1) if not is_nan(epoch) else TF.elapsed(t, 1)
	var x := r.position.x
	var hot := r.has_point(_hover)
	_text(Vector2(x, base), main, Th.FONT_TIME, Th.TEXT)
	if _time_w.x != float(main.length()):  # tabular digits: the width only changes with the length
		_time_w = Vector2(main.length(), font.get_string_size(main, HORIZONTAL_ALIGNMENT_LEFT, -1, _fs(Th.FONT_TIME)).x)
	x += _time_w.y + 6.0 * ui
	if not is_nan(epoch):
		_text(Vector2(x, base), "UTC", Th.FONT_SMALL, Th.TEXT_DIM)
		x += 34.0 * ui
	var ks: Array = store.locate(t)
	var k: int = ks[0] if float(ks[1]) < 0.5 else mini(int(ks[0]) + 1, store.n_rows - 1)
	var on: bool = absf(store.t(k) - t) < 1e-6
	var sub := "%s  ·  %s" % [TF.elapsed(t), ("sample %d / %d" % [k, store.n_rows - 1]) if on else "interpolated"]
	_text(Vector2(x, base), sub, Th.FONT_SMALL, Th.TEXT_DIM if not hot else Th.TEXT)
	if hot:
		_ci.draw_line(Vector2(r.position.x, r.end.y - 3.0 * ui), Vector2(x + 200.0 * ui, r.end.y - 3.0 * ui), Th.ACCENT_SOFT, 1.0)


func _draw_ruler() -> void:
	var r: Rect2 = _rects["ruler"]
	var key := Vector3(view0, view1, r.size.x)
	if key != _ticks_key:
		_ticks = Ticks.make(view0, view1, epoch, r.size.x, 7.0 * ui)
		_ticks_key = key
	var lanes_bottom: float = _rects["track"].end.y
	for tk: Dictionary in _ticks:
		var x := t_to_x(tk["t"])
		if tk["major"]:
			_ci.draw_line(Vector2(x, r.position.y + 2.0 * ui), Vector2(x, lanes_bottom), Th.TICK_MINOR, 1.0)
			_ci.draw_line(Vector2(x, r.position.y + 2.0 * ui), Vector2(x, r.position.y + 12.0 * ui), Th.TICK, 1.0)
			_text(Vector2(x + 4.0 * ui, r.position.y + 11.0 * ui), tk["label"], Th.FONT_SMALL, Th.TEXT_DIM)
		else:
			_ci.draw_line(Vector2(x, r.position.y + 2.0 * ui), Vector2(x, r.position.y + 6.0 * ui), Th.TICK, 1.0)


func _span_rect(lane: Rect2, t0: float, t1: float) -> Rect2:
	var x0 := maxf(t_to_x(t0), lane.position.x)
	var x1 := minf(t_to_x(t1), lane.end.x)
	return Rect2(x0, lane.position.y, maxf(x1 - x0, 1.0), lane.size.y)


func _draw_lanes() -> void:
	var gx: float = _rects["gutter_x"].position.x
	for row: Dictionary in _lane_rows:
		var lane: Rect2 = row["rect"]
		_ci.draw_style_box(_box["lane"], lane)
		_text(Vector2(gx, lane.position.y + lane.size.y * 0.5 + _fs(Th.FONT_SMALL) * 0.36),
			String(row["label"]).left(10), Th.FONT_SMALL, Th.TEXT_DIM)
		if row["kind"] == "sky":
			for s: Array in lanes.eclipse_spans():
				if s[1] < view0 or s[0] > view1:
					continue
				var c: Color = [Th.SUNLIT, Th.PENUMBRA, Th.UMBRA][int(s[2])]
				var sr := _span_rect(lane, s[0], s[1])
				_ci.draw_rect(sr, c)
				if int(s[2]) == Lanes.Light.UMBRA:
					_ci.draw_rect(sr, Th.UMBRA_EDGE, false, 1.0)
		else:
			var i: int = row["i"]
			var box: StyleBoxFlat = _box["site%d" % i]
			for p: Array in lanes.passes(i):
				if p[1] < view0 or p[0] > view1:
					continue
				var sr := _span_rect(lane, p[0], p[1])
				if sr.size.x < lane.size.y:  # too narrow for rounded ends
					_ci.draw_rect(sr, box.bg_color)
				else:
					_ci.draw_style_box(box, sr)
				# Peak elevation as a small tick, if there is room.
				var xm := t_to_x(p[2])
				if sr.size.x > 6.0 * ui and xm > lane.position.x and xm < lane.end.x:
					_ci.draw_line(Vector2(xm, lane.position.y + 2.0 * ui), Vector2(xm, lane.end.y - 2.0 * ui), Color(1, 1, 1, 0.7), 1.0)


func _draw_pins() -> void:
	var r: Rect2 = _rects["ruler"]
	for b: Dictionary in bookmarks.items:
		var t: float = b["t_s"]
		if t < view0 or t > view1:
			continue
		var x := t_to_x(t)
		var c := Color.from_string(String(b["color"]), Th.BOOKMARK)
		var hot := String(b["id"]) == _hover_pin
		var s := (4.5 if hot else 3.5) * ui
		# A diamond between the tick labels and the lanes, with a faint line down through the lanes.
		var yc := r.end.y - s - 0.5 * ui
		_ci.draw_line(Vector2(x, yc), Vector2(x, _rects["track"].end.y), Color(c, 0.35), 1.0)
		_ci.draw_colored_polygon(PackedVector2Array([Vector2(x, yc - s), Vector2(x + s, yc), Vector2(x, yc + s),
			Vector2(x - s, yc)]), c)


func _draw_playhead() -> void:
	var t: float = clock.t
	if t < view0 - 1e-9 or t > view1 + 1e-9:
		return
	var x := t_to_x(t)
	var tr: Rect2 = _rects["track"]
	_ci.draw_line(Vector2(x, tr.position.y), Vector2(x, tr.end.y), Th.ACCENT, 2.0 * ui)
	_ci.draw_circle(Vector2(x, tr.position.y + 1.0 * ui), 4.5 * ui, Th.ACCENT)


func _draw_overview() -> void:
	var r: Rect2 = _rects["overview"]
	_ci.draw_style_box(_box["over"], r)
	var lo: float = clock.t_min
	var span: float = maxf(clock.t_max - lo, 1e-9)
	var xw0 := r.position.x + (view0 - lo) / span * r.size.x
	var xw1 := r.position.x + (view1 - lo) / span * r.size.x
	_ci.draw_style_box(_box["win_fit"] if fit else _box["win"], Rect2(xw0, r.position.y, maxf(xw1 - xw0, 3.0 * ui), r.size.y))


func _draw_overview_head() -> void:
	var r: Rect2 = _rects["overview"]
	var lo: float = clock.t_min
	var span: float = maxf(clock.t_max - lo, 1e-9)
	var xp: float = r.position.x + (clock.t - lo) / span * r.size.x
	_ci.draw_line(Vector2(xp, r.position.y - 1.0 * ui), Vector2(xp, r.end.y + 1.0 * ui), Th.ACCENT, 2.0 * ui)


func _draw_hover() -> void:
	if _hover_lines.is_empty() or _drag == "pan":
		return
	if _hover_pin == "":
		var tr: Rect2 = _rects["track"]
		_ci.draw_line(Vector2(_hover_x_t, tr.position.y), Vector2(_hover_x_t, tr.end.y), Th.HOVER, 1.0)
	_tooltip(_hover_x_t, _hover_lines)


func _tooltip(x: float, lines: Array) -> void:
	var fs := _fs(Th.FONT_SMALL)
	var w := 0.0
	for l: String in lines:
		w = maxf(w, font.get_string_size(l, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x)
	w += 20.0 * ui
	var h := lines.size() * (fs + 5.0 * ui) + 12.0 * ui
	var r := Rect2(clampf(x - w / 2.0, 0.0, size.x - w), -h - 8.0 * ui, w, h)
	_card(r)
	for i in lines.size():
		_text(Vector2(r.position.x + 10.0 * ui, r.position.y + 6.0 * ui + (i + 1) * (fs + 5.0 * ui) - 4.0 * ui), lines[i], Th.FONT_SMALL,
			Th.TEXT if i == 0 else Th.TEXT_DIM)


static func _fmt_rate(r: float) -> String:
	return ("%.2f" % r).rstrip("0").rstrip(".") if absf(r) < 10.0 else "%d" % int(r)
