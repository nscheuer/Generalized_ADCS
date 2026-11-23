extends PanelContainer
## The moment under the playhead, shown while paused (I toggles it): attitude
## relative to LVLH, quaternion, body rates, pointing error, light and passes.

const Th := preload("res://src/ui/timeline/theme.gd")
const TF := preload("res://src/ui/timeline/time_fmt.gd")
const Att := preload("res://src/ui/timeline/attitude.gd")

var enabled := true
var epoch := NAN
var _title: Label
var _sub: Label
var _grid: GridContainer
var _values: Dictionary = {}  ## row name -> value Label
var _shown_t := NAN
var last: Dictionary = {}  ## the values shown (for checks and copying)

const ROWS := ["Roll / pitch / yaw", "Quaternion", "Body rate", "Pointing error", "Altitude", "Light", "Passes"]


func _ready() -> void:
	var font := Th.font()
	theme = Th.ui_theme(1.0)
	mouse_filter = Control.MOUSE_FILTER_STOP
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 4)
	add_child(box)
	var head := HBoxContainer.new()
	_title = Label.new()
	_title.add_theme_font_override("font", font)
	_title.add_theme_color_override("font_color", Th.TEXT)
	_title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	head.add_child(_title)
	var copy := Button.new()
	copy.text = "Copy"
	copy.flat = true
	copy.tooltip_text = "Copy this state as JSON"
	copy.pressed.connect(func() -> void: DisplayServer.clipboard_set(JSON.stringify(last, " ")))
	head.add_child(copy)
	box.add_child(head)
	_sub = Label.new()
	_sub.add_theme_font_override("font", font)
	_sub.add_theme_color_override("font_color", Th.TEXT_DIM)
	_sub.add_theme_font_size_override("font_size", Th.FONT_SMALL)
	box.add_child(_sub)
	var sep := HSeparator.new()
	sep.add_theme_color_override("separator", Th.PANEL_BORDER)
	box.add_child(sep)
	_grid = GridContainer.new()
	_grid.columns = 2
	_grid.add_theme_constant_override("h_separation", 14)
	box.add_child(_grid)
	for row: String in ROWS:
		var k := Label.new()
		k.text = row
		k.add_theme_color_override("font_color", Th.TEXT_DIM)
		k.add_theme_font_size_override("font_size", Th.FONT_SMALL)
		_grid.add_child(k)
		var v := Label.new()
		v.add_theme_font_override("font", font)
		v.add_theme_color_override("font_color", Th.TEXT)
		v.add_theme_font_size_override("font_size", Th.FONT_SMALL + 1)
		_grid.add_child(v)
		_values[row] = v
	visible = false


## Show the state at time t (frame: FrameStore sample) while paused; hide otherwise.
func show_at(wanted: bool, t: float, frame: Dictionary, lanes_at: Dictionary) -> void:
	visible = enabled and wanted
	if not visible or (t == _shown_t and not last.is_empty()):
		return
	_shown_t = t
	var d := Att.describe(frame)
	_title.text = (TF.utc(epoch + t, 1) + " UTC") if not is_nan(epoch) else TF.elapsed(t, 1)
	_sub.text = TF.elapsed(t, 1)
	var rpy: Array = d["rpy_lvlh_deg"]
	var q: Array = d["q_gcrs_body"]
	var w: Array = d["w_body_deg_s"]
	_values["Roll / pitch / yaw"].text = "%+.2f°  %+.2f°  %+.2f°   (vs LVLH)" % [rpy[0], rpy[1], rpy[2]]
	_values["Quaternion"].text = "%+.4f  %+.4f  %+.4f  %+.4f" % [q[0], q[1], q[2], q[3]]
	_values["Body rate"].text = "%+.3f  %+.3f  %+.3f  °/s" % [w[0], w[1], w[2]]
	var pe: Variant = d.get("pointing_err_deg")
	_values["Pointing error"].text = ("%.3f°" % pe) if pe != null else "—"
	_values["Altitude"].text = "%.2f km" % d["altitude_km"]
	var sf: float = d["shadow_factor"]
	_values["Light"].text = "Sunlit" if sf >= 0.999 else ("Eclipse" if sf <= 0.001 else "Penumbra %.0f%%" % (100.0 * sf))
	var over: Array = lanes_at.get("passes", [])
	_values["Passes"].text = ", ".join(over) if not over.is_empty() else "—"
	last = d
	last["utc"] = TF.utc(epoch + t, 3) if not is_nan(epoch) else null
