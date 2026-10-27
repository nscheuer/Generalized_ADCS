extends RefCounted
## Colours and sizes of the timeline and state card (one place to restyle).

const PANEL := Color(0.075, 0.085, 0.105, 0.86)
const PANEL_BORDER := Color(1, 1, 1, 0.07)
const TEXT := Color(0.93, 0.95, 0.98)
const TEXT_DIM := Color(0.60, 0.65, 0.72)
const TEXT_FAINT := Color(0.42, 0.46, 0.52)
const ACCENT := Color(0.38, 0.64, 1.0)
const ACCENT_SOFT := Color(0.38, 0.64, 1.0, 0.18)
const TRACK := Color(1, 1, 1, 0.045)
const TICK := Color(1, 1, 1, 0.22)
const TICK_MINOR := Color(1, 1, 1, 0.09)
const HOVER := Color(1, 1, 1, 0.45)
const BUTTON_HOVER := Color(1, 1, 1, 0.08)
const BUTTON_ACTIVE := Color(1, 1, 1, 0.14)
const ERROR := Color(1.0, 0.42, 0.42)
const SUNLIT := Color(1.0, 0.80, 0.34, 0.55)
const PENUMBRA := Color(0.55, 0.50, 0.42, 0.75)
const UMBRA := Color(0.10, 0.12, 0.20, 1.0)
const UMBRA_EDGE := Color(0.32, 0.38, 0.55, 0.9)
const BOOKMARK := Color(1.0, 0.82, 0.40)

const RADIUS := 12.0
const FONT_SIZE := 13
const FONT_SMALL := 11
const FONT_TIME := 16


static func panel_box(scale: float, alpha: float = 1.0) -> StyleBoxFlat:
	var b := StyleBoxFlat.new()
	b.bg_color = Color(PANEL, PANEL.a * alpha)
	b.border_color = PANEL_BORDER
	b.set_border_width_all(1)
	b.set_corner_radius_all(int(RADIUS * scale))
	b.shadow_color = Color(0, 0, 0, 0.35)
	b.shadow_size = int(10 * scale)
	b.set_content_margin_all(10 * scale)
	return b


## The UI font with tabular (fixed-width) digits, so running times don't jitter.
static func font() -> Font:
	var f := FontVariation.new()
	f.base_font = ThemeDB.fallback_font
	f.opentype_features = {TextServerManager.get_primary_interface().name_to_tag("tnum"): 1}
	return f


## A Godot Theme for the timeline's popups, state card and text fields: same font,
## sizes and colours as the custom-drawn dock, opaque panels.
static func ui_theme(scale: float) -> Theme:
	var t := Theme.new()
	t.default_font = font()
	t.default_font_size = int(round(FONT_SIZE * scale))
	for cls: String in ["Label", "Button", "LineEdit", "PopupMenu"]:
		t.set_color("font_color", cls, TEXT)
	t.set_color("font_hover_color", "Button", Color.WHITE)
	t.set_color("font_pressed_color", "Button", ACCENT)
	t.set_color("font_uneditable_color", "LineEdit", TEXT_DIM)
	t.set_color("font_placeholder_color", "LineEdit", TEXT_FAINT)
	t.set_color("caret_color", "LineEdit", ACCENT)
	t.set_color("selection_color", "LineEdit", ACCENT_SOFT)
	var field := StyleBoxFlat.new()
	field.bg_color = Color(0.03, 0.035, 0.045, 1.0)
	field.border_color = Color(ACCENT, 0.6)
	field.set_border_width_all(1)
	field.set_corner_radius_all(int(8 * scale))
	field.set_content_margin_all(6 * scale)
	t.set_stylebox("normal", "LineEdit", field)
	t.set_stylebox("focus", "LineEdit", field)
	var popup := panel_box(scale)
	popup.bg_color = Color(PANEL, 1.0)
	t.set_stylebox("panel", "PopupPanel", popup)
	t.set_stylebox("panel", "PopupMenu", popup)
	t.set_stylebox("panel", "PanelContainer", popup)
	var hover := StyleBoxFlat.new()
	hover.bg_color = BUTTON_HOVER
	hover.set_corner_radius_all(int(6 * scale))
	t.set_stylebox("hover", "PopupMenu", hover)
	t.set_stylebox("hover", "Button", hover)
	return t
