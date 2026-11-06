extends Node3D
## Base class for overlays described in scene.json "overlays" (see
## ADCS/viz/overlays.py). The viewer draws only what is described.
##
## Spacecraft-attached overlays are children of the spacecraft node: local
## coordinates are BODY-frame metres, and it already carries the spacecraft's
## attitude and scaled-space scale. A length of n pixels on screen is
## n * d / focal metres locally, where d is the true camera-craft distance;
## the scaled-space factor cancels out.

const D := preload("res://src/core/dmath.gd")
const LABEL_PX := 15.0  ## label text height on screen [px], whatever the camera's FOV

var desc: Dictionary = {}
var scene: Dictionary = {}
var info: Dictionary = {}  ## last frame's diagnostics, for checks (see diagnostics())


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	desc = descriptor
	scene = scene_desc
	name = String(descriptor.get("id", "overlay"))
	visible = bool(descriptor.get("visible", true))


## ctx: {frame, store, world, d_craft, k_craft, focal, rot (body -> GCRS, row-major), cam_pos, cam_rot, t, frames}
func update(_ctx: Dictionary) -> void:
	pass


## Last frame's diagnostics; overlays that update every frame build them only when asked.
func diagnostics() -> Dictionary:
	return info


static func color_of(desc_color: Variant, fallback: Color) -> Color:
	return Color.from_string(String(desc_color), fallback) if desc_color != null else fallback


static func unshaded(color: Color, on_top: bool, transparent: bool = false) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = color
	m.no_depth_test = on_top
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	if transparent or color.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
	if on_top:
		m.render_priority = 10
	return m


## A billboard label that stays readable and in front of everything.
## Label3D's fixed_size keeps text constant in world-per-distance units, so its
## screen size still follows the projection; fit_label() corrects for that.
static func make_label(text: String, color: Color) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.modulate = color
	l.fixed_size = true
	l.font_size = 32
	l.outline_size = 8
	l.no_depth_test = true
	l.render_priority = 20
	l.outline_render_priority = 19
	l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	return l


## Glyph height on screen = pixel_size * font_size * focal * s [px]; the
## billboard keeps the parents' scale s (the spacecraft's scaled-space factor).
static func fit_label(l: Label3D, focal: float, s: float) -> void:
	l.pixel_size = LABEL_PX / (float(l.font_size) * maxf(focal, 1.0) * maxf(s, 1e-12))


## Basis taking local +Y to unit vector v (BODY), with given x/z and y scales.
static func align_y(v: Vector3, xz: float, y: float) -> Basis:
	var yv := v.normalized()
	var helper := Vector3.FORWARD if absf(yv.dot(Vector3.FORWARD)) < 0.9 else Vector3.RIGHT
	var xv := helper.cross(yv).normalized()
	var zv := xv.cross(yv)
	return Basis(xv * xz, yv * y, zv * xz)
