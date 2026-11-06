extends "res://src/overlays/overlay.gd"
## A vector from the spacecraft's centre of mass: a channel (BODY or GCRS) or a
## derived vector (@sun, @nadir, @velocity, @boresight:<name>). Lengths:
## physical, fixed_px or log (see ADCS/viz/overlays.py).

enum Src { CHANNEL, SUN, NADIR, VELOCITY, BORESIGHT }
enum Scale { LOG, FIXED_PX, PHYSICAL }

var mesh_node: MeshInstance3D
var label: Label3D
var _src := Src.CHANNEL
var _channel := ""
var _boresight := Vector3(0, 0, 1)
var _scale := Scale.LOG
var _px := 90.0
var _px_min := 24.0
var _px_max := 180.0
var _px_per_decade := 40.0
var _units_per_m := 1.0
var _width := 3.0
var _ref := 0.0
var _dir := Vector3.ZERO  ## last drawn direction (BODY), length [px] and magnitude; zero length = hidden
var _length_px := 0.0
var _magnitude := 0.0


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	var src := String(desc["source"])
	match src:
		"@sun": _src = Src.SUN
		"@nadir": _src = Src.NADIR
		"@velocity": _src = Src.VELOCITY
		_:
			if src.begins_with("@boresight:"):
				_src = Src.BORESIGHT
				var b: Array = scene.get("boresights", {}).get(src.substr(11), [0, 0, 1])
				_boresight = Vector3(b[0], b[1], b[2]).normalized()
			else:
				_channel = src
	_scale = {"fixed_px": Scale.FIXED_PX, "physical": Scale.PHYSICAL}.get(String(desc.get("scale", "log")), Scale.LOG)
	_px = float(desc.get("px", 90.0))
	_px_min = float(desc.get("px_min", 24.0))
	_px_max = float(desc.get("px_max", 180.0))
	_px_per_decade = float(desc.get("px_per_decade", 40.0))
	_units_per_m = float(desc.get("units_per_m", 1.0)) if desc.get("units_per_m") != null else 1.0
	_width = float(desc.get("width_px", 3.0))
	_ref = float(desc["ref"]) if desc.get("ref") != null else 0.0
	var color := color_of(desc.get("color"), Color.ORANGE)
	mesh_node = MeshInstance3D.new()
	mesh_node.mesh = unit_arrow()
	mesh_node.material_override = unshaded(color, bool(desc.get("on_top", true)))
	mesh_node.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mesh_node)
	if desc.get("label") != null:
		label = make_label(String(desc["label"]), color)
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		add_child(label)


## Shaft (radius 0.5) along +Y from 0 to 0.78, head (radius 1.2) to 1.0.
static func unit_arrow() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := 12
	for i in n:
		var a0 := TAU * i / n
		var a1 := TAU * (i + 1) / n
		var p0 := Vector3(cos(a0), 0, sin(a0))
		var p1 := Vector3(cos(a1), 0, sin(a1))
		# shaft side
		for v: Vector3 in [p0 * 0.5, p1 * 0.5, p1 * 0.5 + Vector3.UP * 0.78, p0 * 0.5, p1 * 0.5 + Vector3.UP * 0.78, p0 * 0.5 + Vector3.UP * 0.78]:
			st.add_vertex(v)
		# head cone and its base
		for v: Vector3 in [p0 * 1.2 + Vector3.UP * 0.78, p1 * 1.2 + Vector3.UP * 0.78, Vector3.UP,
				p0 * 1.2 + Vector3.UP * 0.78, Vector3.UP * 0.78, p1 * 1.2 + Vector3.UP * 0.78]:
			st.add_vertex(v)
	return st.commit()


## Vector in BODY coordinates (float64), or an empty array if unavailable.
func vector_body(ctx: Dictionary) -> PackedFloat64Array:
	var f: Dictionary = ctx["frame"]
	var v: PackedFloat64Array
	match _src:
		Src.SUN: v = D.sub(f["r_sun"], f["r_sc"])
		Src.NADIR: v = D.scale(f["r_sc"], -1.0)
		Src.VELOCITY: v = f["v_sc"]
		Src.BORESIGHT: return PackedFloat64Array([_boresight.x, _boresight.y, _boresight.z])
		_:
			var store: RefCounted = ctx["store"]
			if not store.has(_channel):
				return PackedFloat64Array()
			v = store.value(_channel, ctx["t"])
			if ctx["frames"].get(_channel, "BODY") == "BODY":
				return v
	return D.mat_t_vec(ctx["rot"], v)


func update(ctx: Dictionary) -> void:
	var v := vector_body(ctx)
	var m := D.norm(v) if v.size() == 3 else 0.0
	if not m > 0.0:  # also NaN
		mesh_node.visible = false
		if label:
			label.visible = false
		_length_px = 0.0
		return
	var px_per_m: float = ctx["focal"] / maxf(ctx["d_craft"], 1e-9)  # screen px per local metre
	var length_px: float
	match _scale:
		Scale.PHYSICAL:
			length_px = m / _units_per_m * px_per_m
		Scale.FIXED_PX:
			length_px = _px
		_:
			var ref := _ref if _ref > 0.0 else _auto_ref(ctx)
			length_px = clampf(_px_min + _px_per_decade * log(m / ref) / log(10.0), _px_min, _px_max)
	var length_m := length_px / px_per_m
	var dir := Vector3(v[0] / m, v[1] / m, v[2] / m)
	mesh_node.visible = true
	mesh_node.basis = align_y(dir, 0.5 * _width / px_per_m, length_m)
	if label:
		label.visible = true
		label.position = dir * length_m * 1.06
		fit_label(label, ctx["focal"], ctx["k_craft"])
	_dir = dir
	_length_px = length_px
	_magnitude = m


func diagnostics() -> Dictionary:
	if _length_px <= 0.0:
		return {"shown": false}
	return {"shown": true, "dir_body": [_dir.x, _dir.y, _dir.z], "length_px": _length_px, "magnitude": _magnitude}


## Log-scale reference: 10th percentile of the magnitude over the log (once).
func _auto_ref(ctx: Dictionary) -> float:
	if _ref > 0.0:
		return _ref
	var store: RefCounted = ctx["store"]
	var src := _channel
	var mags := PackedFloat64Array()
	if store.has(src):
		var stride := maxi(1, store.n_rows / 200)
		for k in range(0, store.n_rows, stride):
			var r: PackedFloat64Array = store.get_row(src, k)
			var mag := sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
			if mag > 0.0 and not is_nan(mag):
				mags.append(mag)
	mags.sort()
	_ref = mags[int(mags.size() * 0.1)] if mags.size() > 0 else 1.0
	return _ref
