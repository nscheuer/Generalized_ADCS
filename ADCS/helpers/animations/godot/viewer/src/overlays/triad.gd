extends "res://src/overlays/overlay.gd"
## Three axes at the spacecraft (X red, Y green, Z blue) of the body, GCRS or
## LVLH frame. LVLH: Z nadir, Y negative orbit normal, X = Y x Z (~velocity).

const Arrow := preload("res://src/overlays/arrow.gd")

var arrows: Array = []


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	for c: Color in [Color(1, 0.25, 0.25), Color(0.3, 1, 0.3), Color(0.35, 0.55, 1)]:
		var a: Node3D = Arrow.new()
		a.configure({"id": "axis", "source": "@nadir", "color": c.to_html(), "scale": "fixed_px",
			"px": float(desc.get("px", 70.0)), "width_px": 2.0, "on_top": bool(desc.get("on_top", true))}, scene_desc)
		add_child(a)
		arrows.append(a)


func update(ctx: Dictionary) -> void:
	var f: Dictionary = ctx["frame"]
	var rot: PackedFloat64Array = ctx["rot"]
	var axes: Array
	match String(desc.get("frame", "body")):
		"gcrs":
			axes = [D.v3(1, 0, 0), D.v3(0, 1, 0), D.v3(0, 0, 1)]
		"lvlh":
			var z := D.unit(D.scale(f["r_sc"], -1.0))
			var y := D.unit(D.scale(D.cross(f["r_sc"], f["v_sc"]), -1.0))
			axes = [D.cross(y, z), y, z]
		_:
			axes = []
	var px_per_m: float = ctx["focal"] / maxf(ctx["d_craft"], 1e-9)
	for i in 3:
		var v: PackedFloat64Array = D.mat_t_vec(rot, axes[i]) if axes.size() == 3 else \
			PackedFloat64Array([1.0 if i == 0 else 0.0, 1.0 if i == 1 else 0.0, 1.0 if i == 2 else 0.0])
		var a: Node3D = arrows[i]
		var dir := Vector3(v[0], v[1], v[2]).normalized()
		a.mesh_node.basis = align_y(dir, 1.0 / px_per_m, float(desc.get("px", 70.0)) / px_per_m)
