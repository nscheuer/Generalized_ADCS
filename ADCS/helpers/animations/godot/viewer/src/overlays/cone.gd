extends "res://src/overlays/overlay.gd"
## A sensor's field of view: a translucent cone about its boresight (BODY) with
## its half-angle, both from scene.json; length fixed on screen.

var mesh_node: MeshInstance3D
var _boresight := Vector3.BACK
var _half := 0.1


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	for s: Dictionary in scene_desc.get("sensors", []):
		if s.get("id") == desc.get("sensor"):
			var b: Array = s.get("boresight_body", [0, 0, 1])
			_boresight = Vector3(b[0], b[1], b[2]).normalized()
			_half = float(s.get("fov_half_rad", 0.1))
	var color := color_of(desc.get("color"), Color(0.4, 0.8, 1.0))
	color.a = float(desc.get("opacity", 0.25))
	mesh_node = MeshInstance3D.new()
	mesh_node.mesh = unit_cone(minf(_half, deg_to_rad(89.0)))
	mesh_node.material_override = unshaded(color, false, true)
	mesh_node.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(mesh_node)


## Apex at the origin, axis +Y, slant height 1, open base.
static func unit_cone(half: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := 48
	var r := sin(half)
	var h := cos(half)
	for i in n:
		var a0 := TAU * i / n
		var a1 := TAU * (i + 1) / n
		st.add_vertex(Vector3.ZERO)
		st.add_vertex(Vector3(r * cos(a0), h, r * sin(a0)))
		st.add_vertex(Vector3(r * cos(a1), h, r * sin(a1)))
	return st.commit()


func update(ctx: Dictionary) -> void:
	var px_per_m: float = ctx["focal"] / maxf(ctx["d_craft"], 1e-9)
	var length_m := float(desc.get("length_px", 140.0)) / px_per_m
	mesh_node.basis = align_y(_boresight, length_m, length_m)
	info = {"half_angle_rad": _half, "boresight_body": [_boresight.x, _boresight.y, _boresight.z]}
