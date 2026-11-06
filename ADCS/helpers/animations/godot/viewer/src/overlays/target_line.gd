extends "res://src/overlays/overlay.gd"
## From the spacecraft toward the active pointing target:
## kind 1 (direction): fixed length on screen along the target direction;
## kind 2 (attitude): along the target attitude's boresight;
## kind 3 (position): to the target point (e.g. a ground site), scaled with the Earth.

var mesh := ImmediateMesh.new()
var mat: StandardMaterial3D
var mesh_node: MeshInstance3D


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	mat = unshaded(color_of(desc.get("color"), Color.LAWN_GREEN), true)
	mesh_node = MeshInstance3D.new()
	mesh_node.mesh = mesh
	mesh_node.custom_aabb = AABB(Vector3.ONE * -1.0e12, Vector3.ONE * 2.0e12)
	add_child(mesh_node)


func update(ctx: Dictionary) -> void:
	var store: RefCounted = ctx["store"]
	var f: Dictionary = ctx["frame"]
	mesh.clear_surfaces()
	if not store.has("target_kind"):
		return
	var t: float = ctx["t"]
	var ks: Array = store.locate(t)
	var kind := int(round(store.get_row("target_kind", ks[0])[0]))
	var tgt: PackedFloat64Array = store.get_row("target", ks[0])
	var cam: PackedFloat64Array = ctx["cam_pos"]
	var r_sc: PackedFloat64Array = f["r_sc"]
	var a: PackedFloat64Array
	var b: PackedFloat64Array
	if kind == 3:
		# Position target: draw with the Earth's scaled-space factor (it lies on/near the Earth).
		var k: float = ctx["world"].info.get("k_earth", 1.0)
		a = D.scale(D.sub(r_sc, cam), k)
		b = D.scale(D.sub(D.scale(PackedFloat64Array([tgt[1], tgt[2], tgt[3]]), 1000.0), cam), k)
	elif kind == 1 or kind == 2:
		var dir: PackedFloat64Array
		if kind == 1:
			dir = D.unit(PackedFloat64Array([tgt[1], tgt[2], tgt[3]]))
		else:
			var bs: PackedFloat64Array = store.get_row("boresight_body", ks[0]) if store.has("boresight_body") else D.v3(0, 0, 1)
			dir = D.unit(D.mat_vec(D.rot_mat(tgt), bs))
		# Spacecraft-relative, fixed on screen: the craft's own scaled-space factor.
		var k_c: float = ctx["world"].info.get("k_craft", 1.0)
		var length := float(desc.get("length_px", 220.0)) * float(ctx["d_craft"]) / float(ctx["focal"])
		a = D.scale(D.sub(r_sc, cam), k_c)
		b = D.scale(D.add(D.sub(r_sc, cam), D.scale(dir, length)), k_c)
	else:
		info = {"kind": kind}
		return
	mesh.surface_begin(Mesh.PRIMITIVE_LINES, mat)
	mesh.surface_add_vertex(D.to_godot(a))
	mesh.surface_add_vertex(D.to_godot(b))
	mesh.surface_end()
	info = {"kind": kind}
