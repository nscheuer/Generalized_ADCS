extends "res://src/overlays/overlay.gd"
## Orbit track (GCRS path around now) or ground track (sub-satellite path fixed
## on the rotating Earth). Not attached to the spacecraft.
##
## The line is built from the log's rows for a chunk of time around now, with
## vertices stored relative to an anchor next to the spacecraft (float64 until
## the subtraction, so the part near the spacecraft keeps millimetre precision).
## Each frame only the node transform changes: camera offset, scaled-space
## factor and, for the ground track, the Earth's rotation. A shader clips the
## line to [now - past_s, now + future_s]. The mesh is rebuilt when now leaves
## the chunk or the spacecraft moves far from the anchor relative to the camera.

const R_EARTH := 6378137.0
const ANCHOR_RATIO := 1000.0  ## rebuild when |spacecraft - anchor| > this x camera distance

var mesh := ArrayMesh.new()
var mat := ShaderMaterial.new()
var mesh_node: MeshInstance3D
var ground := false
var _anchor := PackedFloat64Array()  ## GCRS (orbit) or ITRS (ground) metres
var _t_ref := 0.0
var _chunk := Vector2(INF, -INF)  ## time span the mesh covers


func configure(descriptor: Dictionary, scene_desc: Dictionary) -> void:
	super.configure(descriptor, scene_desc)
	ground = String(desc.get("track", "orbit")) == "ground"
	var c := color_of(desc.get("color"), Color.LIGHT_BLUE)
	mat.shader = preload("res://shaders/track.gdshader")
	mat.set_shader_parameter("color", Vector3(c.r, c.g, c.b))
	mesh_node = MeshInstance3D.new()
	mesh_node.mesh = mesh
	mesh_node.material_override = mat
	mesh_node.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mesh_node.custom_aabb = AABB(Vector3.ONE * -1.0e12, Vector3.ONE * 2.0e12)
	add_child(mesh_node)


## Interpolated track point: Hermite position (GCRS) and slerped Earth
## orientation (ground), so a finite log sample interval does not become a
## visibly faceted orbit line.
func _point_at(store: RefCounted, t: float) -> PackedFloat64Array:
	var ks: Array = store.locate(t)
	var k: int = ks[0]
	var s: float = ks[1]
	var k1 := mini(k + 1, store.n_rows - 1)
	var h: float = store.t(k1) - store.t(k)
	var r0: PackedFloat64Array = store.get_row("r_sc", k)
	var r1: PackedFloat64Array = store.get_row("r_sc", k1)
	var r_km: PackedFloat64Array
	if h > 0.0 and store.has("v_sc"):
		r_km = D.hermite(r0, store.get_row("v_sc", k), r1, store.get_row("v_sc", k1), h, s)
	else:
		r_km = D.add(D.scale(r0, 1.0 - s), D.scale(r1, s))
	var r := D.scale(r_km, 1000.0)
	if not ground:
		return r
	var q: PackedFloat64Array = D.slerp(store.get_row("q_itrs_gcrs", k),
		store.get_row("q_itrs_gcrs", k1), s)
	var qn := sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
	var qu := PackedFloat64Array([q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn])
	return D.scale(D.unit(D.mat_vec(D.rot_mat(qu), r)), R_EARTH + 3000.0)


func _rebuild(store: RefCounted, t: float, anchor: PackedFloat64Array) -> void:
	var past := float(desc.get("past_s", 5400.0))
	var future := float(desc.get("future_s", 1800.0))
	var k0: int = store.locate(t - 2.0 * past)[0]
	var k1: int = mini(store.locate(t + 2.0 * future)[0] + 1, store.n_rows - 1)
	var row_dt: float = (store.t_max() - store.t_min()) / maxf(store.n_rows - 1.0, 1.0)
	var span := past + future
	var spacing := span / maxf(float(desc.get("samples", 600)), 2.0)
	var stride := maxi(1, int(spacing / maxf(row_dt, 1e-9)))
	var subdivisions := 8 if desc.has("samples") else 1
	var verts := PackedVector3Array()
	var uvs := PackedVector2Array()
	_anchor = anchor
	_t_ref = t
	var k := k0
	while true:
		var next_k := mini(k + stride, k1)
		var t0: float = store.t(k)
		var t1: float = store.t(next_k)
		# Subdivide each stored interval with Hermite interpolation. Four
		# segments smooth the visible line without allocating thousands of
		# vertices for every rebuild.
		for j in subdivisions:
			var ti: float = lerpf(t0, t1, float(j) / float(subdivisions))
			var p := _point_at(store, ti)
			verts.append(Vector3(p[0] - anchor[0], p[1] - anchor[1], p[2] - anchor[2]))
			uvs.append(Vector2(ti - _t_ref, 0.0))
		if next_k == k1:
			var p_end := _point_at(store, t1)
			verts.append(Vector3(p_end[0] - anchor[0], p_end[1] - anchor[1], p_end[2] - anchor[2]))
			uvs.append(Vector2(t1 - _t_ref, 0.0))
			break
		k = next_k
	# Ensure the spacecraft's current interpolated position is an actual vertex.
	# Shader clipping alone can end each independently rendered strip on a
	# different pixel when the cutoff falls between sampled vertices.
	if not ground and t >= store.t(k0) and t <= store.t(k1):
		var p_now := _point_at(store, t)
		var insert_at := verts.size()
		var has_now := false
		for i in verts.size():
			if is_equal_approx(uvs[i].x, t - _t_ref):
				has_now = true
				break
			if uvs[i].x > t - _t_ref:
				insert_at = i
				break
		if not has_now:
			verts.insert(insert_at, Vector3(p_now[0] - anchor[0], p_now[1] - anchor[1], p_now[2] - anchor[2]))
			uvs.insert(insert_at, Vector2(t - _t_ref, 0.0))
	mesh.clear_surfaces()
	if verts.size() >= 2:
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = verts
		arrays[Mesh.ARRAY_TEX_UV] = uvs
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_LINE_STRIP, arrays)
	_chunk = Vector2(store.t(k0), store.t(k1))


func update(ctx: Dictionary) -> void:
	var store: RefCounted = ctx["store"]
	if store.n_rows < 2 or not store.has("r_sc"):
		return
	var t: float = ctx["t"]
	var past := float(desc.get("past_s", 5400.0))
	var future := float(desc.get("future_s", 1800.0))
	var f: Dictionary = ctx["frame"]
	var cam: PackedFloat64Array = ctx["cam_pos"]
	var r_now := D.rot_mat(f["q_itrs_gcrs"]) if ground else PackedFloat64Array()
	var here: PackedFloat64Array = f["r_sc"]
	if ground:
		here = D.scale(D.unit(D.mat_vec(r_now, here)), R_EARTH + 3000.0)
	var here_gcrs := D.mat_t_vec(r_now, here) if ground else here
	var stale: bool = _anchor.is_empty() \
		or (t - past < _chunk.x and _chunk.x > store.t_min()) \
		or (t + future > _chunk.y and store.t_max() - _chunk.y > (past + future) / float(desc.get("samples", 600))) \
		or D.norm(D.sub(here, _anchor)) > ANCHOR_RATIO * D.norm(D.sub(here_gcrs, cam))
	if stale:
		_rebuild(store, t, here)
	var m := D.transpose(r_now) if ground else PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
	var k: float = ctx["world"].info.get("k_earth", 1.0)
	var anchor_gcrs := D.mat_vec(m, _anchor)
	mesh_node.transform = Transform3D(D.basis_from(m).scaled(Vector3.ONE * k), D.to_godot(D.scale(D.sub(anchor_gcrs, cam), k)))
	mat.set_shader_parameter("t0", t - past - _t_ref)
	mat.set_shader_parameter("t1", t + future - _t_ref)
	info = {"k": k, "chunk": [_chunk.x, _chunk.y], "anchor_t": _t_ref}
